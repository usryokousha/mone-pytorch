import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import minimize

from typing import Tuple

"""Implementation of the routing algorithm for the MONE model."""


def get_expert_probs(
    router_probs: torch.Tensor, token_mask: torch.Tensor
) -> torch.Tensor:
    """Get probabilities for assigned experts based on logits and token mask"""
    expert_probs = router_probs.gather(2, token_mask.unsqueeze(-1)).squeeze(-1)
    return expert_probs


class EPR(nn.Module):
    """
    Expert Preferred Router for the MONE model.

    This router assigns input tokens to experts based on a capacity distribution
    and learned routing probabilities. It supports jitter noise for improved training stability.

    Args:
        dim (int): The dimension of the input tokens.
        capacity_distribution (List[float]): A list of floats representing the capacity
                                             distribution across experts. Must sum to 1.0.
        dtype (torch.dtype, optional): The data type for the module. Defaults to torch.float32.

    Attributes:
        dim (int): The dimension of the input tokens.
        capacity_distribution (List[float]): The capacity distribution across experts.
        dtype (torch.dtype): The data type for the module.
    """

    def __init__(self, dim: int, capacity_distribution: Tuple[float]):
        super().__init__()
        self.dim = dim
        assert (
            sum(capacity_distribution) == 1.0
        ), "The sum of the capacity distribution must be 1.0"
        self.capacity_distribution = capacity_distribution
        self.num_experts = len(capacity_distribution)
        self.router = nn.Linear(dim, self.num_experts)
        self._init_weights()

    def _init_weights(self):
        # uniform distribution for the router weights
        nn.init.uniform_(self.router.weight, a=-2e-2, b=2e-2)
        nn.init.zeros_(self.router.bias)

    @torch.amp.autocast(enabled=False, device_type="cuda")
    def _compute_router_probs(
        self, input_tokens: torch.Tensor, jitter_noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the router probabilities for the input tokens.
        Keeps computation in float32 for numerical stability.
        """
        logits = self.router(input_tokens)
        if jitter_noise > 0.0:
            noise = (
                torch.randn_like(logits, dtype=logits.dtype, device=logits.device)
                * jitter_noise
            )
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        return probs

    def _assign_tokens_to_experts(
        self, router_probs: torch.Tensor, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """
        Assign tokens to experts based on router probabilities and capacity constraints.

        Args:
            router_probs: Tensor of shape [batch_size, num_tokens, num_experts] containing router probabilities
            num_tokens: Number of tokens to assign
            device: Device to place tensors on

        Returns:
            token_mask: Tensor of shape [batch_size, num_tokens] containing expert assignments
        """
        batch_size = router_probs.shape[0]

        # Initialize token assignments with -1 (unassigned)
        token_mask = torch.full(
            (batch_size, num_tokens), -1, dtype=torch.long, device=device
        )

        # For each expert, assign tokens
        for j in reversed(range(self.num_experts)):
            # Check if any capacity is left for this expert
            capacity = int(math.floor(self.capacity_distribution[j] * num_tokens))
            if capacity == 0:
                continue

            # Get the router probabilities for expert j
            r_j = router_probs[:, :, j]  # [batch_size, num_tokens]

            # Mask already assigned tokens
            r_j = r_j.masked_fill(token_mask != -1, float("-inf"))

            # Get the top-k tokens for this expert
            _, topk_indices = torch.topk(r_j, capacity, dim=1)

            # Assign tokens to the expert
            token_mask[:, topk_indices] = j

        # Assign any remaining unassigned tokens to expert 0
        token_mask[token_mask == -1] = 0

        return token_mask

    def forward(self, input_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_probs = self._compute_router_probs(input_tokens)
        token_mask = self._assign_tokens_to_experts(
            router_probs, input_tokens.shape[1], input_tokens.device
        )
        expert_probs = get_expert_probs(router_probs, token_mask)
        return token_mask, expert_probs


def _exponential_annealing(
    step: int,
    start_step: int,
    max_steps: int,
    min_val: float = 0.1,
    max_val: float = 1.0,
) -> float:
    """Anneal the value exponentially from max_val to min_val over max_steps."""
    # Calculate decay rate to reach min_temp at max_steps with a delay of start_step
    decay_rate = -math.log(min_val) / (max_steps - start_step)
    return max(min_val, max_val * math.exp(-decay_rate * (step - start_step)))


def _cosine_annealing(
    step: int,
    start_step: int,
    max_steps: int,
    min_val: float = 0.1,
    max_val: float = 1.0,
) -> float:
    """Anneal the value cosinely from max_val to min_val over max_steps with a delay of start_step."""
    return max(
        min_val,
        max_val
        * (1 + math.cos(math.pi * (step - start_step) / (max_steps - start_step)))
        / 2,
    )


class SEPR(nn.Module):
    """
    Stochastic Expert Preferred Router (SEPR)

    This router extends the Expert Preferred Router (EPR) by incorporating the
    Gumbel-Softmax trick with a straight-through estimator to enable differentiable
    token-to-expert assignments and expert capacity constraints.

    Args:
        dim (int): The dimension of the input tokens.
        capacity_distribution (List[float]): A list of floats representing the capacity
                                             distribution across experts. Must sum to 1.0.

    Attributes:
        dim (int): The dimension of the input tokens.
        capacity_distribution (torch.Tensor): The capacity distribution across experts.
    """

    def __init__(
        self,
        dim: int,
        capacity_distribution: Tuple[float],
        capacity_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.capacity_loss_weight = capacity_loss_weight
        assert (
            abs(sum(capacity_distribution) - 1.0) < 1e-6
        ), "The sum of the capacity distribution must be 1.0"
        self.capacity_distribution = torch.tensor(capacity_distribution)
        self.num_experts = len(capacity_distribution)
        self.router = nn.Linear(dim, self.num_experts)
        self._init_weights()

    def _init_weights(self):
        # Uniform distribution for the router weights
        nn.init.uniform_(self.router.weight, a=-2e-2, b=2e-2)
        nn.init.zeros_(self.router.bias)

    def _compute_router_probs(
        self, input_tokens: torch.Tensor, temperature: float = 1.0, hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the router probabilities for the input tokens using Gumbel-Softmax.

        Args:
            input_tokens (torch.Tensor): The input tokens of shape [batch_size, seq_length, dim].
            tau (float): Temperature parameter for Gumbel-Softmax.
            hard (bool): Whether to return hard one-hot vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - router_probs: Tensor of shape [batch_size, seq_length, num_experts].
                - logits: The raw logits from the router before applying Gumbel-Softmax.
        """
        logits = self.router(input_tokens)
        if self.training:
            # Training mode: use Gumbel-Softmax with provided temperature
            router_probs = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        else:
            # Inference mode: use deterministic assignments
            indices = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_length]
            router_probs = F.one_hot(indices, num_classes=self.num_experts).type_as(
                logits
            )
        return router_probs, logits

    def forward(
        self, input_tokens: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Stochastic Expert Preferred Router (SEPR).

        Args:
            input_tokens (torch.Tensor): Input tokens of shape [batch_size, seq_length, dim].
            tau (float): Temperature parameter for Gumbel-Softmax.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - token_mask: Tensor of shape [batch_size, seq_length] with expert assignments.
                - expert_probs: Tensor of shape [batch_size, seq_length] with expert probabilities.
                - capacity_loss: Scalar tensor representing the capacity loss (only during training).
        """
        # Compute router probabilities and logits
        router_probs, logits = self._compute_router_probs(
            input_tokens.to(torch.float32), temperature=temperature, hard=True
        )

        # Token assignments (indices of the experts)
        token_mask = torch.argmax(router_probs, dim=-1)

        # Expert probabilities (soft probabilities used for weighting expert outputs)
        soft_probs = F.softmax(logits, dim=-1)
        expert_probs = soft_probs.gather(2, token_mask.unsqueeze(-1)).squeeze(-1)

        # Compute capacity loss only during training
        if self.training:
            batch_size, seq_length, _ = router_probs.shape
            expert_token_counts = router_probs.sum(dim=(0, 1))  # Shape: [num_experts]
            empirical_capacity = expert_token_counts / (batch_size * seq_length)
            capacity_loss = self.capacity_loss_weight * F.mse_loss(
                empirical_capacity,
                self.capacity_distribution.to(empirical_capacity.device),
            )
        else:
            capacity_loss = torch.tensor(0.0, device=input_tokens.device)

        return token_mask, expert_probs, capacity_loss


class NestedCombine(nn.Module):
    """
    Combine the output of the experts.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros((1,), dtype=dtype))
        self.dtype = dtype

    def forward(
        self,
        z: torch.Tensor,
        router_probs: torch.Tensor,
        z_prime: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            z: Output of attention layer.
            z_prime: Output of feedforward layer.
            router_probs: Probabilities assigned by the router.
        """
        if z_prime is None:
            z_prime = 0.0
        return z + (self.alpha * router_probs + 1) * z_prime


def compute_capacity_distribution(e_c, E, delta=2, beta=10):
    """
    Computes the capacity distribution c_i for the MoNE framework.

    Args:
        e_c (float): Effective capacity, a value between 0 and 1.
        E (int): Number of nested experts.
        delta (float): Incentive parameter for larger models (>1). Default is 2.
        beta (float): Entropy regularization parameter (>0). Default is 10.

    Returns:
        c (np.ndarray): Capacity distribution array of shape (E,).
    """
    assert 0 < e_c < 1, "Effective capacity e_c must be between 0 and 1."
    assert E >= 1, "Number of experts E must be at least 1."
    assert delta > 1, "Delta must be greater than 1."
    assert beta > 0, "Beta must be greater than 0."

    # Initial guess for c_i
    c_init = np.full(E, 1.0 / E)

    # Model dimensions d_i (assuming d_i = D / 2^(E - i))
    d_ratios = np.array([1 / 2 ** (E - i - 1) for i in range(E)])

    # Objective function to minimize (negative of the maximization problem)
    def objective(c):
        term1 = -np.sum(c / delta ** np.arange(E))
        term2 = beta * np.sum(
            c * np.log(c + 1e-12)
        )  # Add small epsilon to avoid log(0)
        return term1 + term2

    # Equality constraints
    constraints = [
        {"type": "eq", "fun": lambda c: np.sum(c) - 1},  # Sum of capacities equals 1
        {
            "type": "eq",
            "fun": lambda c: np.sum(c * d_ratios)
            - e_c,  # Effective capacity constraint
        },
    ]

    # Bounds for c_i (between 0 and 1)
    bounds = [(0, 1) for _ in range(E)]

    # Solve the optimization problem
    result = minimize(
        objective,
        c_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "disp": False, "maxiter": 1000},
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    c = result.x
    c = np.clip(c, 0, 1)  # Ensure c_i are within [0, 1]

    # Normalize to ensure sum to 1 due to possible numerical errors
    c /= c.sum()

    return c


class CapacityScheduler:
    def __init__(
        self,
        patch_size: int,
        image_size: int,
        max_epochs: int,
        min_capacity: float = 0.1,
        max_capacity: float = 1.0,
        num_experts: int = 4,
        delta: float = 2,
        beta: float = 10,
        annealing_type: str = "exponential",
    ):
        self.patch_size = patch_size
        self.image_size = image_size
        self.max_epochs = max_epochs
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.num_experts = num_experts
        self.delta = delta
        self.beta = beta
        self.annealing_type = annealing_type
        self.start_epoch = 0
        self.epoch = 0

    def update(self) -> float:
        # Update the epoch
        self.epoch += 1

        # Compute the effective capacity for the current step
        if self.annealing_type == "exponential":
            self.effective_capacity = _exponential_annealing(
                self.epoch,
                self.start_epoch,
                self.max_epochs,
                self.min_capacity,
                self.max_capacity,
            )
        elif self.annealing_type == "cosine":
            self.effective_capacity = _cosine_annealing(
                self.epoch,
                self.start_epoch,
                self.max_epochs,
                self.min_capacity,
                self.max_capacity,
            )

        # Compute the capacity distribution based on new effective capacity
        self.capacity_distribution = compute_capacity_distribution(
            self.effective_capacity, self.num_experts, self.delta, self.beta
        )

        return self.capacity_distribution, self.adjusted_patch_size

    @property
    def adjusted_patch_size(self) -> int:
        # Compute the patch size based on the effective capacity
        return max(1, int(self.patch_size * self.effective_capacity))


if __name__ == "__main__":
    # Parameters
    e_c = 0.6  # Effective capacity (between 0 and 1)
    E = 3  # Number of experts
    delta = 2  # Incentive parameter (>1)
    beta = 10  # Entropy regularization parameter (>0)

    # Compute capacity distribution
    capacity_distribution = compute_capacity_distribution(e_c, E, delta, beta)

    print("Capacity Distribution c_i:")
    for i, c_i in enumerate(capacity_distribution):
        print(f"Expert {i+1}: c_{i+1} = {c_i:.4f}")

    # Verify constraints
    print("\nVerification:")
    print(f"Sum of c_i: {np.sum(capacity_distribution):.4f} (should be 1.0)")
    effective_capacity = np.sum(
        capacity_distribution * np.array([2 ** (E - i - 1) for i in range(E)])
    ) / (2 ** (E - 1))
    print(f"Effective Capacity e_c: {effective_capacity:.4f} (should be {e_c})")

    import time

    def benchmark(router, input_tokens):
        start_time = time.time()
        router(input_tokens)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        end_time = time.time()
        return end_time - start_time

    # Define input tokens
    batch_size = 32
    num_tokens = 100
    dim = 128
    input_tokens = torch.randn(batch_size, num_tokens, dim).cuda()

    # Capacity distribution from above

    # Initialize router
    router = EPR(dim=dim, capacity_distribution=capacity_distribution).cuda()

    # Forward pass
    token_mask, router_probs = router(input_tokens)

    print("Token Mask:", token_mask.cpu())
    print("Router Probabilities:", router_probs.cpu())

    # benchmark
    print(f"Execution time EPR: {benchmark(router, input_tokens):.6f} seconds")

    router = SEPR(dim=dim, capacity_distribution=capacity_distribution).cuda()

    token_mask, router_probs, capacity_loss = router(input_tokens)

    print("Token Mask:", token_mask.cpu())
    print("Router Probabilities:", router_probs.cpu())
    print("Capacity Loss:", capacity_loss)

    # benchmark
    print(f"Execution time SEPR: {benchmark(router, input_tokens):.6f} seconds")

    capacity_scheduler = CapacityScheduler(
        patch_size=16,
        image_size=224,
        max_epochs=100,
        min_capacity=0.5,
        max_capacity=1.0,
        num_experts=4,
        delta=2,
        beta=10,
        annealing_type="exponential",
    )
    # Let's look at effective capacity and patch size over epochs by graphing
    import matplotlib.pyplot as plt

    effective_capacity_list = []
    patch_size_list = []
    for _ in range(100):
        capacity_distribution, patch_size = capacity_scheduler.update()
        effective_capacity_list.append(capacity_scheduler.effective_capacity)
        patch_size_list.append(patch_size)
    # Plot the results with different graphs
    # Label the axes
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_ylabel("Effective Capacity")
    ax2.set_ylabel("Adjusted Patch Size")
    ax1.set_xlabel("Epoch")
    ax2.set_xlabel("Epoch")
    ax1.plot(effective_capacity_list, label="Effective Capacity")
    ax2.plot(patch_size_list, label="Adjusted Patch Size")
    ax1.legend()
    ax2.legend()
    plt.show()
