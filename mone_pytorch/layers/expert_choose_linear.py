import torch
import torch.nn as nn

from layers.routing import ExpertsChooseRouter, ExpertsChooseMaskedRouter
from typing import Optional


# def gather_experts(
#     x: torch.Tensor, expert_indices: torch.Tensor, feature_dim: int
# ) -> torch.Tensor:
#     """
#     Gather tokens for each expert based on routing indices.

#     Args:
#         x: Input tensor of shape (B, T, D) or (B, E, T, D)
#         expert_indices: Expert assignment indices of shape (B, C, E)
#         feature_dim: Size of feature dimension to expand indices

#     Returns:
#         Gathered tensor of shape (B, E, C, D)
#     """
#     num_experts = expert_indices.shape[1]

#     # Expand input if needed (B, T, D) -> (B, E, T, D)
#     if x.dim() == 3:
#         x = x.unsqueeze(1).expand(-1, num_experts, -1, -1)

#     # Create gather indices (B, E, C, D)
#     gather_idx = expert_indices[..., None].expand(
#         -1, -1, -1, feature_dim
#     )

#     # Gather tokens (B, E, C, D)
#     return x.gather(dim=2, index=gather_idx)


# def scatter_experts(
#     x_expert: torch.Tensor, expert_indices: torch.Tensor, num_tokens: int, dim: int = 2
# ) -> torch.Tensor:
#     """
#     Scatter expert outputs back to token positions.

#     Args:
#         x_expert: Expert outputs of shape (B, E, C, D)
#         expert_indices: Expert assignment indices of shape (B, C, E)
#         num_tokens: Number of tokens in sequence
#         dim: Dimension to scatter on (default 1 for token dimension)

#     Returns:
#         Scattered tensor of shape (B, T, D) after summing expert outputs
#     """
#     feature_dim = x_expert.shape[-1]

#     # Initialize output tensor
#     output_shape = list(x_expert.shape)
#     output_shape[dim] = num_tokens
#     x_new = x_expert.new_zeros(output_shape)

#     # Create scatter indices
#     scatter_idx = (
#         expert_indices.unsqueeze(-1).expand(-1, -1, -1, feature_dim)
#     )

#     # Scatter and sum across experts
#     x_new.scatter_(dim=dim, index=scatter_idx, src=x_expert)
#     return x_new.sum(dim=1) if dim == 2 else x_new.sum(dim=2)
def gather_experts(x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
    """
    Advanced-indexing version of gather_experts.

    Args:
        x: Input tensor of shape (B, T, D) or (B, E, T, D)
        expert_indices: Expert assignment indices of shape (B, E, C)
        feature_dim: Size of feature dimension (D)
    
    Returns:
        Gathered tensor of shape (B, E, C, D)
    """

    B = x.shape[0]
    E, C = expert_indices.shape[1], expert_indices.shape[2]
    # If x is (B, T, D), we need to match the shape (B, E, T, D) used in the original code.
    # Instead of calling .expand, we rely on the fact that experts are just indexes into T.
    # We'll proceed similarly and assume x has been suitably prepared or we can just do:

    x = x.unsqueeze(1).expand(-1, E, -1, -1)

    # Create indexing tensors for advanced indexing:
    b_idx = torch.arange(B, device=x.device)[:, None, None]  # (B,1,1)
    e_idx = torch.arange(E, device=x.device)[None, :, None]  # (1,E,1)
    c_idx = torch.arange(C, device=x.device)[None, None, :]  # (1,1,C)

    # advanced indexing: x_expert[b,e,c,:] = x[b,e,expert_indices_transposed[b,e,c],:]
    x_expert = x[b_idx, e_idx, expert_indices[b_idx,e_idx,c_idx], :]

    return x_expert


def scatter_experts(
    x_expert: torch.Tensor, 
    expert_indices: torch.Tensor, 
    num_tokens: int
) -> torch.Tensor:
    """
    Scatter expert outputs back to token positions using index_add_ without looping over the batch.

    Args:
        x_expert (torch.Tensor): Expert outputs of shape (B, E, C, D)
        expert_indices (torch.Tensor): Expert assignment indices of shape (B, C, E)
        num_tokens (int): Number of tokens (T)
        dim (int): Dimension to scatter on (default 2 for token dimension)

    Returns:
        torch.Tensor: Scattered tensor of shape (B, T, D)
    """
    B, E, C, D = x_expert.shape
    # Flatten experts and capacity into a single dimension: (B, E*C, D)
    flat_x_expert = x_expert.reshape(B, E*C, D)

    # expert_indices: (B, C, E) -> (B, E, C) -> (B, E*C)
    flat_indices = expert_indices.reshape(B, E*C)
    
    # Create the output tensor: (B, T, D)
    x_new = x_expert.new_zeros(B, num_tokens, D)
    # Flatten (B, T) into a single dimension (B*T):
    x_new_flat = x_new.view(B * num_tokens, D)

    # Compute per-batch offsets to ensure no index collisions:
    # For batch b, indices should be offset by b * num_tokens.
    offsets = torch.arange(B, device=x_expert.device) * num_tokens

    # Add offsets to each batch's indices:
    flat_indices_offset = flat_indices + offsets.unsqueeze(1)  # (B, E*C) + (B,1)

    # Flatten everything for a single index_add_ call:
    flat_indices_offset_flat = flat_indices_offset.reshape(-1)      # (B*E*C,)
    flat_x_expert_flat = flat_x_expert.reshape(-1, D)               # (B*E*C, D)

    # Single index_add_ call to accumulate all experts back to (B*T):
    x_new_flat.index_add_(0, flat_indices_offset_flat, flat_x_expert_flat)

    # Reshape to (B, T, D)
    return x_new


@torch.compile
class ExpertsChooseContract(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.expert_out_features = out_features // num_experts
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self):
        # Expert feature preparation (M, D) -> (E, O, I)
        return self.linear.weight.reshape(
            self.num_experts, self.expert_out_features, self.in_features
        )

    @property
    def bias(self):
        if self.linear.bias is None:
            return None
        # Expert bias preparation (M,) -> (E, O)
        return self.linear.bias.reshape(self.num_experts, self.expert_out_features)

    def forward(
        self,
        x: torch.Tensor, 
        expert_indices: torch.Tensor,
    ) -> torch.Tensor: 
        """
        Forward pass for the ExpertsChooseContract layer.

        Args:
            x: Input tensor of shape (B, T, D)
            expert_indices: Expert assignment indices of shape (B, C, E)

        Returns:
            Output tensor of shape (B, E, C, O)
        """
        # Use gather_experts instead of manual gathering
        selected_x = gather_experts(x, expert_indices)

        # Expert computation
        x_expert = torch.einsum("beci,eoi->beco", selected_x, self.weight)

        # Add bias and apply gating
        if self.bias is not None:
            x_expert = x_expert + self.bias.unsqueeze(0).unsqueeze(2)

        return x_expert


@torch.compile
class ExpertsChooseExpand(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.expert_in_features = in_features // num_experts
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self):
        # Expert feature preparation (O, M) -> (E, O, I)
        return self.linear.weight.reshape(
            self.num_experts, self.out_features, self.expert_in_features
        )

    @property
    def bias(self):
        return self.linear.bias

    def forward(
        self,
        x_expert: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_gate: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor: 
        """
        Forward pass for the ExpertsChooseExpand layer.

        Args:
            x_expert: Expert outputs of shape (B, E, C, I)
            expert_indices: Expert assignment indices of shape (B, C, E)
            expert_gate: Expert gating values of shape (B, C, E)
            num_tokens: Number of tokens in sequence

        Returns:
            Output tensor of shape (B, T, O)
        """
        # Expert computation
        x_expert = torch.einsum("beci,eoi->beco", x_expert, self.weight)

        # Add bias
        if self.bias is not None:
            x_expert = x_expert + self.bias

        # Apply gating
        x_expert = x_expert * expert_gate.unsqueeze(-1)

        # Use scatter_experts to reassemble output
        return scatter_experts(x_expert, expert_indices, num_tokens)

@torch.compile
class ExpertsChooseMaskedContract(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.expert_out_features = out_features // num_experts
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self):
        # Expert feature preparation (M, D) -> (E, O, I)
        return self.linear.weight.reshape(
            self.num_experts, self.expert_out_features, self.in_features
        )

    @property
    def bias(self):
        if self.linear.bias is None:
            return None
        # Expert bias preparation (M,) -> (E, O)
        return self.linear.bias.reshape(self.num_experts, self.expert_out_features)

    def forward(self, x: torch.Tensor, dispatch_mask: torch.Tensor) -> torch.Tensor:
        expert_inputs = torch.einsum("bt...,btec->bec...", x, dispatch_mask)
        expert_outputs = torch.einsum("beci,eoi->beco", expert_inputs, self.weight)
        if self.bias is not None:
            expert_outputs = expert_outputs + self.bias.unsqueeze(0).unsqueeze(2)
        return expert_outputs

@torch.compile
class ExpertsChooseMaskedExpand(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.expert_in_features = in_features // num_experts
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self):
        # Expert feature preparation (O, M) -> (E, O, I)
        return self.linear.weight.reshape(
            self.num_experts, self.out_features, self.expert_in_features
        )

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor, combine_array: torch.Tensor) -> torch.Tensor:
        expert_outputs = torch.einsum("beci,eoi->beco", x, self.weight)
        if self.bias is not None:
            expert_outputs = expert_outputs + self.bias
        combined_outputs = torch.einsum(
            "bec...,btec->bt...", expert_outputs, combine_array
        )
        return combined_outputs


class ExpertsChooseMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.expert_choose_contract = ExpertsChooseContract(
            in_features, out_features, num_experts, bias
        )
        self.expert_choose_expand = ExpertsChooseExpand(
            in_features, out_features, num_experts, bias
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_gate: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        x_expert = self.expert_choose_contract(x, expert_indices, expert_gate)
        x_expert = self.activation(x_expert)
        return self.expert_choose_expand(
            x_expert, expert_indices, expert_gate, num_tokens
        )

def experts_choose_linear(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_gate: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Derive dimensions from inputs
    batch_size, num_tokens, in_features = x.shape
    out_features = weight.size(0)
    num_experts = expert_gate.size(-1)

    # Expert feature preparation (O, I) -> (E, O, I)
    expert_in_features = in_features // num_experts
    w_experts = weight.reshape(num_experts, out_features, expert_in_features)

    # Reshape input (B, T, I) -> (B, E, T, I)
    x = x.reshape(batch_size, num_tokens, num_experts, expert_in_features)

    # Gather tokens for each expert (B, E, T, I) -> (B, E, C, I)
    selected_x = gather_experts(
        x.permute(0, 2, 1, 3), expert_indices, expert_in_features
    )

    # Expert computation (B, E, C, I) @ (E, I, O) -> (B, E, C, O)
    w_experts_t = w_experts.transpose(1, 2)
    x_expert = torch.einsum("beci,eio->beco", selected_x, w_experts_t)

    # Add bias and apply gating
    if bias is not None:
        x_expert = x_expert + bias
    x_expert = x_expert * expert_gate.unsqueeze(-1)

    # Scatter results back
    return scatter_experts(x_expert, expert_indices, num_tokens, dim=1)

class ExpertsChooseLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        router: ExpertsChooseRouter = None,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.router = router

    @property
    def num_experts(self):
        return self.router.num_experts

    @property
    def capacity(self):
        return self.router.capacity

    def forward(self, x: torch.Tensor, c: int = None) -> torch.Tensor:
        expert_gate, expert_indices = self.router.compute_routing_indices(x, c)
        return experts_choose_linear(
            x,
            expert_gate,
            expert_indices,
            self.weight,
            self.bias,
        )


# Test

if __name__ == "__main__":
    import time
    import triton

    capacity = 30
    num_experts = 4
    embed_dim = 768
    contract_dim = 768
    expand_dim = embed_dim
    batch_size = 256
    seq_len = 196

    def benchmark_function(func, inputs, iterations=100):
        times = []
        peaks = []
        for i in range(iterations):
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            func(*inputs)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            peaks.append(torch.cuda.max_memory_allocated() / 1024 ** 3)
        return sum(times) / len(times) * 1000, sum(peaks) / len(peaks)
    
    # run triton benchmark for any function
    def triton_benchmark(func, inputs):
        def benchmark_func():
            return func(*inputs)
        return triton.testing.do_bench(benchmark_func)

    
    # Set up masked router
    masked_router = ExpertsChooseMaskedRouter(embed_dim, num_experts=num_experts).cuda()

    # Set up inputs
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()

    # Test mask router performance with triton
    runtime = triton_benchmark(masked_router, (x, capacity))
    print(f"Masked router latency: {runtime} ms.")

    # Test mask router performance
    latency, peak = benchmark_function(masked_router, (x, capacity))
    print(f"Masked router latency: {latency} ms, peak memory: {peak} GB")

    # Set up masks
    dispatch_mask, combine_array = masked_router(x, c=capacity)

    # Test masked expand performance
    contract_layer = ExpertsChooseMaskedContract(embed_dim, contract_dim, num_experts).cuda()
    c_runtime = triton_benchmark(contract_layer, (x, dispatch_mask))
    print(f"Masked contract latency: {c_runtime} ms.")

    # Test masked contract performance
    c_latency, c_peak = benchmark_function(contract_layer, (x, dispatch_mask))
    print(f"Masked contract latency: {c_latency} ms, peak memory: {c_peak} GB")

    # Contract
    x_contract = contract_layer(x, dispatch_mask)

    # Test masked expand performance
    expand_layer = ExpertsChooseMaskedExpand(contract_dim, embed_dim, num_experts).cuda()
    e_runtime = triton_benchmark(expand_layer, (x_contract, combine_array))
    print(f"Masked expand latency: {e_runtime} ms.")

    # Test masked expand performance
    e_latency, e_peak = benchmark_function(expand_layer, (x_contract, combine_array))
    print(f"Masked expand latency: {e_latency} ms, peak memory: {e_peak} GB")

    # total runtime
    print(f"Total runtime for masked expand and contract: {c_latency + e_latency} ms")

    # Let's clean up
    del x
    del x_contract
    del combine_array
    del dispatch_mask
    del contract_layer
    del expand_layer
    del masked_router
    torch.cuda.empty_cache()

    # Let's try the sparse router
    router = ExpertsChooseRouter(embed_dim, num_experts=num_experts).cuda()

    # Set up inputs
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()

    # Test sparse router performance
    runtime = triton_benchmark(router, (x, capacity))
    print(f"Sparse router latency: {runtime} ms.")

    # Test sparse router performance
    latency, peak = benchmark_function(router, (x, capacity))
    print(f"Sparse router latency: {latency} ms, peak memory: {peak} GB")

    # Set up masks
    expert_gate, expert_indices = router(x, c=capacity)

    # Test sparse contract performance
    contract_layer = ExpertsChooseContract(embed_dim, contract_dim, num_experts).cuda()
    c_runtime = triton_benchmark(contract_layer, (x, expert_indices))
    print(f"Sparse contract latency: {c_runtime} ms.")

    # Test sparse contract performance
    c_latency, c_peak = benchmark_function(contract_layer, (x, expert_indices))
    print(f"Sparse contract latency: {c_latency} ms, peak memory: {c_peak} GB")

    # Contract
    x_contract = contract_layer(x, expert_indices)

    # Test sparse expand performance
    expand_layer = ExpertsChooseExpand(contract_dim, embed_dim, num_experts).cuda()
    e_runtime = triton_benchmark(expand_layer, (x_contract, expert_indices,expert_gate, x.shape[1]))
    print(f"Sparse expand latency: {e_runtime} ms.")

    # Test sparse expand performance
    e_latency, e_peak = benchmark_function(expand_layer, (x_contract, expert_indices,expert_gate, x.shape[1]))
    print(f"Sparse expand latency: {e_latency} ms, peak memory: {e_peak} GB")

    # total runtime
    print(f"Total runtime for sparse expand and contract: {c_latency + e_latency} ms")

    # Let's clean up
    del x_contract
    del expert_gate
    del expert_indices
    del x
    del contract_layer
    del expand_layer
    del router
    torch.cuda.empty_cache()

    # Set up inputs
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()

    # Do baseline linear layer
    linear_layer = nn.Linear(embed_dim, embed_dim, bias=True).cuda()
    runtime = triton_benchmark(linear_layer, (x,))
    print(f"Baseline linear latency: {runtime} ms.")

    # Test baseline linear performance
    latency, peak = benchmark_function(linear_layer, (x,))
    print(f"Baseline linear latency: {latency} ms, peak memory: {peak} GB")

    # Let's clean up
    del x
    del linear_layer
    torch.cuda.empty_cache()

    # Test sparse outputs vs masked outputs
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()
    sparse_router = ExpertsChooseRouter(embed_dim, num_experts=num_experts).cuda()
    masked_router = ExpertsChooseMaskedRouter(embed_dim, num_experts=num_experts).cuda()

    sparse_expert_gate, sparse_expert_indices = sparse_router(x, c=capacity)
    dispatch_mask, combine_array = masked_router(x, c=capacity)

    sparse_x_contract = ExpertsChooseContract(embed_dim, contract_dim, num_experts).cuda()(x, sparse_expert_indices)
    masked_x_contract = ExpertsChooseMaskedContract(embed_dim, contract_dim, num_experts).cuda()(x, dispatch_mask)

    print(f"All close: {torch.allclose(sparse_x_contract, masked_x_contract)}")
    print(f"Max diff: {torch.max(torch.abs(sparse_x_contract - masked_x_contract))}")





