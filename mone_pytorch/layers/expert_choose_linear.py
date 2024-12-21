import torch
import torch.nn as nn

from mone_pytorch.layers.routing import ExpertsChooseRouter, ExpertsChooseMaskedRouter
from typing import Optional


def gather_experts(x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (B, E, T, D)
        expert_indices: (B, E, C)
    Returns:
        (B, E, C, D)
    """
    B = x.shape[0]
    E, C = expert_indices.shape[1], expert_indices.shape[2]

    # (B, T, D) -> (B, E, T, D)
    x = x.unsqueeze(1).expand(-1, E, -1, -1)

    # Create index tensors:
    b_idx = torch.arange(B, device=x.device)[:, None, None]  # (B,1,1)
    e_idx = torch.arange(E, device=x.device)[None, :, None]  # (1,E,1)
    c_idx = torch.arange(C, device=x.device)[None, None, :]  # (1,1,C)

    # (B, E, T, D) -> (B, E, C, D)
    x_expert = x[b_idx, e_idx, expert_indices[b_idx, e_idx, c_idx], :]

    return x_expert


def scatter_experts(
    x_expert: torch.Tensor, expert_indices: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """
    Args:
        x_expert: (B, E, C, D)
        expert_indices: (B, C, E)
        num_tokens: T
    Returns:
        (B, T, D)
    """
    B, E, C, D = x_expert.shape

    # (B, E, C, D) -> (B, E*C, D)
    flat_x_expert = x_expert.reshape(B, E * C, D)

    # (B, C, E) -> (B, E*C)
    flat_indices = expert_indices.reshape(B, E * C)

    # Initialize: (B, T, D)
    x_new = x_expert.new_zeros(B, num_tokens, D)

    # (B, T, D) -> (B*T, D)
    x_new_flat = x_new.view(B * num_tokens, D)

    # Offset computation for batches
    offsets = torch.arange(B, device=x_expert.device) * num_tokens

    # (B, E*C) + (B,1) -> (B, E*C)
    flat_indices_offset = flat_indices + offsets.unsqueeze(1)

    # (B, E*C) -> (B*E*C,)
    flat_indices_offset_flat = flat_indices_offset.reshape(-1)
    # (B, E*C, D) -> (B*E*C, D)
    flat_x_expert_flat = flat_x_expert.reshape(-1, D)

    # Accumulate and reshape: (B*T, D) -> (B, T, D)
    x_new_flat.index_add_(0, flat_indices_offset_flat, flat_x_expert_flat)
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
        # Combined einsum operation: (bt...,btec,eoi->beco)
        expert_outputs = torch.einsum("bt...,btec,eoi->beco", x, dispatch_mask, self.weight)
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

    def forward(self, x: torch.Tensor, combine_array: torch.Tensor, dispatch_mask: torch.Tensor = None) -> torch.Tensor:
        if dispatch_mask is not None:
            x = x.reshape(x.shape[0], -1, self.num_experts, self.expert_in_features)
            x = torch.einsum("btei,btec->beci", x, dispatch_mask)

        if self.bias is not None:
            # Append ones to input for bias
            x_homo = torch.cat([x, torch.ones_like(x[...,:1])], dim=-1)
            # Append bias to weight matrix
            w_homo = torch.cat([
                self.weight,
                self.bias.reshape(1, self.out_features).expand(self.num_experts, -1).unsqueeze(-1)
            ], dim=-1)
            # Single combined operation with homogeneous coordinates
            return torch.einsum("beci,eoi,btec->bt...", x_homo, w_homo, combine_array)
        else:
            # Original operation when no bias
            return torch.einsum("beci,eoi,btec->bt...", x, self.weight, combine_array)


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
        x_expert = self.expert_choose_contract(x, expert_indices)
        x_expert = self.activation(x_expert)
        return self.expert_choose_expand(
            x_expert, expert_indices, expert_gate, num_tokens
        )


class ExpertsChooseMaskedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.expert_choose_contract = ExpertsChooseMaskedContract(
            in_features, out_features, num_experts, bias
        )
        self.expert_choose_expand = ExpertsChooseMaskedExpand(
            in_features, out_features, num_experts, bias
        )
        self.activation = activation

    def forward(
        self, x: torch.Tensor, dispatch_mask: torch.Tensor, combine_array: torch.Tensor
    ) -> torch.Tensor:
        x_expert = self.expert_choose_contract(x, dispatch_mask)
        x_expert = self.activation(x_expert)
        return self.expert_choose_expand(x_expert, combine_array)


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

    capacity = 50
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
            peaks.append(torch.cuda.max_memory_allocated() / 1024**3)
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
    print(f"Masked router latency: {runtime:.4f} ms.")

    # Test mask router performance
    latency, peak = benchmark_function(masked_router, (x, capacity))
    print(f"Masked router latency: {latency:.4f} ms, peak memory: {peak:.4f} GB")

    # Set up masks
    dispatch_mask, combine_array = masked_router(x, c=capacity)

    # Test masked expand performance
    contract_layer = ExpertsChooseMaskedContract(
        embed_dim, contract_dim, num_experts
    ).cuda()
    c_runtime = triton_benchmark(contract_layer, (x, dispatch_mask))
    print(f"Masked contract latency: {c_runtime:.4f} ms.")

    # Test masked contract performance
    c_latency, c_peak = benchmark_function(contract_layer, (x, dispatch_mask))
    print(f"Masked contract latency: {c_latency:.4f} ms, peak memory: {c_peak:.4f} GB")

    # Contract
    x_contract = contract_layer(x, dispatch_mask)

    # Test masked expand performance
    expand_layer = ExpertsChooseMaskedExpand(
        contract_dim, embed_dim, num_experts
    ).cuda()
    e_runtime = triton_benchmark(expand_layer, (x_contract, combine_array))
    print(f"Masked expand latency: {e_runtime:.4f} ms.")

    # Test masked expand performance
    e_latency, e_peak = benchmark_function(expand_layer, (x_contract, combine_array))
    print(f"Masked expand latency: {e_latency:.4f} ms, peak memory: {e_peak:.4f} GB")

    # total runtime
    print(
        f"Total runtime for masked expand and contract: {c_latency + e_latency:.4f} ms"
    )

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
    print(f"Sparse router latency: {runtime:.4f} ms.")

    # Test sparse router performance
    latency, peak = benchmark_function(router, (x, capacity))
    print(f"Sparse router latency: {latency:.4f} ms, peak memory: {peak:.4f} GB")

    # Set up masks
    expert_gate, expert_indices = router(x, c=capacity)

    # Test sparse contract performance
    contract_layer = ExpertsChooseContract(embed_dim, contract_dim, num_experts).cuda()
    c_runtime = triton_benchmark(contract_layer, (x, expert_indices))
    print(f"Sparse contract latency: {c_runtime:.4f} ms.")
    print(f"Sparse contract latency: {c_runtime:.4f} ms.")

    # Test sparse contract performance
    c_latency, c_peak = benchmark_function(contract_layer, (x, expert_indices))
    print(f"Sparse contract latency: {c_latency:.4f} ms, peak memory: {c_peak:.4f} GB")

    # Contract
    x_contract = contract_layer(x, expert_indices)

    # Test sparse expand performance
    expand_layer = ExpertsChooseExpand(contract_dim, embed_dim, num_experts).cuda()
    e_runtime = triton_benchmark(
        expand_layer, (x_contract, expert_indices, expert_gate, x.shape[1])
    )
    print(f"Sparse expand latency: {e_runtime:.4f} ms.")

    # Test sparse expand performance
    e_latency, e_peak = benchmark_function(
        expand_layer, (x_contract, expert_indices, expert_gate, x.shape[1])
    )
    print(f"Sparse expand latency: {e_latency:.4f} ms, peak memory: {e_peak:.4f} GB")

    # total runtime
    print(
        f"Total runtime for sparse expand and contract: {c_latency + e_latency:.4f} ms"
    )

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

    # setup masked router
    masked_router = ExpertsChooseMaskedRouter(embed_dim, num_experts=num_experts).cuda()
    dispatch_mask, combine_array = masked_router(x, c=capacity)

    # Test masked MLP
    masked_mlp = ExpertsChooseMaskedMLP(embed_dim, embed_dim, num_experts).cuda()
    masked_mlp_runtime = triton_benchmark(masked_mlp, (x, dispatch_mask, combine_array))
    print(f"Masked MLP latency: {masked_mlp_runtime:.4f} ms.")

    # Test masked MLP performance
    masked_mlp_latency, masked_mlp_peak = benchmark_function(
        masked_mlp, (x, dispatch_mask, combine_array)
    )
    print(
        f"Masked MLP latency: {masked_mlp_latency:.4f} ms, peak memory: {masked_mlp_peak:.4f} GB"
    )

    # Let's clean up
    del x
    del masked_mlp
    del masked_router
    del dispatch_mask
    del combine_array
    torch.cuda.empty_cache()

    x = torch.randn(batch_size, seq_len, embed_dim).cuda()

    # Let's try the sparse router
    router = ExpertsChooseRouter(embed_dim, num_experts=num_experts).cuda()
    expert_gate, expert_indices = router(x, c=capacity)

    # Test sparse MLP
    sparse_mlp = ExpertsChooseMLP(embed_dim, embed_dim, num_experts).cuda()
    sparse_mlp_runtime = triton_benchmark(
        sparse_mlp, (x, expert_indices, expert_gate, x.shape[1])
    )
    print(f"Sparse MLP latency: {sparse_mlp_runtime:.4f} ms.")

    # Test sparse MLP performance
    sparse_mlp_latency, sparse_mlp_peak = benchmark_function(
        sparse_mlp, (x, expert_indices, expert_gate, x.shape[1])
    )
    print(
        f"Sparse MLP latency: {sparse_mlp_latency:.4f} ms, peak memory: {sparse_mlp_peak:.4f} GB"
    )
    print(
        f"Improving latency by {100 - sparse_mlp_latency / masked_mlp_latency * 100:.2f}%"
    )
    print(
        f"Improving peak memory by {100 - sparse_mlp_peak / masked_mlp_peak * 100:.2f}%"
    )

    # Let's clean up
    del x
    del router
    del expert_gate
    del expert_indices
    del sparse_mlp
    torch.cuda.empty_cache()

    x = torch.randn(batch_size, seq_len, embed_dim).cuda()

    # test mlp layer baseline
    mlp_layer = torch.compile(
        nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    ).cuda()

    mlp_layer_runtime = triton_benchmark(mlp_layer, (x,))
    print(f"MLP layer latency: {mlp_layer_runtime:.4f} ms.")

    # test mlp layer performance
    mlp_layer_latency, mlp_layer_peak = benchmark_function(mlp_layer, (x,))
    print(
        f"MLP layer latency: {mlp_layer_latency:.4f} ms, peak memory: {mlp_layer_peak:.4f} GB"
    )

    # Compare mlp layer with sparse mlp
    print(
        f"Improving latency by {100 - mlp_layer_latency / sparse_mlp_latency * 100:.2f}%"
    )
    print(
        f"Improving peak memory by {100 - mlp_layer_peak / sparse_mlp_peak * 100:.2f}%"
    )

    # Let's clean up
    del x
    del mlp_layer
    torch.cuda.empty_cache()

    # Let's test out masked expand standalone
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()
    router = ExpertsChooseMaskedRouter(embed_dim, num_experts=num_experts).cuda()
    dispatch_mask, combine_array = router(x, c=capacity)
    expand_layer = ExpertsChooseMaskedExpand(embed_dim, embed_dim, num_experts).cuda()
    expand_layer_runtime = triton_benchmark(expand_layer, (x, combine_array, dispatch_mask))
    print(f"Masked expand latency: {expand_layer_runtime:.4f} ms.")
    # use benchmark function
    expand_layer_latency, expand_layer_peak = benchmark_function(expand_layer, (x, combine_array, dispatch_mask))
    print(f"Masked expand latency: {expand_layer_latency:.4f} ms, peak memory: {expand_layer_peak:.4f} GB")
