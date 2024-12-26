import torch
import torch.nn as nn

from mone_pytorch.layers.routing import ExpertsChooseMaskedRouter
from typing import Optional


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
        if self.linear.bias is not None:
            # Expert bias preparation (M,) -> (E, O)
            return self.linear.bias.reshape(self.num_experts, self.expert_out_features)
        else:
            return None

    def forward(
        self, x: torch.Tensor, dispatch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Deal with case where there is no dispatch of experts
        if dispatch_mask is not None:
            # Combined einsum operation: (bt...,btec,eoi->beco)
            expert_outputs = torch.einsum(
                "bt...,btec,eoi->beco", x, dispatch_mask, self.weight
            )
        else:
            # Combined einsum operation: (bt...,eoi->beto)
            expert_outputs = torch.einsum("bt...,eoi->beto", x, self.weight)
        if self.bias is not None:
            expert_outputs = expert_outputs + self.bias.unsqueeze(0).unsqueeze(2)
        return expert_outputs


def experts_choose_masked_contract(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dispatch_mask: Optional[torch.Tensor] = None,
    num_experts: int = None,
) -> torch.Tensor:
    """Standalone function for experts choose masked contract operation.
    
    Args:
        x: Input tensor of shape (batch, tokens, ..., in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)
        dispatch_mask: Optional mask tensor of shape (batch, tokens, experts, capacity)
        num_experts: Number of experts
    """
    # Reshape weight (M, D) -> (E, O, I)
    expert_out_features = weight.size(0) // num_experts
    weight = weight.reshape(num_experts, expert_out_features, weight.size(1))
    
    # Deal with case where there is no dispatch of experts
    if dispatch_mask is not None:
        # Combined einsum operation: (bt...,btec,eoi->beco)
        expert_outputs = torch.einsum("bt...,btec,eoi->beco", x, dispatch_mask, weight)
    else:
        # Combined einsum operation: (bt...,eoi->beto)
        expert_outputs = torch.einsum("bt...,eoi->beto", x, weight)
    
    if bias is not None:
        # Reshape bias (M,) -> (E, O)
        bias = bias.reshape(num_experts, expert_out_features)
        expert_outputs = expert_outputs + bias.unsqueeze(0).unsqueeze(2)
    
    return expert_outputs


def experts_choose_masked_expand(
    x: torch.Tensor,
    weight: torch.Tensor,
    combine_array: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dispatch_mask: Optional[torch.Tensor] = None,
    num_experts: int = None,
) -> torch.Tensor:
    """Standalone function for experts choose masked expand operation.
    
    Args:
        x: Input tensor of shape (batch, tokens, experts, capacity, in_features) or (batch, experts, capacity, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        combine_array: Combine tensor of shape (batch, tokens, experts, capacity)
        bias: Optional bias tensor of shape (out_features,)
        dispatch_mask: Optional mask tensor of shape (batch, tokens, experts, capacity)
        num_experts: Number of experts
    """
    # Reshape weight (O, M) -> (E, O, I)
    expert_in_features = weight.size(1) // num_experts
    weight = weight.reshape(num_experts, -1, expert_in_features)
    
    # Deal with case where there is no contract layer
    if dispatch_mask is not None:
        x = x.reshape(x.shape[0], -1, num_experts, expert_in_features)
        x = torch.einsum("btei,btec->beci", x, dispatch_mask)

    if bias is not None:
        # Append ones to input for bias
        x_homo = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        # Append bias to weight matrix
        w_homo = torch.cat(
            [
                weight,
                bias.reshape(1, -1).expand(num_experts, -1).unsqueeze(-1),
            ],
            dim=-1,
        )
        # Single combined operation with homogeneous coordinates
        return torch.einsum("beci,eoi,btec->bt...", x_homo, w_homo, combine_array)
    else:
        # Original operation when no bias
        return torch.einsum("beci,eoi,btec->bt...", x, weight, combine_array)


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
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(
        self,
        x: torch.Tensor,
        combine_array: torch.Tensor,
        dispatch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return experts_choose_masked_expand(
            x,
            self.weight,
            combine_array,
            self.bias,
            dispatch_mask,
            num_experts=self.num_experts,
        )


class ExpertsChooseMaskedMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        mlp_ratio: float = 0.5,
        out_features: Optional[int] = None,
        num_experts: int = 4,
        bias: bool = True,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        hidden_features = int(out_features * mlp_ratio)
        self.expert_choose_contract = ExpertsChooseMaskedContract(
            in_features, hidden_features, num_experts, bias
        )
        self.expert_choose_expand = ExpertsChooseMaskedExpand(
            hidden_features, out_features, num_experts, bias
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        combine_array: torch.Tensor,
        dispatch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_expert = self.expert_choose_contract(x, dispatch_mask)
        x_expert = self.activation(x_expert)
        return self.expert_choose_expand(x_expert, combine_array)


# Test

if __name__ == "__main__":
    import time
    import triton

    capacity = 60
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

    # Set up inputs
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()

    # setup masked router
    masked_router = ExpertsChooseMaskedRouter(embed_dim, num_experts=num_experts).cuda()
    dispatch_mask, combine_array = masked_router(x, c=capacity)

    # Test masked MLP
    masked_mlp = ExpertsChooseMaskedMlp(
        embed_dim, mlp_ratio=4, num_experts=num_experts
    ).cuda()
    masked_mlp_runtime = triton_benchmark(masked_mlp, (x, combine_array, dispatch_mask))
    print(f"Masked MLP latency: {masked_mlp_runtime:.4f} ms.")

    # Test masked MLP performance
    masked_mlp_latency, masked_mlp_peak = benchmark_function(
        masked_mlp, (x, combine_array, dispatch_mask)
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

    # Let's clean up
    del x
    del mlp_layer
    torch.cuda.empty_cache()

    # Let's test out masked expand standalone
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()
    router = ExpertsChooseMaskedRouter(embed_dim, num_experts=num_experts).cuda()
    dispatch_mask, combine_array = router(x, c=capacity)
    expand_layer = ExpertsChooseMaskedExpand(embed_dim, embed_dim, num_experts).cuda()
    expand_layer_runtime = triton_benchmark(
        expand_layer, (x, combine_array, dispatch_mask)
    )
    print(f"Masked expand latency: {expand_layer_runtime:.4f} ms.")
    # use benchmark function
    expand_layer_latency, expand_layer_peak = benchmark_function(
        expand_layer, (x, combine_array, dispatch_mask)
    )
    print(
        f"Masked expand latency: {expand_layer_latency:.4f} ms, peak memory: {expand_layer_peak:.4f} GB"
    )
