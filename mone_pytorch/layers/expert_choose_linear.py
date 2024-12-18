import torch
import torch.nn as nn
import torch.nn.functional as F

from mone_pytorch.layers.routing import ExpertsChooseRouter


def experts_choose_linear_expand(
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    expert_gate: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """This performs an expansion of each expert's input dimension"""
    # Derive dimensions from inputs
    batch_size, num_tokens, in_features = x.shape
    out_features = weight.size(0)
    num_experts = expert_gate.size(-1)
    
    # Expert feature preparation (E, O, I) -> (E, I, O)
    expert_features = in_features // num_experts
    w_experts = weight.reshape(num_experts, out_features, expert_features)

    # Input reshaping and padding (B, T, E, I) -> (B, E, T, I)
    x = x.reshape(batch_size, num_tokens, num_experts, expert_features)
    x = F.pad(x, (0, out_features - expert_features))
    x_t = x.permute(0, 2, 1, 3)

    # Prepare indices and gates (B, C, E) -> (B, E, C)
    expert_indices = expert_indices.permute(0, 2, 1)
    expert_gate = expert_gate.permute(0, 2, 1)

    # Token selection (B, E, C) -> (B, E, C, I)
    gather_idx = expert_indices[..., None].expand(-1, -1, -1, out_features)
    selected_x = x_t.gather(dim=2, index=gather_idx)
    selected_x = selected_x[..., :expert_features]

    # Expert computation (E, I, O) @ (B, E, C, I) -> (B, E, C, O)
    w_experts_t = w_experts.transpose(1, 2)
    x_expert = torch.einsum("beci,eio->beco", selected_x, w_experts_t)

    if bias is not None:
        x_expert = x_expert + bias

    # Apply gating (B, E, C, O) * (B, E, C, 1) -> (B, E, C, O)
    x_expert = x_expert * expert_gate[..., None]

    # Reassemble output (B, E, C, O) -> (B, T, O)
    x_t_new = torch.zeros_like(x_t)
    x_t_new.scatter_(dim=2, index=gather_idx, src=x_expert)
    x_new = x_t_new.permute(0, 2, 1, 3)
    x_new = x_new.sum(dim=-2)

    return x_new

def experts_choose_linear_contract(
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    expert_gate: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """This performs a contraction of each expert's input dimension"""
    # Derive dimensions from inputs
    batch_size, num_tokens, in_features = x.shape
    out_features = weight.size(0)
    num_experts = expert_gate.size(-1)
    
    # Expert feature preparation (O, I) -> (E, out_chunk, I)
    expert_features = out_features // num_experts
    w_experts = weight.reshape(num_experts, expert_features, in_features)

    # Prepare indices and gates (B, C, E) -> (B, E, C)
    expert_indices = expert_indices.permute(0, 2, 1)
    expert_gate = expert_gate.permute(0, 2, 1)

    # Token selection (B, T, I) -> (B, E, C, I)
    gather_idx = expert_indices[..., None].expand(-1, -1, -1, in_features)
    x_expanded = x.unsqueeze(1).expand(batch_size, num_experts, num_tokens, in_features)
    selected_x = x_expanded.gather(dim=2, index=gather_idx)

    # Expert computation (B, E, C, I) @ (E, out_chunk, I) -> (B, E, C, out_chunk)
    x_expert = torch.einsum('beci,eoi->beco', selected_x, w_experts)

    # Add bias if present
    if bias is not None:
        b_experts = bias.reshape(num_experts, expert_features)
        x_expert = x_expert + b_experts.unsqueeze(0).unsqueeze(2)

    # Apply gating (B, E, C, out_chunk) * (B, E, C, 1)
    x_expert = x_expert * expert_gate.unsqueeze(-1)

    # Reassemble output (B, E, C, out_chunk) -> (B, T, E, out_chunk)
    x_t_new = torch.zeros(
        (batch_size, num_tokens, num_experts, expert_features), 
        device=x.device, 
        dtype=x.dtype
    )
    token_gather_idx = expert_indices.unsqueeze(-1).expand(-1, -1, -1, expert_features)
    x_t_new.scatter_(dim=1, index=token_gather_idx, src=x_expert)

    return x_t_new.reshape(batch_size, num_tokens, out_features)


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
        return expert_choose_linear_expand(
            self.weight,
            self.bias,
            x,
            expert_gate,
            expert_indices,
        )


# Test

if __name__ == "__main__":
    import time

    x = torch.randn(256, 100, 768).cuda()
    router = ExpertsChooseRouter(768, num_experts=4).cuda()
    layer = ExpertsChooseLinear(768, 768, router=router).cuda()
    print(layer(x, c=33).shape)

    # test performance
    # test latency
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(100):
        start = time.time()
        layer(x, c=50)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    print(f"Average latency: {sum(times) / len(times)} seconds")

    # track memory usage
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB")

    # compare at bfloat16
    layer = ExpertsChooseLinear(768, 768, router=router).cuda()
    layer.to(dtype=torch.bfloat16)
    x = x.to(dtype=torch.bfloat16)
    print(layer(x, c=25).shape)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(100):
        start = time.time()
        layer(x, c=50)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    print(f"Average latency: {sum(times) / len(times)} seconds")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB")
