import torch
import torch.nn as nn
import torch.nn.functional as F

from mone_pytorch.layers.routing import ExpertsChooseRouter

class ExpertsChooseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, router: ExpertsChooseRouter = None, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.router = router

    @property
    def num_experts(self):
        return self.router.num_experts

    @property
    def capacity(self):
        return self.router.capacity

    def forward(self, x: torch.Tensor, c: int = None) -> torch.Tensor:
        batch_size, num_tokens, in_features = x.shape
        # shape of expert_indices is (batch_size, capacity, num_experts)
        expert_probs = self.router._compute_router_probs(x)
        expert_gate, expert_indices = torch.topk(expert_probs, k=c, dim=1)

        # reshape weight to include experts
        expert_features = self.in_features // self.num_experts
        w_experts = self.weight.reshape(self.num_experts, -1, expert_features)

        # pad input in case output dimension is greater than input dimension
        x = x.reshape(batch_size, num_tokens, self.num_experts, -1)
        x = F.pad(x, (0, self.out_features - expert_features))

        # prepare indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=x.device)[:, None, None]
        expert_indices = expert_indices.unsqueeze(-1)  # add dim for features
        
        # use loop to save memory (but with advanced indexing)
        for i in range(self.num_experts):
            # directly index using batch indices and expert indices
            selected_x = x[batch_indices, expert_indices[:, :, i], i, :expert_features].squeeze(-2)
            x_expert = F.linear(selected_x, w_experts[i], self.bias)
            
            # directly assign results using the same indexing
            x[batch_indices.squeeze(-1), expert_indices[:, :, i].squeeze(-1), i] = (
                x_expert * expert_gate[:, :, i, None]
            )

        x = x.sum(dim=-2)

        return x
    
# Test 

if __name__ == "__main__":
    import time
    x = torch.randn(256, 100, 768).cuda()
    router = ExpertsChooseRouter(768, num_experts=4).cuda()
    layer = ExpertsChooseLinear(768, 768, router=router).cuda()

    # test performance
    # test latency
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(10):
        start = time.time()
        layer(x, c=10)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    print(f"Average latency: {sum(times) / len(times)} seconds")

    # track memory usage
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB")


