import torch
import torch.nn as nn
from torch.nn import functional as F
from mone_pytorch.layers.routing import Router

from typing import Union, Dict

@torch.compile
class MixtureLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        router: Router = None,
        return_metrics: bool = False,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.router = router
        self.return_metrics = return_metrics

    @property
    def num_experts(self):
        return self.router.num_experts

    def _get_metrics(self, dispatch_mask: torch.Tensor, combine_array: torch.Tensor):
        batch_size, num_tokens = dispatch_mask.shape[-2:]
        num_tokens_dispatched = dispatch_mask.sum()

        # router confidence is the average confidence of the tokens that were dispatched
        router_confidence = combine_array.sum() / num_tokens_dispatched
        num_tokens_dispatched_somewhere = torch.max(
            dispatch_mask.squeeze(-1), dim=(-1)
        ).values.sum()
        total_tokens = batch_size * num_tokens

        # fraction of dropped tokens is the fraction of tokens that were not dispatched anywhere
        fraction_of_dropped_tokens = (
            1.0 - num_tokens_dispatched_somewhere / total_tokens
        )
        return {
            "router_confidence": router_confidence,
            "fraction_of_dropped_tokens": fraction_of_dropped_tokens,
        }

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        # get distribution of tokens across experts
        router_probs = self.router._compute_router_probs(x)
        # reshape x to split features across experts
        x_split = x.reshape(batch_size, num_tokens, self.num_experts, -1).permute(
            0, 2, 1, 3
        )
        # apply the linear transformation
        expert_outputs = torch.einsum(
            "b e t f, e p f -> b t e p",
            x_split,
            self.weight.reshape(
                self.num_experts,
                self.out_features,
                self.in_features // self.num_experts,
            ),
        )
        # combine the expert outputs
        combined_outputs = torch.einsum(
            "b t e p, b t e -> b t p",
            expert_outputs,
            router_probs,
        )
        return combined_outputs

    def forward_sparse(self, x: torch.Tensor, expert_capacity: int) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        dispatch_mask, combine_array = self.router(x, c=expert_capacity)

        # Reshape x to split features across experts
        # shape: [batch, capacity, num_experts, features_per_expert]
        x_split = x.reshape(batch_size, num_tokens, self.num_experts, -1)

        # apply the linear transformation
        expert_outputs = torch.einsum(
            "b t e f, b t e c, e p f -> b e c p",
            x_split,
            dispatch_mask,
            self.weight.reshape(
                self.num_experts,
                self.out_features,
                self.in_features // self.num_experts,
            ),
        )

        if self.bias is not None:
            expert_outputs = expert_outputs + self.bias.reshape(1, 1, -1)

        # shape: [batch, num_tokens, out_features]
        combined_outputs = torch.einsum(
            "b e c p, b t e c -> b t p",
            expert_outputs,
            combine_array,
        )

        if self.return_metrics:
            return combined_outputs, self._get_metrics(dispatch_mask, combine_array)
        return combined_outputs

    def forward(
        self, x: torch.Tensor, expert_capacity: int
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if expert_capacity is None or expert_capacity == x.shape[1]:
            return self.forward_full(x)
        else:
            return self.forward_sparse(x, expert_capacity)


# test mixture linear
if __name__ == "__main__":
    from mone_pytorch.layers.routing import ExpertsChooseRouter

    expert_capacity = 100
    x = torch.randn(256, 100, 768).cuda()
    router = ExpertsChooseRouter(dim=768, num_experts=16, bias=True).cuda()
    mixture_linear = MixtureLinear(768, 768, router=router, return_metrics=True).cuda()
    print(mixture_linear(x, expert_capacity=expert_capacity).shape)

    # test sparse performance
    # get peak memory usage 
    import time
    import torch
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(10):
        output_sparse = mixture_linear.forward_sparse(x, expert_capacity=expert_capacity)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    print(f"Time taken: {sum(times) / len(times)} seconds")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB")

    # test dense performance
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    times = []
    for i in range(10):
        output_dense = mixture_linear.forward_full(x)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    print(f"Time taken: {sum(times) / len(times)} seconds")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB")

    # compare regular linear
    linear = nn.Linear(768, 768, bias=True).cuda()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    times = []
    for i in range(10):
        output_linear = linear(x)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    print(f"Time taken: {sum(times) / len(times)} seconds")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3} GB")
