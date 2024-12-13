import torch
import torch.nn as nn
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
            "b e t f, e p f -> b e t p",
            x_split,
            self.weight.reshape(
                self.num_experts,
                self.out_features,
                self.in_features // self.num_experts,
            ),
        )
        # combine the expert outputs
        combined_outputs = torch.einsum(
            "b e t p, b t e -> b t p",
            expert_outputs,
            router_probs,
        )
        return combined_outputs

    def forward_sparse(self, x: torch.Tensor, expert_capacity: int) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape

        # get distribution of tokens across experts
        dispatch_mask, combine_array = self.router(x, c=expert_capacity)

        # Reshape x to split features across experts
        # shape: [batch, capacity, num_experts, features_per_expert]
        x_split = x.reshape(batch_size, num_tokens, self.num_experts, -1)

        # shape: [batch, expert_capacity, num_experts, features_per_expert]
        x_split = torch.einsum(
            "b t e f, b t e c -> b e c f",
            x_split,
            dispatch_mask,
        )

        # apply the linear transformation
        expert_outputs = torch.einsum(
            "b e c f, e p f -> b e c p",
            x_split,
            self.weight.reshape(
                self.num_experts,
                self.out_features,
                self.in_features // self.num_experts,
            ),
        )
        if self.bias is not None:
            expert_outputs += self.bias.reshape(1, 1, self.out_features)

            # shape: [batch, num_tokens, out_features]
        combined_outputs = torch.einsum(
            "b e c p, b t e c -> b t p ...",
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

    expert_capacity = 80
    x = torch.randn(32, 100, 128)
    router = ExpertsChooseRouter(dim=128, num_experts=8, bias=True)
    mixture_linear = MixtureLinear(128, 128, router=router, return_metrics=True)
    output, metrics = mixture_linear(x, expert_capacity=expert_capacity)
    print(output.shape)
    print(metrics)

    # profile latency on gpu
    import time

    times = []
    for i in range(100):
        start = time.time()
        mixture_linear(x, expert_capacity=expert_capacity)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    print(f"Time taken: {sum(times)/100:.6f} seconds")
