import torch
import torch.nn as nn
from mone_pytorch.layers.routing import Router

from typing import Union, Tuple


class MixtureLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        router: Router = None,
        combine_outputs: bool = True,
        masked_dispatch: bool = False,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.router = router
        self.combine_outputs = combine_outputs
        self.masked_dispatch = masked_dispatch

    @property
    def num_experts(self):
        return self.router.num_experts

    def forward(
        self, x: torch.Tensor, expert_capacity: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_tokens, _ = x.shape

        # get distribution of tokens across experts
        dispatch_mask, combine_array = self.router(x, c=expert_capacity)

        # Reshape x to split features across experts
        # shape: [batch, capacity, num_experts, features_per_expert]
        x_split = x.reshape(batch_size, num_tokens, self.num_experts, -1)

        # shape: [batch, expert_capacity, num_experts, features_per_expert]
        x_split = torch.einsum(
            "b c e f, b t e c -> b e c f",
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

        if self.combine_outputs:
            # shape: [batch, num_tokens, out_features]
            combined_outputs = torch.einsum(
                "b e c p, b t e c -> b t p ...",
                expert_outputs,
                combine_array,
            )
            return combined_outputs
        else:
            return expert_outputs, combine_array


# test mixture linear
if __name__ == "__main__":
    from mone_pytorch.layers.routing import ExpertsChooseRouter

    expert_capacity = 100
    x = torch.randn(32, 100, 128)
    router = ExpertsChooseRouter(dim=128, num_experts=4, bias=True)
    mixture_linear = MixtureLinear(128, 128, router=router)
    print(mixture_linear(x, expert_capacity=expert_capacity))
