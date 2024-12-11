import torch

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from fvcore.nn import ActivationCountAnalysis
    has_fvcore_profiling = True
except ImportError as e:
    FlopCountAnalysis = None
    ActivationCountAnalysis = None
    has_fvcore_profiling = False
    print(f"Failed to import fvcore: {e}")


def profile_fvcore(model, input_size=(3, 224, 224), input_dtype=torch.float32, max_depth=4,
                   batch_size=1, detailed=False, force_cpu=False, other_inputs=None):
    if force_cpu:
        model = model.to('cpu')
    device, _ = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.zeros((batch_size,) + input_size, device=device, dtype=input_dtype)
    if other_inputs is not None:
        example_input = (example_input, *other_inputs)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        print(flop_count_table(fca, max_depth=max_depth))
    return fca, fca.total(), aca, aca.total()