import importlib
from omegaconf import OmegaConf


def build_model(cfg):
    arch = cfg.nested.get("arch", "").lower()
    arch = arch.replace("_", "") if arch else ""
    model_name = cfg.model.name
    model_fn_str = f"{arch}{model_name}"

    # Or using getattr on a module
    model_fn = getattr(
        importlib.import_module("mone_pytorch.models.vision_transformer"), 
        model_fn_str
    )
    return model_fn(**OmegaConf.to_container(cfg.model))
