import importlib


def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate an object from a config dict / OmegaConf node.

    Supports both styles used in this repo:
    - {"target": "path.to.Class", "params": {...}}
    - {"class_path": "path.to.Class", "init_args": {...}}
    """
    if config is None:
        return None

    if "target" in config:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))

    if "class_path" in config:
        return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

    raise KeyError("Expected key `target` or `class_path` to instantiate.")

