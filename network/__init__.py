import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

import model


def define_model(opt):
    if hasattr(model, opt.model.type):
        return getattr(model, opt.model.type)(opt)
    else:
        raise ValueError(
            f"Unknown model type: {opt.model.type}")
