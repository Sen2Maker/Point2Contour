import yaml
from dotted.collection import DottedDict  
import os


def load_and_merge_config(args):

    config_path = args.config
    if args.resume:
        if not args.resume_dir:
            raise ValueError("--resume requires --resume_dir.")
        config_path = os.path.join(args.resume_dir, 'config.yaml')

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config_dict.update(vars(args))

    config_dict['source_config_path'] = args.config

    if args.resume:
        config_dict['log_path'] = args.resume_dir
    else:

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dict['log_path'] = get_unique_log_path(project_root, 'Train_0001')

    opt = DottedDict(config_dict)

    return opt

def get_unique_log_path(root_path: str, logging_root: str) -> str:

    base_dir = os.path.join(root_path, "results")
    os.makedirs(base_dir, exist_ok=True)
    prefix, num = logging_root.rsplit("_", 1)
    num = int(num)
    while True:
        candidate = os.path.join(base_dir, f"{prefix}_{num:04d}")
        if not os.path.exists(candidate):
            return candidate
        num += 1
