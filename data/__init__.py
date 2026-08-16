import os, sys
from torch.utils.data import DataLoader


current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)


import dataset


def get_dataset(dataset_name):

    if hasattr(dataset, dataset_name):

        return getattr(dataset, dataset_name)

    return None


def get_dataloader(opt, dataset_mode='train'):

    if dataset_mode not in ['train', 'test', 'val']:
        raise NotImplementedError  

    DatasetType = get_dataset(opt.dataset.name)
    if DatasetType is None:
        raise NotImplementedError('Unknown dataset type.')  

    opt.dataset.mode = dataset_mode

    my_dataset = DatasetType(opt.dataset, data_root=opt.data_root)

    opt['dataset_size'] = len(my_dataset)

    ds_collate_fn = None  

    if hasattr(my_dataset, 'collate_fn'):
        ds_collate_fn = my_dataset.collate_fn

    loader_kwargs = {
        'dataset': my_dataset,
        'collate_fn': ds_collate_fn,
        'shuffle': opt.shuffle,
        'batch_size': opt.num_batch,
        'pin_memory': opt.pin_memory,
        'num_workers': opt.num_workers,
    }
    if opt.num_workers > 0:
        loader_kwargs['prefetch_factor'] = opt.prefetch_factor

    my_dataloader = DataLoader(**loader_kwargs)

    return my_dataloader
