import json as _json
import os as _os

import torch.utils.data as _data
from torch.utils.data._utils.collate import default_collate as _default_collate


def safe_collate(batch):
    r"""
    Savely select batches/skip errors in file loading.
    """
    return _default_collate([b for b in batch if b])


class ETDataLoader(_data.DataLoader):

    def __init__(self, **kw):
        super(ETDataLoader, self).__init__(**kw)

    @classmethod
    def new(cls, **kw):
        loader_kw = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }

        for k in loader_kw.keys():
            loader_kw[k] = kw.get(k, loader_kw.get(k))

        if kw.get('use_ddp'):
            if kw.get('mode') == 'train':
                loader_kw['sampler'] = _data.distributed.DistributedSampler(loader_kw['dataset'])
            loader_kw['num_workers'] = (loader_kw['num_workers'] + kw['num_gpus'] - 1) // kw['num_gpus']
            loader_kw['batch_size'] = loader_kw['batch_size'] // kw['num_gpus']

        return cls(collate_fn=safe_collate, **loader_kw)


class ETDataset(_data.Dataset):
    def __init__(self, mode='init', limit=float('inf')):
        self.mode = mode
        self.limit = limit
        self.dataspecs = {}
        self.indices = []

    def load_index(self, dataset_name, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append([dataset_name, file])

    def _load_indices(self, dataset_name, files, **kw):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        for file in files:
            if len(self) >= self.limit:
                break
            self.load_index(dataset_name, file)

        if kw.get('verbose', True):
            print(f'{dataset_name}, {self.mode}, {len(self)} Indices Loaded')

    def __getitem__(self, index):
        r"""
        Logic to load one file and send to model. The mini-batch generation will be handled by Dataloader.
        Here we just need to write logic to deal with single file.
        """
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    def transforms(self, **kw):
        return None

    def add(self, files, debug=True, **kw):
        r"""
        An extra layer for added flexibility.
        """
        self.dataspecs[kw['name']] = kw
        self._load_indices(dataset_name=kw['name'], files=files, debug=debug)

    @classmethod
    def pool(cls, args, dataspecs, split_key=None, load_sparse=False):
        r"""
        This method takes multiple dataspecs and pools the first splits of all the datasets.
        So that we can train one single model on all the datasets. It will automatically refer correct data files,
            no need to move files in single folder.
        """
        all_d = []
        for dspec in dataspecs:
            for split in _os.listdir(dspec['split_dir']):
                split = _json.loads(open(dspec['split_dir'] + _os.sep + split).read())
                if load_sparse:
                    for file in split[split_key]:
                        if len(all_d) >= args['load_limit']:
                            break
                        d = cls(mode=split_key)
                        d.add(files=[file], debug=False, **dspec)
                        all_d.append(d)
                    if args['verbose']:
                        print(f'{len(all_d)} sparse dataset loaded.')
                else:
                    if len(all_d) <= 0:
                        all_d.append(cls(mode=split_key, limit=args['load_limit']))
                    all_d[0].add(files=split[split_key], debug=args['verbose'], **dspec)
                """
                Pooling only works with 1 split at the moment.
                """
                break

        return all_d
