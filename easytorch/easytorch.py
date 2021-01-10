import json as _json
import os as _os
from argparse import ArgumentParser as _AP

from typing import List as _List, Union as _Union, Callable as _Callable

from easytorch.data import datautils as _du
from easytorch.vision import plotter as _logutils
import easytorch.config as _cf
import torch.distributed as _dist
import torch.multiprocessing as _mp
import warnings as _warn
import functools as _functools

_sep = _os.sep


def job(gpu, et_obj, dspec, dataset_cls, trainer_cls, data_splitter):
    world_size = et_obj.args['world_size']
    if not world_size:
        world_size = et_obj.args['num_gpus'] * et_obj.args['num_nodes']
    world_rank = et_obj.args['node_rank'] * et_obj.args['num_gpus'] + gpu
    _dist.init_process_group(backend=et_obj.args['dist_backend'],
                             init_method=et_obj.args['dist_url'],
                             world_size=world_size, rank=world_rank)
    et_obj._run(dspec, dataset_cls, trainer_cls, data_splitter)


def test(i, a):
    print(i, a)


class EasyTorch:
    _MODES_ = ['test', 'train']
    _MODE_ERR_ = \
        'argument *** phase *** is required and must be passed to either' \
        '\n1). EasyTorch(..,phase=<value>,..)' \
        '\n2). runtime arguments 2). python main.py -ph <value> | ' \
        f'\nPossible values are:{_MODES_}' \
        '\n\t- train (runs all train, validation, and test steps)' \
        '\n\t- test (only runs test step; either by picking best saved model, ' \
        '\n\tor loading provided weights in pretrained_path argument) '

    def __init__(self, dataspecs: _List[dict],
                 args: _Union[dict, _AP] = _cf.default_args,
                 phase: str = None,
                 batch_size: int = None,
                 epochs: int = None,
                 learning_rate: float = None,
                 gpus: _List[int] = None,
                 pin_memory: bool = None,
                 num_workers: int = None,
                 dataset_dir: str = None,
                 load_limit: int = None,
                 log_dir: str = None,
                 pretrained_path: str = None,
                 verbose: bool = None,
                 seed: int = None,
                 force: bool = None,
                 patience: int = None,
                 load_sparse: bool = None,
                 num_folds: int = None,
                 split_ratio: _List[float] = None,
                 **kw):

        self._init_args_(args)
        if phase: self.args.update(phase=phase)
        if batch_size: self.args.update(batch_size=batch_size)
        if epochs: self.args.update(epochs=epochs)
        if learning_rate: self.args.update(learning_rate=learning_rate)
        if gpus: self.args.update(gpus=gpus)
        if pin_memory: self.args.update(pin_memory=pin_memory)
        if num_workers: self.args.update(num_workers=num_workers)
        if dataset_dir: self.args.update(dataset_dir=dataset_dir)
        if load_limit: self.args.update(load_limit=load_limit)
        if log_dir: self.args.update(log_dir=log_dir)
        if pretrained_path: self.args.update(pretrained_path=pretrained_path)
        if verbose: self.args.update(verbose=verbose)
        if seed: self.args.update(seed=seed)
        if force: self.args.update(force=force)
        if patience: self.args.update(patience=patience)
        if load_sparse: self.args.update(load_sparse=load_sparse)
        if num_folds: self.args.update(num_folds=num_folds)
        if split_ratio: self.args.update(split_ratio=split_ratio)
        self.args.update(**kw)
        self._init_dataspecs_(dataspecs)

        assert (self.args.get('phase') in self._MODES_), self._MODE_ERR_

        self.args.update(verbose=self.args.get('verbose', True))
        self.args.update(gpus=self.args.get('gpus', []))

        if self.args['verbose'] and len(self.args['gpus']) > _cf.num_gpus:
            _warn.warn(f"Number of GPUs provided: {len(self.args['gpus'])}, but only {_cf.num_gpus} available.\n")
            self.args['gpus'] = list(range(_cf.num_gpus))

        if self.args['verbose'] and len(self.args['gpus']) > 0 and not _cf.cuda_available:
            _warn.warn(f"arg '-gpus' {self.args['gpus']} provided by default "
                       f"but cuda not available. Using CPU(slow).\n")

        self._init_ddp_()

        # self.args = _etutils.FrozenDict(self.args)

    def _init_args_(self, args):
        if isinstance(args, _AP):
            self.args = vars(args.parse_args())
        elif isinstance(args, dict):
            self.args = {**args}
        else:
            raise ValueError('2nd Argument of EasyTorch could be only one of :ArgumentParser, dict')

    def _init_dataspecs_(self, dataspecs):
        """
        Need to add -data(base folder for dataset) to all the directories in dataspecs.
        THis makes it flexible to access dataset from arbitrary location.
        """
        self.dataspecs = [{**dspec} for dspec in dataspecs]
        for dspec in self.dataspecs:
            for k in dspec:
                if 'dir' in k:
                    dspec[k] = _os.path.join(self.args['dataset_dir'], dspec[k])

    def _init_ddp_(self):
        if all([_cf.cuda_available, _cf.num_gpus > 1, len(self.args['gpus']) > 1]):
            self.args['use_ddp'] = True
            self.args['num_gpus'] = len(self.args['gpus'])
        else:
            self.args['use_ddp'] = False

    def _get_train_dataset(self, split, dspec, dataset_cls):
        r"""
        Load the train data from current fold/split.
        """
        train_dataset = dataset_cls(mode='train', limit=self.args['load_limit'])
        train_dataset.add(files=split.get('train', []), debug=self.args['verbose'], **dspec)
        return train_dataset

    def _get_validation_dataset(self, split, dspec, dataset_cls):
        r"""
        Load the validation data from current fold/split.
        """
        val_dataset = dataset_cls(mode='eval', limit=self.args['load_limit'])
        val_dataset.add(files=split.get('validation', []), debug=self.args['verbose'], **dspec)
        return val_dataset

    def _get_test_dataset(self, split, dspec, dataset_cls):
        r"""
        Load the test data from current fold/split.
        If -sp/--load-sparse arg is set, we need to load one image in one dataloader.
        So that we can correctly gather components of one image(components like output patches)
        """
        test_dataset_list = []
        if self.args.get('load_sparse'):
            for f in split.get('test', []):
                if len(test_dataset_list) >= self.args['load_limit']:
                    break
                test_dataset = dataset_cls(mode='eval', limit=self.args['load_limit'])
                test_dataset.add(files=[f], debug=False, **dspec)
                test_dataset_list.append(test_dataset)
            if self.args['verbose']:
                print(f'{len(test_dataset_list)} sparse dataset loaded.')
        else:
            test_dataset = dataset_cls(mode='eval', limit=self.args['load_limit'])
            test_dataset.add(files=split.get('test', []), debug=self.args['verbose'], **dspec)
            test_dataset_list.append(test_dataset)
        return test_dataset_list

    def _run(self, dspec, dataset_cls, trainer_cls,
             data_splitter: _Callable = _du.init_kfolds_):
        trainer = trainer_cls(self.args)

        trainer.cache['log_dir'] = self.args['log_dir'] + _sep + dspec['name']
        if _du.create_splits_(trainer.cache['log_dir'], dspec):
            data_splitter(dspec=dspec, args=self.args)

        """
        We will save the global scores of all folds if any.
        """
        global_score = trainer.new_metrics()
        global_averages = trainer.new_averages()
        trainer.cache['global_test_score'] = []

        """
        The easytorch.metrics.Prf1a() has Precision,Recall,F1,Accuracy,and Overlap implemented.
         We use F1 as default score to monitor while doing validation and save best model.
         And we will have loss returned by easytorch.metrics.Averages() while training.
        """
        trainer.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
        trainer.cache.update(monitor_metric='f1', metric_direction='maximize')

        """
        reset_dataset_cache() is an intervention to set any specific needs for each dataset. For example:
            - custom log_dir
            - Monitor some other metrics
            - Set metrics direction differently.
        """
        trainer.reset_dataset_cache()

        """
        Run for each splits.
        """
        _os.makedirs(trainer.cache['log_dir'], exist_ok=True)
        for split_file in _os.listdir(dspec['split_dir']):
            split = _json.loads(open(dspec['split_dir'] + _sep + split_file).read())

            """
            Experiment id is split file name. For the example of k-fold.
            """
            trainer.cache['experiment_id'] = split_file.split('.')[0]
            trainer.cache['checkpoint'] = trainer.cache['experiment_id'] + '.pt'
            trainer.cache.update(best_epoch=0, best_score=0.0)
            if trainer.cache['metric_direction'] == 'minimize':
                trainer.cache['best_score'] = 1e11

            trainer.check_previous_logs()
            trainer.init_nn()

            """
            Clear cache to save scores for each fold
            """
            trainer.cache.update(training_log=[], validation_log=[], test_score=[])

            """
            An intervention point if anyone wants to change things for each fold.
            """
            trainer.reset_fold_cache()

            """###########  Run training phase ########################"""
            if self.args['phase'] == 'train':
                trainset = self._get_train_dataset(split, dspec, dataset_cls)
                valset = self._get_validation_dataset(split, dspec, dataset_cls)
                trainer.train(trainset, valset)
                cache = {**self.args, **trainer.cache, **dspec, **trainer.nn, **trainer.optimizer}
                _logutils.save_cache(cache, experiment_id=trainer.cache['experiment_id'])
            """#########################################################"""

            if self.args['phase'] == 'train' or self.args['pretrained_path'] is None:
                """
                Best model will be split_name.pt in training phase, and if no pretrained path is supplied.
                """
                trainer.load_best_model()

            """########## Run test phase. ##############################"""
            testset = self._get_test_dataset(split, dspec, dataset_cls)
            test_averages, test_score = trainer.evaluation(split_key='test', save_pred=True,
                                                           dataset_list=testset)

            """
            Accumulate global scores-scores of each fold to report single global score for each datasets.
            """
            global_averages.accumulate(test_averages)
            global_score.accumulate(test_score)

            """
            Save the calculated scores in list so that later we can do extra things(Like save to a file.)
            """
            trainer.cache['test_score'].append([*test_averages.get(), *test_score.get()])
            trainer.cache['global_test_score'].append([split_file, *test_averages.get(), *test_score.get()])
            _logutils.save_scores(trainer.cache, experiment_id=trainer.cache['experiment_id'],
                                  file_keys=['test_score'])

        """
        Finally, save the global score to a file
        """
        trainer.cache['global_test_score'].append(['Global', *global_averages.get(), *global_score.get()])
        _logutils.save_scores(trainer.cache, file_keys=['global_test_score'])

    def run(self, dataset_cls, trainer_cls,
            data_splitter: _Callable = _du.init_kfolds_):
        r"""
        Run for individual datasets
        """
        for dspec in self.dataspecs:
            if self.args['use_ddp']:
                _mp.spawn(test, nprocs=self.args['num_gpus'],
                          args=('HI There', ))
            else:
                self._run(dspec, dataset_cls, trainer_cls, data_splitter)

    def _run_pooled(self, dataset_cls, trainer_cls,
                    data_splitter: _Callable = _du.init_kfolds_):
        trainer = trainer_cls(self.args)

        """
        Check if the splits are given. If not, create new.
        """
        for dspec in self.dataspecs:
            trainer.cache['log_dir'] = self.args['log_dir'] + _sep + dspec['name']
            if _du.create_splits_(trainer.cache['log_dir'], dspec):
                data_splitter(dspec=dspec, args=self.args)

        """
        Create log-dir by concatenating all the involved dataset names.
        """
        trainer.cache['log_dir'] = self.args['log_dir'] + _sep + 'pooled_' + '_'.join(
            [d['name'] for d in self.dataspecs])

        """
        Default global score holder for each datasets.
        Save the latest time(maximize current time.). One can also maximize/minimize any other score from
        easytorch.metrics.ETMetrics() class by overriding _reset_dataset_cache.
        """
        trainer.cache['global_test_score'] = []
        global_score = trainer.new_metrics()
        global_averages = trainer.new_averages()

        """
        The easytorch.metrics.Prf1a() has Precision,Recall,F1,Accuracy,and Overlap implemented.
         We use F1 as default score to monitor while doing validation and save best model.
         And we will have loss returned by easytorch.metrics.Averages() while training.
        """
        trainer.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
        trainer.cache.update(monitor_metric='f1', metric_direction='maximize')

        """
        reset_dataset_cache() is an intervention to set any specific needs for each dataset. For example:
            - custom log_dir
            - Monitor some other metrics
            - Set metrics direction differently.
        """
        trainer.reset_dataset_cache()
        _os.makedirs(trainer.cache['log_dir'], exist_ok=True)

        trainer.cache['experiment_id'] = 'pooled'
        trainer.cache['checkpoint'] = trainer.cache['experiment_id'] + '.pt'

        trainer.cache.update(best_epoch=0, best_score=0.0)
        if trainer.cache['metric_direction'] == 'minimize':
            trainer.cache['best_score'] = 1e11

        trainer.check_previous_logs()
        trainer.init_nn()

        """
        Clear cache to save scores for each fold
        """
        trainer.cache.update(training_log=[], validation_log=[], test_score=[])

        """
        An intervention point if anyone wants to change things for each fold.
        """
        trainer.reset_fold_cache()

        if self.args['phase'] == 'train':
            train_dataset = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='train',
                                             load_sparse=False)[0]
            val_dataset = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='validation',
                                           load_sparse=False)[0]
            trainer.train(train_dataset, val_dataset)
            cache = {**self.args, **trainer.cache, 'dataspecs': self.dataspecs}
            _logutils.save_cache(cache, experiment_id=cache['experiment_id'])

        if self.args['phase'] == 'train' or self.args['pretrained_path'] is None:
            """
            Best model will be split_name.pt in training phase, and if no pretrained path is supplied.
            """
            trainer.load_best_model()

        test_dataset_list = dataset_cls.pool(self.args, dataspecs=self.dataspecs, split_key='test',
                                             load_sparse=self.args['load_sparse'])
        test_averages, test_score = trainer.evaluation(split_key='test', save_pred=True, dataset_list=test_dataset_list)

        global_averages.accumulate(test_averages)
        global_score.accumulate(test_score)
        trainer.cache['test_score'].append(['Global', *global_averages.get(), *global_score.get()])
        _logutils.save_scores(trainer.cache, experiment_id=trainer.cache['experiment_id'], file_keys=['test_score'])

    def run_pooled(self, dataset_cls, trainer_cls,
                   data_splitter: _Callable = _du.init_kfolds_):
        r"""
        Run in pooled fashion.
        """
        self._run_pooled(dataset_cls, trainer_cls, data_splitter)
