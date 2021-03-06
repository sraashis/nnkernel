r"""
The main core of EasyTorch
"""

import math as _math
import os as _os
from collections import OrderedDict as _ODict

import torch as _torch

import easytorch.config as _config
import easytorch.data as _etdata
import easytorch.utils as _etutils
from easytorch.metrics import metrics as _base_metrics
from easytorch.utils.tensorutils import initialize_weights as _init_weights
from .vision import plotter as _log_utils
from easytorch.utils.logger import *

_sep = _os.sep


class ETTrainer:
    def __init__(self, args: dict):
        r"""
        args: receives the arguments passed by the ArgsParser.
        cache: Initialize all immediate things here. Like scores, loss, accuracies...
        nn:  Initialize our models here.
        optimizer: Initialize our optimizers.
        """
        self.args = _etutils.FrozenDict(args)
        self.cache = _ODict()
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()

    def init_nn(self, **kw):
        r"""
        Call to user implementation of:
            Initialize models.
            Initialize random/pre-trained weights.
            Initialize/Detect GPUS.
            Initialize optimizer.
        """

        self._init_nn_model()
        # Print number of parameters in all models.
        if self.args['verbose']:
            for k, m in self.nn.items():
                if isinstance(m, _torch.nn.Module):
                    success(f'Total params in {k}:'
                            f' {sum(p.numel() for p in m.parameters() if p.requires_grad)}')

        self._init_nn_weights(**kw)
        self._init_optimizer()
        self._set_device()

    def _init_nn_weights(self, **kw):
        r"""
        By default, will initialize network with Kaimming initialization.
        If path to pretrained weights are given, it will be used instead.
        """
        if self.args['pretrained_path'] is not None:
            self.load_checkpoint(self.args['pretrained_path'])
        elif self.args['phase'] == 'train':
            _torch.manual_seed(self.args['seed'])
            for mk in self.nn:
                _init_weights(self.nn[mk])

    def load_checkpoint_from_key(self, key='checkpoint'):
        self.load_checkpoint(self.cache['log_dir'] + _sep + self.cache[key])

    def load_checkpoint(self, full_path):
        r"""
        Load checkpoint from the given path:
            If it is an easytorch checkpoint, try loading all the models.
            If it is not, assume it's weights to a single model and laod to first model.
        """
        try:
            chk = _torch.load(full_path)
        except:
            chk = _torch.load(full_path, map_location='cpu')

        if chk.get('source', 'Unknown').lower() == 'easytorch':
            for m in chk['models']:
                try:
                    self.nn[m].module.load_state_dict(chk['models'][m])
                except:
                    self.nn[m].load_state_dict(chk['models'][m])

            for m in chk['optimizers']:
                try:
                    self.optimizer[m].module.load_state_dict(chk['optimizers'][m])
                except:
                    self.optimizer[m].load_state_dict(chk['optimizers'][m])
        else:
            mkey = list(self.nn.keys())[0]
            try:
                self.nn[mkey].module.load_state_dict(chk)
            except:
                self.nn[mkey].load_state_dict(chk)

    def _init_nn_model(self):
        r"""
        User cam override and initialize required models in self.nn dict.
        """
        raise NotImplementedError('Must be implemented in child class.')

    def _set_device(self):
        r"""
        Initialize GPUs based on whats provided in args(Default [0])
        Expects list of GPUS as [0, 1, 2, 3]., list of GPUS will make it use DataParallel.
        If no GPU is present, CPU is used.
        """
        self.device['gpu'] = _torch.device("cpu")
        if _config.cuda_available and len(self.args['gpus']) >= 1:
            self.device['gpu'] = _torch.device(f"cuda:{self.args['gpus'][0]}")
            if len(self.args['gpus']) >= 2:
                for model_key in self.nn:
                    self.nn[model_key] = _torch.nn.DataParallel(self.nn[model_key], self.args['gpus'])
        for model_key in self.nn:
            self.nn[model_key] = self.nn[model_key].to(self.device['gpu'])

    def _init_optimizer(self):
        r"""
        Initialize required optimizers here. Default is Adam,
        """
        first_model = list(self.nn.keys())[0]
        self.optimizer['adam'] = _torch.optim.Adam(self.nn[first_model].parameters(),
                                                   lr=self.args['learning_rate'])

    def new_metrics(self):
        r"""
        User can override to supply desired implementation of easytorch.metrics.ETMetrics().
            Example: easytorch.metrics.Pr11a() will work with precision, recall, F1, Accuracy, IOU scores.
        """
        return _base_metrics.Prf1a()

    def new_averages(self):
        r""""
        Should supply an implementation of easytorch.metrics.ETAverages() that can keep track of multiple averages.
            Example: multiple loss, or any other values.
        """
        return _base_metrics.ETAverages(num_averages=1)

    def check_previous_logs(self):
        r"""
        Checks if there already is a previous run and prompt[Y/N] so that
        we avoid accidentally overriding previous runs and lose temper.
        User can supply -f True flag to override by force.
        """
        if self.args['force']:
            warn('Forced overriding previous logs.')
            return
        i = 'y'
        if self.args['phase'] == 'train':
            train_log = f"{self.cache['log_dir']}{_sep}{self.cache['experiment_id']}_log.json"
            if _os.path.exists(train_log):
                i = input(f"*** {train_log} *** \n Exists. OVERRIDE [y/n]:")

        if self.args['phase'] == 'test':
            test_log = f"{self.cache['log_dir']}{_sep}{self.cache['experiment_id']}_test_scores.csv"
            if _os.path.exists(test_log):
                if _os.path.exists(test_log):
                    i = input(f"*** {test_log} *** \n Exists. OVERRIDE [y/n]:")

        if i.lower() == 'n':
            raise FileExistsError(f' ##### {self.args["log_dir"]} directory is not empty. #####')

    def save_checkpoint(self, file_name, src='easytorch'):
        checkpoint = {'source': src}
        for k in self.nn:
            checkpoint['models'] = {}
            try:
                checkpoint['models'][k] = self.nn[k].module.state_dict()
            except:
                checkpoint['models'][k] = self.nn[k].state_dict()
        for k in self.optimizer:
            checkpoint['optimizers'] = {}
            try:
                checkpoint['optimizers'][k] = self.optimizer[k].module.state_dict()
            except:
                checkpoint['optimizers'][k] = self.optimizer[k].state_dict()
        _torch.save(checkpoint, self.cache['log_dir'] + _sep + file_name)

    def reset_dataset_cache(self):
        r"""
        An extra layer to reset cache for each dataspec. For example:
        1. Set a new score to monitor:
            self.cache['monitor_metric'] = 'Precision'
            self.cache['metric_direction'] = 'maximize'
                            OR
            self.cache['monitor_metric'] = 'MSE'
            self.cache['metric_direction'] = 'minimize'
                            OR
                    to save latest model
            self.cache['monitor_metric'] = 'time'
            self.cache['metric_direction'] = 'maximize'
        2. Set new log_headers based on what is returned by get() method
            of your implementation of easytorch.metrics.ETMetrics and easytorch.metrics.ETAverages class:
            For example, the default implementation is:
            - The get method of easytorch.metrics.ETAverages class returns the average loss value.
            - The get method of easytorch.metrics.Prf1a returns Precision,Recall,F1,Accuracy
            - so Default heade is [Loss,Precision,Recall,F1,Accuracy]
        3. Set new log_dir based on different experiment versions on each datasets as per info. received from arguments.
        """
        pass

    def reset_fold_cache(self):
        """Nothing specific to do here.
        Just keeping in case we need to intervene with each of the k-folds just like each datasets above.
        """
        pass

    def save_if_better(self, epoch, metrics):
        r"""
        Save the current model as best if it has better validation scores.
        """
        sc = getattr(metrics, self.cache['monitor_metric'])
        if callable(sc):
            sc = sc()

        if (self.cache['metric_direction'] == 'maximize' and sc >= self.cache['best_score']) or (
                self.cache['metric_direction'] == 'minimize' and sc <= self.cache['best_score']):
            self.save_checkpoint(self.cache['checkpoint'])
            self.cache['best_score'] = sc
            self.cache['best_epoch'] = epoch
            if self.args['verbose']:
                success(f"BEST! Model *** Saved *** : {self.cache['best_score']}")
        else:
            if self.args['verbose']:
                warn(f"Not best: {sc}, {self.cache['best_score']} in ep: {self.cache['best_epoch']}")

    def iteration(self, batch):
        r"""
        Left for user to implement one mini-bath iteration:
        Example:{
                    inputs = batch['input'].to(self.device['gpu']).float()
                    labels = batch['label'].to(self.device['gpu']).long()
                    out = self.nn['model'](inputs)
                    loss = F.cross_entropy(out, labels)
                    out = F.softmax(out, 1)
                    _, pred = torch.max(out, 1)
                    sc = self.new_metrics()
                    sc.add(pred, labels)
                    avg = self.new_averages()
                    avg.add(loss.item(), len(inputs))
                    return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}
                }
        Note: loss, averages, and metrics are required, whereas others are optional
            -we will have to do backward on loss
            -we need to keep track of loss
            -we need to keep track of metrics
        """
        return {'metrics': _base_metrics.ETMetrics(), 'averages': _base_metrics.ETAverages(num_averages=1)}

    def save_predictions(self, dataset, its):
        r"""
        If one needs to save complex predictions result like predicted segmentations.
         -Especially with U-Net architectures, we split images and train.
        Once the argument --sp/-sparse-load is set to True,
        the argument 'its' will receive all the patches of single image at a time.
        From there, we can recreate the whole image.
        """
        pass

    def evaluation(self, split_key=None, save_pred=False, dataset_list=None):
        r"""
        Evaluation phase that handles validation/test phase
        split-key: the key to list of files used in this particular evaluation.
        The program will create k-splits(json files) as per specified in --nf -num_of_folds
         argument with keys 'train', ''validation', and 'test'.
        """
        for k in self.nn:
            self.nn[k].eval()

        if self.args['verbose']:
            info('')
            info(f'Running {split_key}')

        eval_avg = self.new_averages()
        eval_metrics = self.new_metrics()
        val_loaders = [_etdata.ETDataLoader.new(mode='eval', shuffle=False, dataset=d, **self.args) for d in
                       dataset_list]
        with _torch.no_grad():
            for loader in val_loaders:
                its = []
                metrics = self.new_metrics()
                avg = self.new_averages()
                for i, batch in enumerate(loader):

                    it = self.iteration(batch)
                    if not it.get('metrics'):
                        it['metrics'] = _base_metrics.ETMetrics()

                    metrics.accumulate(it['metrics'])
                    avg.accumulate(it['averages'])
                    if save_pred:
                        its.append(it)
                    if self.args['verbose'] and len(dataset_list) <= 1 and i % int(_math.log(i + 1) + 1) == 0:
                        info(f"Itr:{i}/{len(loader)}, {it['averages'].get()}, {it['metrics'].get()}")

                eval_metrics.accumulate(metrics)
                eval_avg.accumulate(avg)
                if self.args['verbose'] and len(dataset_list) > 1:
                    info(f"{split_key}, {avg.get()}, {metrics.get()}")
                if save_pred:
                    self.save_predictions(loader.dataset, its)

        if self.args['verbose']:
            info(f"{self.cache['experiment_id']} {split_key} metrics: {eval_metrics.get()}")
        return eval_avg, eval_metrics

    def training_iteration(self, batch):
        r"""
        Learning step for one batch.
        We decoupled it so that user could implement any complex/multi/alternate training strategies.
        """
        first_optim = list(self.optimizer.keys())[0]
        self.optimizer[first_optim].zero_grad()
        its = []
        for i in range(self.cache.get('num_iteration', 1)):
            """Accumulate gradients"""
            it = self.iteration(batch)
            it['loss'].backward()
            its.append(it)
        self.optimizer[first_optim].step()
        return self._reduce_iteration(its)

    def _reduce_iteration(self, its):
        if len(its) == 1:
            return its[0]
        reduced = {}.fromkeys(its[0].keys(), None)
        for k in reduced:
            if isinstance(its[0][k], _base_metrics.ETAverages):
                reduced[k] = self.new_averages()
                [reduced[k].accumulate(ik[k]) for ik in its]

            elif isinstance(its[0][k], _base_metrics.ETMetrics):
                reduced[k] = self.new_metrics()
                [reduced[k].accumulate(ik[k]) for ik in its]

            elif isinstance(its[0][k], _torch.Tensor) and not its[0][k].requires_grad and its[0][k].is_leaf:
                reduced[k] = _torch.cat([ik[k] for ik in its])

            else:
                reduced[k] = [ik[k] for ik in its]
        return reduced

    def _on_epoch_end(self, ep, ep_averages, ep_metrics, val_averages, val_metrics):
        r"""
        Any logic to run after an epoch ends.
        """
        _log_utils.plot_progress(self.cache, experiment_id=self.cache['experiment_id'],
                                 plot_keys=['training_log', 'validation_log'], epoch=ep)

    def _on_iteration_end(self, i, ep, it):
        r"""
        Any logic to run after an iteration ends.
        """
        pass

    def _early_stopping(self, ep, ep_averages, ep_metrics, val_averages, val_metrics):
        r"""
        Stop the training based on some criteria.
         For example: the implementation below will stop training if the validation
         scores does not improve within a 'patience' number of epochs.
        """
        if ep - self.cache['best_epoch'] >= self.args.get('patience', 'epochs'):
            return True
        return False

    def train(self, dataset, val_dataset):
        r"""
        Main training loop.
        """
        train_loader = _etdata.ETDataLoader.new(mode='train', shuffle=True, dataset=dataset, **self.args)
        for ep in range(1, self.args['epochs'] + 1):
            if self.args['verbose']: info('')

            for k in self.nn:
                self.nn[k].train()

            _metrics = self.new_metrics()
            _avg = self.new_averages()
            ep_avg = self.new_averages()
            ep_metrics = self.new_metrics()
            for i, batch in enumerate(train_loader, 1):

                it = self.training_iteration(batch)
                if not it.get('metrics'):
                    it['metrics'] = _base_metrics.ETMetrics()

                ep_avg.accumulate(it['averages'])
                ep_metrics.accumulate(it['metrics'])

                """
                Running loss/metrics
                """
                _avg.accumulate(it['averages'])
                _metrics.accumulate(it['metrics'])
                if self.args['verbose'] and i % int(_math.log(i + 1) + 1) == 0:
                    info(f"Ep:{ep}/{self.args['epochs']},Itr:{i}/{len(train_loader)},"
                         f"{_avg.get()},{_metrics.get()}")

                    self.cache['training_log'].append([*_avg.get(), *_metrics.get()])
                    _metrics.reset()
                    _avg.reset()

                self._on_iteration_end(i, ep, it)

            self.cache['training_log'].append([*ep_avg.get(), *ep_metrics.get()])
            val_loss, val_metric = self.evaluation(split_key='validation', dataset_list=[val_dataset])
            self.save_if_better(ep, val_metric)
            self.cache['validation_log'].append([*val_loss.get(), *val_metric.get()])

            self._on_epoch_end(ep, ep_avg, ep_metrics, val_loss, val_metric)
            if self._early_stopping(ep, ep_avg, ep_metrics, val_loss, val_metric):
                break
