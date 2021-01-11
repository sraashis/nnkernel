import os as _os
import copy as _copy
import json as _json


class FrozenDict(dict):
    def __init__(self, _dict):
        super().__init__(_dict)

    def prompt(self, key, value):
        raise ValueError(f'*** '
                         f'Attempt to modify frozen dict '
                         f'[{key} : {self[key]}] with [{key} : {value}]'
                         f' ***')

    def __setitem__(self, key, value):
        if key not in self:
            super(FrozenDict, self).__setitem__(key, value)
        else:
            self.prompt(key, value)

    def update(self, **kw):
        for k, v in kw.items():
            self[k] = v


def save_scores(cache, experiment_id='', file_keys=[]):
    for fk in file_keys:
        with open(cache['log_dir'] + _os.sep + f'{experiment_id}_{fk}.csv', 'w') as file:
            header = 'Scores,' + cache.get('log_header', '')
            for line in [header] + cache[fk] if any(isinstance(ln, list) for ln in cache[fk]) else [cache[fk]]:
                if isinstance(line, list):
                    file.write(','.join([str(s) for s in line]) + '\n')
                else:
                    file.write(f'{line}\n')


def jsonable(obj):
    try:
        _json.dumps(obj)
        return True
    except:
        return False


def clean_recursive(obj):
    r"""
    Make everything in cache safe to save in json files.
    """
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            clean_recursive(v)
        elif isinstance(v, list):
            for i in v:
                clean_recursive(i)
        elif not jsonable(v):
            obj[k] = f'{v}'


def save_cache(cache, experiment_id=''):
    with open(cache['log_dir'] + _os.sep + f"{experiment_id}_log.json", 'w') as fp:
        log = _copy.deepcopy(cache)
        clean_recursive(log)
        _json.dump(log, fp)
