import copy as _copy
import json as _json
import os as _os

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

_plt.switch_backend('agg')
_plt.rcParams["figure.figsize"] = [16, 9]


def plot_progress(cache, experiment_id='', plot_keys=[], num_points=11, epoch=None):
    r"""
    Custom plot to plot data from the cache by keys.
    """
    scaler = _MinMaxScaler()
    for k in plot_keys:
        _plt.clf()

        data = cache.get(k, [])

        if len(data) == 0:
            continue

        header = cache['log_header'].split(',')
        data = _np.array(data)

        n_cols = len(header)
        data = data[:, :n_cols]
        if _np.sum(data) <= 0:
            continue

        df = _pd.DataFrame(data, columns=header)

        if len(df) == 0:
            continue

        for col in df.columns:
            if max(df[col]) > 1:
                df[col] = scaler.fit_transform(df[[col]])

        rollin_window = max(df.shape[0] // num_points + 1, 3)
        rolling = df.rolling(rollin_window, min_periods=1).mean()
        ax = df.plot(x_compat=True, alpha=0.2, legend=0)
        rolling.plot(ax=ax, title=k.upper())

        if epoch and epoch != df.shape[0]:
            """
            Set correct epoch as x-tick-labels.
            """
            xticks = list(range(0, df.shape[0], df.shape[0] // epoch)) + [df.shape[0] - 1]
            ax.set_xticks(xticks)
            ax.set_xticklabels(list(range(len(xticks))))

        _plt.xlabel('Epochs')
        _plt.savefig(cache['log_dir'] + _os.sep + f"{experiment_id}_{k}.png")
        _plt.close('all')
