import os
import pandas as pd
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from srcs.utils import get_logger


class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.logger = get_logger('tensorboard-writer')
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.timer = datetime.now()

    def set_step(self, step):
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            if self.writer:
                attr = getattr(self.writer, name)
                return attr


class BatchMetrics:
    def __init__(self, *keys, postfix='', writer=None):
        self.writer = writer
        self.postfix = postfix
        if postfix:
            keys = [k + postfix for k in keys]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data.loc[:, col] = 0  # Исправлено

    def update(self, key, value, n=1):
        if self.postfix:
            key = key + self.postfix
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.loc[key, "total"] += value * n  # Исправлено
        self._data.loc[key, "counts"] += n  # Исправлено
        self._data.loc[key, "average"] = self._data.loc[key, "total"] / self._data.loc[key, "counts"]  # Исправлено

    def avg(self, key):
        if self.postfix:
            key = key + self.postfix
        return self._data.loc[key, "average"]

    def result(self):
        return dict(self._data["average"])


class EpochMetrics:
    def __init__(self, metric_names, phases=('train', 'valid'), monitoring='off'):
        self.logger = get_logger('epoch-metrics')
        columns = tuple(product(metric_names, phases))
        self._data = pd.DataFrame(columns=columns)
        self.monitor_mode, self.monitor_metric = self._parse_monitoring_mode(monitoring)
        self.topk_idx = []

    def minimizing_metric(self, idx):
        if self.monitor_mode == 'off':
            return 0
        try:
            metric = self._data[self.monitor_metric].loc[idx]
        except KeyError:
            self.logger.warning(f"Warning: Metric '{self.monitor_metric}' is not found. Model performance monitoring is disabled.")
            self.monitor_mode = 'off'
            return 0
        return metric if self.monitor_mode == 'min' else -metric

    def _parse_monitoring_mode(self, monitor_mode):
        if monitor_mode == 'off':
            return 'off', None
        else:
            monitor_mode, monitor_metric = monitor_mode.split()
            monitor_metric = tuple(monitor_metric.split('/'))
            assert monitor_mode in ['min', 'max']
        return monitor_mode, monitor_metric

    def is_improved(self):
        if self.monitor_mode == 'off':
            return True

        last_epoch = self._data.index[-1]
        best_epoch = self.topk_idx[0]
        return last_epoch == best_epoch

    def keep_topk_checkpt(self, checkpt_dir, k=3):
        if len(self.topk_idx) > k and self.monitor_mode != 'off':
            last_epoch = self._data.index[-1]
            self.topk_idx = self.topk_idx[:(k+1)]
            to_delete = last_epoch if last_epoch not in self.topk_idx else self.topk_idx[-1]

            filename = str(checkpt_dir / f'checkpoint-epoch{to_delete.split("-")[1]}.pth')
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

    def update(self, epoch, result):
        epoch_idx = f'epoch-{epoch}'
        for k, v in result.items():  # Исправлено
            self._data.loc[epoch_idx, tuple(k.split('/'))] = v  # Исправлено

        self.topk_idx.append(epoch_idx)
        self.topk_idx = sorted(self.topk_idx, key=self.minimizing_metric)

    def latest(self):
        return self._data[-1:]

    def to_csv(self, save_path=None):
        self._data.to_csv(save_path)

    def __str__(self):
        return str(self._data)
