from srcs.utils.util import is_master
import torch
import torch.distributed as dist
from srcs.trainer.base import BaseTrainer
from srcs.utils import inf_loop
from srcs.logger import BatchMetrics
import logging

class Trainer(BaseTrainer):
    def __init__(self, model, epochs, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, epochs, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.train_metrics = BatchMetrics('loss', *[m.__name__ for m in self.metric_ftns], postfix='/train',
                                          writer=self.writer)
        self.valid_metrics = BatchMetrics('loss', *[m.__name__ for m in self.metric_ftns], postfix='/valid',
                                          writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (batch_data, target) in enumerate(self.data_loader):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            target = target.to(self.device).squeeze(1).float() 

            self.optimizer.zero_grad()
            output_logits, tab_m_loss = self.model(**batch_data)

            lambda_sparse = 1e-3 
            loss = self.criterion(output_logits, target) + lambda_sparse * tab_m_loss.mean()

            loss.backward()
            self.optimizer.step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.6f}')
                
                with torch.no_grad():
                    metric_output = torch.round(torch.sigmoid(output_logits)) 
                    for met in self.metric_ftns:
                        # ИСПРАВЛЕНО: met.name -> met.__name__
                        self.train_metrics.update(met.__name__, met(metric_output.cpu(), target.cpu()))

            if batch_idx == self.len_epoch:
                break
        
        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)
        
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = val_log['loss/valid']
                self.lr_scheduler.step(val_loss)
            else:
                self.lr_scheduler.step()

        self.writer.set_step(epoch)
        # if epoch == 1 and is_master():
        #     keys_in_order = ['image', 'input_ids', 'attention_mask', 'tabular']
        #     input_to_graph = tuple(batch_data[k] for k in keys_in_order)
        #     self.writer.add_graph(self.model, input_to_graph)
        for k, v in log.items():
            self.writer.add_scalar(k + '/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        try:
            self.model.eval()
            self.valid_metrics.reset()
            with torch.no_grad():
                for batch_idx, (batch_data, target) in enumerate(self.valid_data_loader):
                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                    target = target.to(self.device).squeeze(1).float() 

                    # ИСПРАВЛЕНО: Распаковываем вывод модели, как в _train_epoch
                    output_logits, tab_m_loss = self.model(**batch_data)
                    
                    # ИСПРАВЛЕНО: Считаем основной лосс без m_loss от TabNet на валидации
                    loss = self.criterion(output_logits, target)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                    self.valid_metrics.update('loss', loss.item())
                    
                    metric_output = torch.round(torch.sigmoid(output_logits))
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(metric_output.cpu(), target.cpu()))

            return self.valid_metrics.result()
        except Exception as e:
            logging.error(msg=f"{e}")
    

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            # epoch-based training
            total = len(self.data_loader.dataset)
            current = batch_idx * self.data_loader.batch_size
            if dist.is_initialized():
                current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.len_epoch
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)