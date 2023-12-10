import numpy as np
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torch import nn
from torch import optim
import torchvision.transforms.functional as F
from IPython import embed
import matplotlib.pyplot as plt
from helper.augment_data import color_image


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        print(self.optimizer.param_groups[0]['lr'])
        last_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            last_loss = loss.item()
            
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(log["car_part_accuracy"])
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class GanTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer: tuple, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.L1_criterion = nn.CrossEntropyLoss()
        self.L1_lambda = 100

        self.G_optimizer = self.optimizer[0]
        #optim.Adam(self.model.G.parameters(), lr=self.lr_scheduler.get_last_lr()[0], betas=(0.5, 0.999))
        self.D_optimizer = self.optimizer[1]
        #optim.Adam(self.model.D.parameters(), lr=self.lr_scheduler.get_last_lr()[0], betas=(0.5, 0.999))

        self.train_metrics = MetricTracker('G_loss', 'D_loss','L1_loss', *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)
        self.valid_metrics = MetricTracker('G_loss', 'D_loss','L1_loss', *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            b_size = data.shape[0]

            real_class = torch.ones(b_size, 1, 7, 7).to(self.device)
            fake_class = torch.zeros(b_size, 1, 7, 7).to(self.device)

            # Train D
            self.model.D.zero_grad()

            real_patch = self.model.D(target, data)

            real_gan_loss = self.criterion(real_patch, real_class)

            fake = self.model.G(data)

            if self.config["train_pictures"]:
                tmp = fake.cpu().detach().numpy()
                for i in range(tmp.shape[0]):
                    plt.imshow(color_image(tmp[i]))
                    plt.savefig(f"./images/train/{str(i)}")
                # [self.writer.add_image('fake image number ' + str(i), np.transpose(color_image(tmp[i]), (2, 0, 1))) for i in
                # range(tmp.shape[0])]

            fake_patch = self.model.D(fake.detach(), data)
            fake_gan_loss = self.criterion(fake_patch, fake_class)

            D_loss = real_gan_loss + fake_gan_loss
            D_loss.backward()
            self.D_optimizer.step()

            # Train G
            self.model.G.zero_grad()
            fake_patch = self.model.D(fake, data)
            fake_gan_loss = self.criterion(fake_patch, real_class)

            L1_loss = self.L1_criterion(fake, target)
            G_loss = fake_gan_loss + self.L1_lambda * L1_loss
            G_loss.backward()

            self.G_optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('G_loss', G_loss.item())
            self.train_metrics.update('D_loss', D_loss.item())
            self.train_metrics.update('L1_loss', L1_loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(fake, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Real Gan Loss: {real_gan_loss.item()} Fake Gan Loss: {fake_gan_loss.item()} Overall Loss: {L1_loss.item()}')
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler[0].step(G_loss.item())
            self.lr_scheduler[1].step(D_loss.item())
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                b_size = data.shape[0]

                real_class = torch.ones(b_size, 1, 7, 7).to(self.device)
                fake_class = torch.zeros(b_size, 1, 7, 7).to(self.device)

                real_patch = self.model.D(target, data)

                real_gan_loss = self.criterion(real_patch, real_class)

                fake = self.model.G(data)

                fake_patch = self.model.D(fake.detach(), data)
                fake_gan_loss = self.criterion(fake_patch, fake_class)

                D_loss = real_gan_loss + fake_gan_loss

                L1_loss = self.L1_criterion(fake, target)
                G_loss = fake_gan_loss + self.L1_lambda * L1_loss

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('G_loss', G_loss.item())
                self.valid_metrics.update('D_loss', D_loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(fake, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'G_optimizer': self.optimizer[0].state_dict(),
            'D_optimizer': self.optimizer[1].state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        if (torch.cuda.is_available()):
            checkpoint = torch.load(resume_path)
        else:
            checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        elif self.config['optimizer']['resume'] == False:
            self.logger.warning("Info: Optimizer parameters not being resumed.")
        else: 
            self.optimizer[0].load_state_dict(checkpoint['G_optimizer'])
            self.optimizer[1].load_state_dict(checkpoint['D_optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    
    