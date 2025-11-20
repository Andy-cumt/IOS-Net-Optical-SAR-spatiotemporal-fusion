import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from model import *
from LMSfusion import *
from torchRoberts import *
from data import PatchSet, SCALE_FACTOR, Mode
from datafortest import PatchSetfortest, get_pair_pathfortest
import utils

from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import sys
import os


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')

        self.resolution_scale = SCALE_FACTOR
        self.image_size = option.image_size
        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)

        self.last_S = self.train_dir / 'last_S.pth'
        self.best_S = self.train_dir / 'best_S.pth'
        self.best_S2 = self.train_dir / 'best_S2.pth'
        self.best_C = self.train_dir / 'best_C.pth'

        self.logger = utils.get_logger()
        self.logger.info('Model initialization')


        self.Downsampling = Downsampling(SCALE = 8).to(self.device)
        self.robertsmodel = roberts().to(self.device)
        self.LMSfusion = LMSfusion().to(self.device)


        if option.cuda and option.ngpu > 1:
            device_ids = [i for i in range(option.ngpu)]
            self.LMSfusion = nn.DataParallel(self.LMSfusion, device_ids=device_ids)


        self.criterionP2 = CompoundLoss()
        self.robertsLoss = robertsLoss(self.robertsmodel)

        self.optimizer2 = optim.Adam(self.LMSfusion.parameters(), lr=option.lr2, weight_decay=1e-6)

        self.logger.info(str(self.LMSfusion))

        n_params2 = sum(p.numel() for p in self.LMSfusion.parameters() if p.requires_grad)

        self.logger.info(f'There are {n_params2} trainable parameters in SuperrNet.')


    def train_on_epoch(self, n_epoch, data_loader,deploy):
        self.LMSfusion.train()
        epoch_loss2 = utils.AverageMeter()
        epoch_loss21 = utils.AverageMeter()  ####
        epoch_loss22 = utils.AverageMeter()  ####
        epoch_loss23 = utils.AverageMeter()  ####
        epoch_error2 = utils.AverageMeter()
        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]#加速
            inputs, target = data[:-1], data[-1] #[:-1]除最后一个取全部；[-1]取最后一个

            data2 = data.copy()

            self.optimizer2.zero_grad()
            data2[1] = data2[1][:, 0:6, :, :]

            predictions2 = self.LMSfusion(data2)

            mask = target[:, -1, :, :] * 10000.0
            target = target[:, 0:6, :, :]
            for j in range(0, predictions2.shape[1]):
                predictions2[:, j, :, :] = predictions2[:, j, :, :] * mask
                target[:, j, :, :] = target[:, j, :, :] * mask

            loss2, LP21, LP22 = self.criterionP2(predictions2, target, 30)
            LP23 = self.robertsLoss(predictions2, target, 0.5)
            loss2 = loss2 + LP23
            loss2.backward()
            print('loss2:',loss2)

            self.optimizer2.step()

            epoch_loss2.update(loss2.item())
            epoch_loss21.update(LP21.item())
            epoch_loss22.update(LP22.item())
            epoch_loss23.update(LP23.item())

            with torch.no_grad():
                score2 = F.mse_loss(predictions2, target)

            epoch_error2.update(score2.item())

            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        return epoch_loss2.avg, epoch_error2.avg,\
            epoch_loss21.avg, epoch_loss22.avg, epoch_loss23.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader,deviceLabel,deploy):
        self.LMSfusion.eval()

        epoch_loss2 = utils.AverageMeter()
        epoch_loss21 = utils.AverageMeter()  ####
        epoch_loss22 = utils.AverageMeter()  ####
        epoch_loss23 = utils.AverageMeter()  ####
        epoch_error2 = utils.AverageMeter()

        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]

            data2 = data.copy()


            data2[1] = data2[1][:, 0:6, :, :]

            predictions2 = self.LMSfusion(data2)

            mask = target[:, -1, :, :] * 10000.0
            target = target[:, 0:6, :, :]
            for j in range(0, predictions2.shape[1]):
                predictions2[:, j, :, :] = predictions2[:, j, :, :] * mask
                target[:, j, :, :] = target[:, j, :, :] * mask

            loss2, LP21, LP22 = self.criterionP2(predictions2, target,  30)
            LP23 = self.robertsLoss(predictions2, target, 0.5)
            loss2 = loss2 + LP23

            epoch_loss2.update(loss2.item())

            epoch_loss21.update(LP21.item())
            epoch_loss22.update(LP22.item())
            epoch_loss23.update(LP23.item())

            score2 = F.mse_loss(predictions2, target)
            epoch_error2.update(score2.item())

        utils.save_checkpoint(self.LMSfusion, self.optimizer2, self.last_S)

        return epoch_loss2.avg, epoch_error2.avg,\
                epoch_loss21.avg, epoch_loss22.avg, epoch_loss23.avg

    def train(self, train_dir1, val_dir, patch_size, patch_stride, batch_size,
            num_workers=1, epochs=30, resume=True):

        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir1, self.image_size, patch_size, patch_stride,
                            mode=Mode.TRAINING) #所有影像被切成小块
        val_set = PatchSet(val_dir, self.image_size, patch_size, mode=Mode.VALIDATION)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
        least_error2 = sys.maxsize
        least_loss2 = sys.maxsize
        start_epoch = 0

        if resume and self.last_S.exists():
            utils.load_checkpoint(self.last_S, self.LMSfusion, self.optimizer2)

        if self.history.exists():
            df = pd.read_csv(self.history)
            least_error2 = df['verror2'].min()
            least_loss2 = df['vloss2'].min()
            start_epoch = int(df.iloc[-1]['epoch']) + 1

        self.logger.info('Training...')
        scheduler2 = ReduceLROnPlateau(self.optimizer2, mode='min', factor=0.1, patience=5)
        for epoch in range(start_epoch, epochs + start_epoch):
            for param_group in self.optimizer2.param_groups:
                self.logger.info(f"Current learning rate for SuperrNet: {param_group['lr']}")

            tloss2, terror2,tloss21, tloss22,tloss23= self.train_on_epoch(epoch, train_loader,deploy=False)
            vloss2, verror2,vloss21, vloss22,vloss23= self.test_on_epoch(val_loader,epoch,deploy=False)
            csv_header = ['epoch','tloss2', 'terror2',
                          'tloss21','tloss22','tloss23',
                          'vloss2', 'verror2',
                          'vloss21','vloss22','vloss23']
            csv_values = [epoch,  tloss2,  terror2,
                          tloss21, tloss22,tloss23,
                          vloss2, verror2,
                          vloss21, vloss22, vloss23]


            utils.log_csv(self.history, csv_values, header=csv_header)

            scheduler2.step(vloss2)

            if verror2 < least_error2:
                shutil.copy(str(self.last_S), str(self.best_S))
                least_error2 = verror2
            if vloss2 < least_loss2:
                shutil.copy(str(self.last_S), str(self.best_S2))
                least_loss2 = vloss2

    @torch.no_grad()
    def test(self, test_dir, patch_size, test_refs, num_workers=0):
        self.LMSfusion.eval()

        patch_size = utils.make_tuple(patch_size)

        utils.load_checkpoint(self.best_S, model=self.LMSfusion)


        self.logger.info('Testing...')

        image_dirs = [p for p in test_dir.glob('*') if p.is_dir()]
        image_paths = [get_pair_pathfortest(d) for d in image_dirs]

        # 在预测阶段，对图像进行切块的时候必须刚好裁切完全，这样才能在预测结束后进行完整的拼接
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols
        test_set = PatchSetfortest(test_dir, self.image_size, patch_size)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        scaled_patch_size = tuple(i * self.resolution_scale for i in patch_size) #测试数据的分块大小
        scaled_image_size = tuple(i * self.resolution_scale for i in self.image_size) #测试数据的总大小
        pixel_value_scale = 10000
        im_count = 0
        patches = []
        t_start = datetime.now()
        for inputs in test_loader:
            name = image_paths[im_count][-1].name  #-1为最后一个，也就是待预测的影像

            if len(patches) == 0:
                t_start = timer()
                self.logger.info(f'Predict on image {name}')

            # 分块进行预测（每次进入深度网络的都是影像中的一块）
            inputs = [im.to(self.device) for im in inputs]
            inputs2 = inputs.copy()


            inputs2[1] = inputs2[1][:,0:6,:,:]
            inputs2[1] = fillmask(inputs2[1])
            predictions2 = self.LMSfusion(inputs2)

            prediction = predictions2.cpu().numpy() #把cuda格式的结果转换为numpy
            patches.append(prediction * pixel_value_scale)  #im2tensor，先自动除以了10000

            # 完成一张影像以后进行拼接
            if len(patches) == n_blocks:
                result = np.empty((NUM_BANDS, *scaled_image_size), dtype=np.float32)
                block_count = 0
                for i in range(rows):
                    row_start = i * scaled_patch_size[1]
                    for j in range(cols):
                        col_start = j * scaled_patch_size[0]
                        result[:,
                        col_start: col_start + scaled_patch_size[0],
                        row_start: row_start + scaled_patch_size[1]
                        ] = patches[block_count]
                        block_count += 1
                patches.clear()
                # 存储预测影像结果

                metadata = {
                    'driver': 'GTiff',
                    'width': scaled_image_size[1],
                    'height': scaled_image_size[0],
                    'count': NUM_BANDS,
                    'dtype': np.int16
                }

                result = result.astype(np.int16)
                # prototype = str(image_paths[im_count][1])
                utils.save_array_as_tif(result, self.test_dir / name, metadata)
                im_count += 1
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s')
