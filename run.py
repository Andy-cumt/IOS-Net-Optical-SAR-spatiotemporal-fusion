import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import data
from experiment import Experiment

import faulthandler
faulthandler.enable()
# Get cut-down GDAL that rasterio uses

"""
nohup python run.py --lr 1e-3 --num_workers 4 --batch_size 4 --epochs 60 --cuda --ngpu 1 --refs 2 --patch_size 35 --patch_stride 30 --test_patch 75 --pretrained encoder.pth --save_dir out --train_dir data/train --val_dir data/val --test_dir data/val &> out.log &
"""

# 获取模型运行时必须的一些参数
parser = argparse.ArgumentParser(description='Acquire some parameters for fusion restore')

parser.add_argument('--lr2', type=float, default=2e-4,
                    help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=2,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=0, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('F:\\SAROPT_fusion\\predictionPY'),
                    help='the output directory')


# 获取对输入数据进行预处理时的一些参数
parser.add_argument('--data_dir', type=Path, default=Path('F:\\SAROPT_fusion\\PYSTF'),
                    help='the training data directory')

parser.add_argument('--image_size', type=int, nargs='+', default=[400, 400],
                    help='the size of the coarse image (width, height)')
parser.add_argument('--patch_size', type=int, nargs='+', default=50,
                    help='the coarse image patch size for training restore')
parser.add_argument('--patch_stride', type=int, nargs='+', default=40,
                    help='the coarse patch stride for image division')
parser.add_argument('--test_patch', type=int, nargs='+', default=50,#可以为list
                    help='the coarse image patch size for fusion test')
opt = parser.parse_args()

torch.manual_seed(2023)
if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    torch.cuda.manual_seed_all(2023)
    cudnn.benchmark = True  #增加程序的运行效率
    cudnn.deterministic = True #固定随机数种子

if __name__ == '__main__':
    experiment = Experiment(opt)
    train_dir = opt.data_dir / 'train_cut'
    val_dir = opt.data_dir / 'val_cut'
    test_dir = opt.data_dir / 'test'

    if opt.epochs > 0:
        experiment.train(train_dir, val_dir,
                         opt.patch_size, opt.patch_stride, opt.batch_size,
                        num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(test_dir, opt.test_patch, opt.cuda,
                    num_workers=opt.num_workers)

