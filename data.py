from pathlib import Path
import numpy as np
import rasterio
import random
import math
from collections import OrderedDict
import glob
import torch
from torch.utils.data import Dataset
import os
from utils import make_tuple
from enum import Enum, auto, unique

@unique
class Mode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    PREDICTION = auto()


REF_PREFIX_1 = '0'
PRE_PREFIX = '1'
COARSE_PREFIX = 'M'
FINE_PREFIX = 'L'
SAR_PREFIX = 'S'
SCALE_FACTOR = 8


def get_pair_path(im_dir,id_n2, mode: Mode):
    # 将一组数据集按照规定的顺序组织好
    paths = []
    order = OrderedDict()  # 有序字典
    order1 = OrderedDict()  # 有序字典
    order2 = OrderedDict()
    folder_list = os.listdir(im_dir.parents[0])
    selected_folder = random.choice(folder_list)
    selected_folder_path = os.path.join(im_dir.parents[0], selected_folder)
    if mode is Mode.TRAINING:
        #order1[0] = COARSE_PREFIX
        order1[0] = id_n2 + '_' + FINE_PREFIX
        order2[0] = id_n2 + '_' + COARSE_PREFIX
        order2[1] = id_n2 + '_' + SAR_PREFIX
        order2[2] = id_n2 + '_' + FINE_PREFIX

        for prefix in order1.values():
            for path in Path(im_dir).glob('*.tif'):
                if path.name.startswith(prefix): #startswith检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
                    paths.append(path.expanduser().resolve()) #获得目录
                    break

        for prefix in order2.values():
            for path2 in Path(selected_folder_path).glob('*.tif'):
                if path2.name.startswith(prefix): #startswith检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
                    paths.append(path2.expanduser().resolve()) #获得目录
                    break

    else:
        #order[0] = REF_PREFIX_1 + '_' + COARSE_PREFIX
        order[0] = id_n2 + '_' + REF_PREFIX_1 + '_' + FINE_PREFIX
        order[1] = id_n2 + '_' + PRE_PREFIX + '_' + COARSE_PREFIX
        order[2] = id_n2 + '_' + SAR_PREFIX
        order[3] = id_n2 + '_' + PRE_PREFIX + '_' + FINE_PREFIX
        for prefix in order.values():
            for path in Path(im_dir).glob('*.tif'):
                if path.name.startswith(prefix): #startswith检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
                    paths.append(path.expanduser().resolve()) #获得目录
                    break
    # print(paths)
    assert len(paths) == 3 or len(paths) == 4

    return paths


def load_image_pair(directory: Path,id_n2, mode: Mode):
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(directory,id_n2, mode=mode)

    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()#.astype(np.float32)  # C*H*W (numpy.ndarray)
            images.append(im) #包含了所有影像

    # 对数据的尺寸进行验证
    assert images[1].shape[1] * SCALE_FACTOR == images[0].shape[1]
    assert images[1].shape[2] * SCALE_FACTOR == images[0].shape[2]

    return images





class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """
    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, mode=Mode.TRAINING):
        super(PatchSet, self).__init__() #父类：Dataset
        patch_size = make_tuple(patch_size) #元组
        patch_stride = make_tuple(patch_stride) if patch_stride else patch_size

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.mode = mode
        #img=glob .glob(os.path .join(self .root_dir ,'*') )
        #print((img))

        self.image_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)

        #print(self.image_dirs)

        # 计算出图像进行分块以后的patches的数目
        # self.num_patches_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        # self.num_patches_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        # self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y
        self.num_patches_x = image_size[0] // patch_size[0]
        self.num_patches_y = image_size[0] // patch_size[0]
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

    @staticmethod
    def transform(im):
        data = im.astype(np.float32)
        data = torch.from_numpy(data)
        out = data.mul_(0.0001)
        return out

    def map_index(self, index):  # 获得各个块的起始坐标
        id_n = index // (self.num_patches_x * self.num_patches_y)
        # residual = index % (self.num_patches_x * self.num_patches_y)
        # id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        # id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        id_n2 = index % (self.num_patches_x * self.num_patches_y)
        id_n2 = "{:02d}".format(id_n2)
        return id_n, id_n2

    def __getitem__(self, index): #实现索引数据集中的某一个数据（Dataset中的必要）
        #训练的时候我们想随机地获取数据训练模型，那么这个index就是0 ~ n-1中随机抽出来的，想训练max_iters次，那么就从0~n-1中随机抽max_iters次，每次得到的就是index的具体值；
        # 测试的时候我们不需要打乱数据样本的顺序，因此index就是逐个遍历range(n)所得。
        #在这里，index是从总的块数中随机取（低分辨率块）

        id_n, id_n2 = self.map_index(index)  # id_n是图像文件夹索引，如1号文件夹

        images = load_image_pair(self.image_dirs[id_n], id_n2, mode=self.mode)


        patches = [None] * len(images) #输出一下看看是多少，等于[None,None,None,None]

        scales = [SCALE_FACTOR,1,SCALE_FACTOR,SCALE_FACTOR]
        for i in range(len(patches)):#range(len(patches))=[1,4]
            scale = scales[i]
            im = images[i][:,
                 0 * scale:(0 + self.patch_size[0]) * scale,
                 0 * scale:(0 + self.patch_size[1]) * scale]
            patches[i] = self.transform(im) #除10000

        del images[:]
        del images
        return patches

    def __len__(self): #返回数据集大小（Dataset中的必要）
        return self.num_patches
