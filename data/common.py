import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

import os
import torch.nn as nn
import math
import time

from scipy.misc import imread, imresize, imsave, toimage
from scipy.ndimage import convolve
from PIL import Image

def gen_factor(kernel_train, covmat_train):
    sigma0 = np.random.randint(0, 4, 1) * 0.02
    sigma1 = np.random.randint(0, 9, 1) * 0.02
    quality_factor = np.random.randint(0, 10, 1) * 5 + 50
    a = np.random.rand()
    ss = (np.random.randint(1, 6, 1)) / 5
    if a < 0.05:
        scale_factor = ss + 1
        kernel_set = kernel_train[0]
        num = kernel_set.shape[3]
        idx = np.random.randint(0, num, 1)
        filter = kernel_set[:, :, :, idx]
        covmat = covmat_train[0][:, idx]


    elif (a >= 0.05) & (a < 0.3):
        if np.random.rand() < 0.8:
            scale_factor = 2
        else:
            scale_factor = ss + 2
        kernel_set = kernel_train[1]
        num = kernel_set.shape[3]
        idx = np.random.randint(0, num, 1)
        filter = kernel_set[:, :, :, idx]
        covmat = covmat_train[1][:, idx]
    elif (a >= 0.3) & (a < 0.55):
        if np.random.rand() < 0.8:
            scale_factor = 3
        else:
            scale_factor = ss + 3
        kernel_set = kernel_train[2]
        num = kernel_set.shape[3]
        idx = np.random.randint(0, num, 1)
        filter = kernel_set[:, :, :, idx]
        covmat = covmat_train[2][:, idx]
    elif (a >= 0.55) & (a < 0.75):
        if np.random.rand() < 0.8:
            scale_factor = 4
        else:
            scale_factor = ss + 4
        kernel_set = kernel_train[3]
        num = kernel_set.shape[3]
        idx = np.random.randint(0, num, 1)
        filter = kernel_set[:, :, :, idx]
        covmat = covmat_train[3][:, idx]
    else:
        if np.random.rand() < 0.8:
            scale_factor = 8
        else:
            scale_factor = (np.random.randint(1, 15, 1)) / 5 + 5
        kernel_set = kernel_train[4]
        num = kernel_set.shape[3]
        idx = np.random.randint(0, num, 1)
        filter = kernel_set[:, :, :, idx]
        covmat = covmat_train[4][:, idx]

    return filter, scale_factor, quality_factor, sigma0, sigma1, covmat



def im_process(x, name, filter, scale_factor, quality_factor, sigma0, sigma1):
    # filter = filter / 3.0
    # ff = np.zeros([3, 15, 15])
    # fff = np.zeros([3, 1, 15, 15])
    # # ff[0, :, :] = filter
    # # ff[1, :, :] = filter
    # # ff[2, :, :] = filter
    # fff[0, 0, :, :] = filter
    # fff[1, 0, :, :] = filter
    # fff[2, 0, :, :] = filter

    sz = x.shape
    # print(sz)
    y = x
    # x = imfilter(x, fff)
    x = np.float32(x)/255.0
    # # x = convolve(x, ff, mode='nearest')
    x[:, :, 0] = convolve(x[:, :, 0], filter, mode='nearest')
    x[:, :, 1] = convolve(x[:, :, 1], filter, mode='nearest')
    x[:, :, 2] = convolve(x[:, :, 2], filter, mode='nearest')
    y = y[7:sz[0]-8, 7:sz[1]-8, :]
    x = x[7:sz[0]-8, 7:sz[1]-8, :]
    # print(scale_factor)

    x = imresize(x, [int((sz[0]-15)/scale_factor), int((sz[1]-15)/scale_factor)], 'bicubic')
    x = x / 225.0
    sz = x.shape
    # print(sz)
    # print(quality_factor)
    # x = x / 255.0
    sigma = sigma0 + sigma1 * x
    x = x + np.random.randn(sz[0], sz[1], 3) * sigma
    x = x * 255
    x = x.clip(0, 255)
    tt, _ = math.modf(time.time())

    # tt = int(np.random.randint(0,1e13,1))

    out = "tmp/" + name + '%d'%(tt*1e16) + '.jpg'
    x = Image.fromarray(np.uint8(x))
    # x = toimage(x)
    # imsave(out, x, format='jpeg', quality=int(quality_factor))

    x.save(out, 'JPEG', quality=int(quality_factor))
    x = imread(out)
    os.remove(out)

    return sigma * 255.0, x, y

def imfilter(x_np, filter):
    # x = torch.float(x)
    # sz = x.size()
    x = np.ascontiguousarray(x_np.transpose((2, 0, 1)))
    x = torch.from_numpy(x).float()
    x.mul_(1.0 / 255.0)
    sz = x.size()

    filter = torch.from_numpy(filter).float()

    # x = x.view(1, sz[0], sz[1], sz[2])
    # filter = torch.float()
    x = nn.functional.conv2d(x.view(1, 3, sz[1], sz[2]), filter, padding = 7, groups=3) #contiguous().
    # x = x.contiguous().view(3, sz[1], sz[2])
    x = x.squeeze().permute(1, 2, 0)
    return x.numpy()




def get_patch(img_tar, patch_size, name, filter, scale_factor, quality_factor, sigma0, sigma1):
    # ih, iw = img_in.shape[:2]
    patch_size = patch_size+15
    th, tw = img_tar.shape[:2]
    tp = patch_size
    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)

    # p = scale if multi_scale else 1
    # tp = p * patch_size
    # ip = tp // scale

    # ix = random.randrange(0, iw - ip + 1)
    # iy = random.randrange(0, ih - ip + 1)
    # tx, ty = scale * ix, scale * iy

    # img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]



    return im_process(img_tar, name, filter, scale_factor, quality_factor, sigma0, sigma1)

def get_patch_noise(img_tar, patch_size, noise_level):

    ih, iw = img_tar.shape[0:2]
    a = np.random.rand(1)[0] * 0.8 + 0.2
    if (ih * a < patch_size) | (a * iw < patch_size):
        a = np.random.rand(1)[0] * 0.33 + 0.67
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')
        # th, tw = img_tar.shape[:2]
    else:
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')

    th, tw = img_tar.shape[:2]
    tp = patch_size

    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)
    img_tar = np.expand_dims(img_tar, axis=2)
    #print(img_tar.shape)
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    noises = np.random.normal(scale=noise_level, size=img_tar.shape)
    noises = noises.round()
    img_tar_noise = img_tar.astype(np.int16) + noises.astype(np.int16)
    # x_noise = x_noise.clip(0, 255).astype(np.uint8)

    return img_tar_noise, img_tar


def add_img_noise(img_tar, noise_level):
    img_tar = np.expand_dims(img_tar, axis=2)
    #print(img_tar.shape)
    ih, iw = img_tar.shape[0:2]
    ih = int(ih//8*8)
    iw = int(iw//8*8)
    img_tar = img_tar[0:ih, 0:iw, :]
    noises = np.random.normal(scale=noise_level, size=img_tar.shape)
    noises = noises.round()
    img_tar_noise = img_tar.astype(np.int16) + noises.astype(np.int16)
    # x_noise = x_noise.clip(0, 255).astype(np.uint8)

    return img_tar_noise, img_tar



def get_patch_bic(img_tar, patch_size, scale_factor):
    # ih, iw = img_in.shape[:2]
    # patch_size = patch_size

    ih, iw = img_tar.shape[0:2]
    a = np.random.rand(1)[0] * 0.8 + 0.2
    if (ih*a < patch_size) | (a*iw < patch_size):
        a = np.random.rand(1)[0] * 0.33 + 0.67
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')
        # th, tw = img_tar.shape[:2]
    else:
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')


    th, tw = img_tar.shape[:2]
    tp = patch_size

    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)
    img_lr = imresize(imresize(img_tar, [int(th/scale_factor), int(tw/scale_factor)], 'bicubic'), [th, tw], 'bicubic')

    # p = scale if multi_scale else 1
    # tp = p * patch_size
    # ip = tp // scale

    # ix = random.randrange(0, iw - ip + 1)
    # iy = random.randrange(0, ih - ip + 1)
    # tx, ty = scale * ix, scale * iy

    # img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    img_lr = img_lr[ty:ty + tp, tx:tx + tp, :]



    return img_lr, img_tar



def get_patch_bic2(img_tar, patch_size, scale_factor, name):
    ih, iw = img_tar.shape[0:2]
    a = np.random.rand(1)[0] * 0.8 + 0.2
    if (ih * a < patch_size) or (a * iw < patch_size):
        a = np.random.rand(1)[0] * 0.1 + 0.9
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')
        # th, tw = img_tar.shape[:2]
    else:
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')
    # patch_size = patch_size
    # a = np.random.rand(1)[0] * 0.5 + 0.5
    # img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')

    th, tw = img_tar.shape[:2]
    tp = patch_size

    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)
    img_lr = imresize(imresize(img_tar, [int(th/scale_factor), int(tw/scale_factor)], 'bicubic'), [th, tw], 'bicubic')

    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    img_lr = img_lr[ty:ty + tp, tx:tx + tp, :]



    return img_lr, img_tar

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.0)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]
