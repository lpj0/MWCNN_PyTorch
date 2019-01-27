import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio
from scipy.misc import imresize

import torch
import torch.utils.data as data
import h5py

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        # if train:
        #     self.args.ext = 'img'
        #     self.dir_hr = '/share/data/genk/Train_Data/'
        #     self.ext = '.png'

        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        # mat = h5py.File('/share/data/genk/imdb_Gray.mat')
        if train:
            mat = h5py.File('E:\MyCode\data_gen\data_gen_vdsr\imdb_gray\imdb_gray.mat')
            self.args.ext = 'mat'
            self.hr_data = mat['images']['labels'][:,:,:,:]
            self.num = self.hr_data.shape[0]

        # kernel_train = sio.loadmat('data/kernels_matrix_ms.mat')

        # kernels = kernel_train['ks']['kernels']
        #
        # self.kernel_train = kernel_train['ks']['kernels'][0][0][0]
        # self.covmat_train = kernel_train['ks']['sigma_xy_ori'][0][0][0]

        # kernel_set = self.kernel_train[0]
        # num = kernel_set.shape[3]

        # ker = kernel.astype(np.float32)

        if self.split == 'test':
            self._set_filesystem(args.dir_data)

        # self._set_filesystem(args.dir_data)

        # def _load_bin():
        #     self.images_hr = np.load(self._name_hrbin())
            # self.images_lr = [
            #     np.load(self._name_lrbin(s)) for s in self.scale
            # ]

        self.images_hr = self._scan()

        # if args.ext == 'img' or benchmark:
        #     self.images_hr = self._scan()
        #     # self.images_hr, self.images_lr = self._scan()
        # elif args.ext.find('sep') >= 0:
        #     self.images_hr = self._scan()
        #     # self.images_hr, self.images_lr = self._scan()
        #     if args.ext.find('reset') >= 0:
        #         print('Preparing seperated binary files')
        #         for v in self.images_hr:
        #             hr = misc.imread(v)
        #             name_sep = v.replace(self.ext, '.npy')
        #             np.save(name_sep, hr)
        #         # for si, s in enumerate(self.scale):
        #         #     for v in self.images_lr[si]:
        #         #         lr = misc.imread(v)
        #         #         name_sep = v.replace(self.ext, '.npy')
        #         #         np.save(name_sep, lr)
        #
        #     self.images_hr = [
        #         v.replace(self.ext, '.npy') for v in self.images_hr
        #     ]
        #     # self.images_lr = [
        #     #     [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
        #     #     for i in range(len(self.scale))
        #     # ]
        #
        # elif args.ext.find('bin') >= 0:
        #     try:
        #         if args.ext.find('reset') >= 0:
        #             raise IOError
        #         print('Loading a binary file')
        #         _load_bin()
        #     except:
        #         print('Preparing a binary file')
        #         bin_path = os.path.join(self.apath, 'bin')
        #         if not os.path.isdir(bin_path):
        #             os.mkdir(bin_path)
        #
        #         # list_hr, list_lr = self._scan()
        #         list_hr = self._scan()
        #         hr = [misc.imread(f) for f in list_hr]
        #         np.save(self._name_hrbin(), hr)
        #         del hr
        #         # for si, s in enumerate(self.scale):
        #         #     lr_scale = [misc.imread(f) for f in list_lr[si]]
        #         #     np.save(self._name_lrbin(s), lr_scale)
        #         #     del lr_scale
        #         _load_bin()
        # else:
        #     print('Please define data type')

    def _scan(self):
        raise NotImplementedError
    #
    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    # def _name_hrbin(self):
    #     raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        if self.train:

            # filter, scale_factor, quality_factor, sigma0, sigma1, covmat = common.gen_factor(self.kernel_train, self.covmat_train)
            #
            # filter = np.squeeze(filter.astype(np.float32))
            # covmat = np.squeeze(covmat.astype(np.float32))
            # covmat_tensor = torch.from_numpy(covmat) * 16.0 / 255.0
            #
            # sigma, lr, hr = self._get_patch(hr, filter, filename, scale_factor, quality_factor, sigma0, sigma1)
            # sigma, lr, hr = common.set_channel([sigma, lr, hr], self.args.n_colors)
            # lh, lw = lr.shape[:2]
            # # hh, hw = hr.shape[:2]
            #
            # covmat_tensor = torch.mul(covmat_tensor.view(3, 1, 1), torch.ones([1, lh, lw]).float())
            #
            # scale_factor_tensor = torch.ones([1, lh, lw]).float()
            # scale_factor_tensor.mul_(int(scale_factor) / 255.0)
            #
            # quality_factor_tensor = torch.ones([1, lh, lw]).float()
            # quality_factor_tensor.mul_(int(110 - quality_factor) / 255.0)


            lr, hr, scale = self._get_patch(hr, filename)
            lh, lw = lr.shape[:2]
            # scale_factor_tensor = torch.ones([1, lh, lw]).float() * (scale / 80.0)

            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)

            # lr_tensor = torch.cat((lr_tensor, quality_factor_tensor), 0)0
            return lr_tensor, hr_tensor, filename
        else:
            #scale = 2
            # scale = self.scale[self.idx_scale]
            lr, hr, _ = self._get_patch(hr, filename)

            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)

            return lr_tensor, hr_tensor, filename


    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        # lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]

        if self.args.ext == 'img' or self.benchmark:
            filename = hr

            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            # lr = np.load(lr)
            hr = np.load(hr)
        elif self.args.ext == 'mat' or self.train:
            hr = self.hr_data[idx, :, :, :]
            #print(hr.shape)
            hr = np.squeeze(hr.transpose((1, 2, 0)))
            filename = str(idx) + '.png'
        else:
            filename = str(idx + 1)




        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return hr, filename

    def _get_patch(self, hr, filename):
        patch_size = self.args.patch_size

        if self.train:
            scale = self.scale[0]
            lr, hr = common.get_patch_noise(
                hr, patch_size, scale
            )
            lr, hr = common.augment([lr, hr])
            return lr, hr, scale
        else:
            scale = self.scale[0]
            lr, hr = common.add_img_noise(
                hr, scale
            )
            return lr, hr, scale
            # lr = common.add_noise(lr, self.args.noise)


    def _get_patch_test(self, hr, scale):

        ih, iw = hr.shape[0:2]
        lr = imresize(imresize(hr, [int(ih/scale), int(iw/scale)], 'bicubic'), [ih, iw], 'bicubic')
        ih = ih // 8 * 8
        iw = iw // 8 * 8
        hr = hr[0:ih, 0:iw, :]
        lr = lr[0:ih, 0:iw, :]

        return lr, hr




    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

