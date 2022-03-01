import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.ndimage import generate_binary_structure, binary_erosion, label

import skimage
import skimage.io
import skimage.feature
from skimage.filters import gaussian
from skimage import dtype_limits, img_as_float

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage.filters import _gaussian_kernel1d



class MNISTCanny(nn.Module):
    def __init__(self, sigma=3, high_threshold=1.6, low_threshold=1.4, use_quantiles=False):
        super(MNISTCanny, self).__init__()
        
        # see https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L202-L206
        truncate = 4  # default value in https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L211
        order = 0  # also default value
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)
        kernel = _gaussian_kernel1d(sigma, order, lw)[::-1].copy()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        self.gaussian_conv = nn.Conv1d(32, 32, kernel.size, groups=32, padding=lw, bias=False)
        self.gaussian_conv.weight.data[:] = torch.FloatTensor(kernel)

        self.mask = nn.Parameter(torch.ones(1, 1, 32, 32))

        self.sobel_major_conv = nn.Conv1d(32, 32, 3, groups=32, padding=0, bias=False)
        self.sobel_major_conv.weight.data[:] = torch.FloatTensor([-1, 0, 1])

        self.sobel_minor_conv = nn.Conv1d(32, 32, 3, groups=32, padding=0, bias=False)
        self.sobel_minor_conv.weight.data[:] = torch.FloatTensor([1, 2, 1])

        # config
        self.eps = 1e-9  # add to sqrt() to prevent nan grad
        self.gamma = 0.005  # margin
        
    def gaussian(self, x):
        # x.shape: [N, C, H, W] (?, 1, 28, 28)
        
        # first 1d conv, axis = 0, vertical direction
        x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 28, 28)
        x = x.view(x.shape[0], 32, 32)  # x.shape: [N, W, H] (?, 28, 28)
        x = self.gaussian_conv(x)  # x.shape: [N, W, H] (?, 28, 28)
        x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 28, 28)

        # second 1d conv, axis=1, horizontal direction
        x = x.view(x.shape[0], 32, 32)  # x.shape: [N, H, W] (?, 28, 28)
        x = self.gaussian_conv(x)  # x.shape: [N, H, W] (?, 28, 28)
        x = x.view(x.shape[0], 1, 32, 32)  # x.shape: [N, C, H, W] (?, 1, 28, 28)
        return x
    
    def sobel(self, x, axis):
        # x.shape: [N, C, H, W]
        
        if axis == 1:
            # horizontal first: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L363
            x = x.view(1, 32, 32)  # x.shape: [N, H, W] (?, 28, 28)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, H, W], (?, 28, 30)
            x = self.sobel_major_conv(x)  # x.shape: [N, H, W] (?, 28, 28)
            x = x.view(x.shape[0], 1, 32, 32)  # x.shape: [N, C, H, W] (?, 1, 28, 28)
            
            # then vertical: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L366
            x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 28, 28)
            x = x.view(x.shape[0], 32, 32)  # x.shape: [N, W, H] (?, 28, 28)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, W, H], (?, 28, 30)
            x = self.sobel_minor_conv(x)  # x.shape: [N, W, H] (?, 28, 28)
            x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 28, 28)
        
        elif axis == 0:
            # vertical first
            x = x.transpose(2, 3)
            x = x.view(x.shape[0], 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_major_conv(x)
            x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)
            
            # then horizontal
            x = x.view(1, 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_minor_conv(x)
            x = x.view(x.shape[0], 1, 32, 32)
        else:
            raise NotImplementedError('Unknown axis {}'.format(axis))
        return x
        
        
    def forward(self, x):
        # x.shape: [N, C, H, W] (?, 1, 28, 28)
        
        # simulate https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py: canny()
        
        # L175-L180
        bleed_over = self.gaussian(self.mask)
        x = self.gaussian(x)
        x = x / (bleed_over + 1e-12)

        # debug_gaussian = x.data.clone()

        jsobel = self.sobel(x, axis=1)
        isobel = self.sobel(x, axis=0)
        abs_isobel = torch.abs(isobel)
        abs_jsobel = torch.abs(jsobel)
        magnitude2 = isobel ** 2 + jsobel ** 2
        magnitude = torch.sqrt(magnitude2 + self.eps)

        # L186-L188
        #
        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        #
        assert x.shape[0] == 1
        s = generate_binary_structure(2, 2)
        mask = self.mask.detach().cpu().numpy()[0, 0]  # mask.shape: [28, 28]
        eroded_mask = binary_erosion(mask, s, border_value=0)
        eroded_mask = eroded_mask & (magnitude2.detach().cpu().numpy()[0, 0] > 0)  # replace magnitude by magnitude2
        eroded_mask = torch.ByteTensor(eroded_mask.astype(np.uint8)).to(x.device)

        # L195-L212
        #
        #--------- Find local maxima --------------
        #
        local_maxima = torch.zeros(x.shape).byte().to(x.device)
        #----- 0 to 45 degrees ------
        pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
        pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, 1:, :][pts[:, :, :-1, :]]
        c2 = magnitude[:, :, 1:, 1:][pts[:, :, :-1, :-1]]
        m = magnitude[pts]
        w = abs_jsobel[pts] / (abs_isobel[pts] + self.eps)
        c_plus = c2 * w + c1 * (1 - w) <= m
        
        s_0_45_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, :-1, :][pts[:, :, 1:, :]]
        c2 = magnitude[:, :, :-1, :-1][pts[:, :, 1:, 1:]]
        c_minus = c2 * w + c1 * (1 - w) <= m
        
        s_0_45_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_0_45 = torch.max(s_0_45_1, s_0_45_2)
        
        local_maxima[pts] = c_plus & c_minus
        
        # L216-L228
        #----- 45 to 90 degrees ------
        # Mix diagonal and vertical
        #
        pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
        pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, :, 1:][pts[:, :, :, :-1]]
        c2 = magnitude[:, :, 1:, 1:][pts[:, :, :-1, :-1]]
        m = magnitude[pts]
        w = abs_isobel[pts] / abs_jsobel[pts]
        c_plus = c2 * w + c1 * (1 - w) <= m
        
        s_45_90_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, :, :-1][pts[:, :, :, 1:]]
        c2 = magnitude[:, :, :-1, :-1][pts[:, :, 1:, 1:]]
        c_minus = c2 * w + c1 * (1 - w) <= m
        
        s_45_90_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_45_90 = torch.max(s_45_90_1, s_45_90_2)
        
        local_maxima[pts] = c_plus & c_minus
        
        # L232-L244
        #----- 90 to 135 degrees ------
        # Mix anti-diagonal and vertical
        #
        pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
        pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1a = magnitude[:, :, :, 1:][pts[:, :, :, :-1]]
        c2a = magnitude[:, :, :-1, 1:][pts[:, :, 1:, :-1]]
        m = magnitude[pts]
        w = abs_isobel[pts] / abs_jsobel[pts]
        c_plus = c2a * w + c1a * (1.0 - w) <= m
        
        s_90_135_1 = F.relu(-m + self.gamma + (c2a * w + c1a * (1.0 - w)))
        
        c1 = magnitude[:, :, :, :-1][pts[:, :, :, 1:]]
        c2 = magnitude[:, :, 1:, :-1][pts[:, :, :-1, 1:]]
        c_minus = c2 * w + c1 * (1.0 - w) <= m
        
        s_90_135_2 = F.relu(-m + self.gamma + (c2a * w + c1a * (1.0 - w)))
        s_90_135 = torch.max(s_90_135_1, s_90_135_2)
        
        local_maxima[pts] = c_plus & c_minus
        
        # L248-L260
        #----- 135 to 180 degrees ------
        # Mix anti-diagonal and anti-horizontal
        #
        pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
        pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, :-1, :][pts[:, :, 1:, :]]
        c2 = magnitude[:, :, :-1, 1:][pts[:, :, 1:, :-1]]
        m = magnitude[pts]
        w = abs_jsobel[pts] / abs_isobel[pts]
        c_plus = c2 * w + c1 * (1 - w) <= m
        
        s_135_180_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, 1:, :][pts[:, :, :-1, :]]
        c2 = magnitude[:, :, 1:, :-1][pts[:, :, :-1, 1:]]
        c_minus = c2 * w + c1 * (1 - w) <= m
        
        s_135_180_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_135_180 = torch.max(s_135_180_1, s_135_180_2)
        
        local_maxima[pts] = c_plus & c_minus


        # Final part
        local_maxima_np = (local_maxima.data.clone().cpu().numpy()==1)[0][0]
        magnitude_np = magnitude.data.clone().cpu().numpy()[0][0]
        high_mask = local_maxima_np & (magnitude_np >= self.high_threshold)
        low_mask = local_maxima_np & (magnitude_np >= self.low_threshold)

        strel = np.ones((3, 3), bool)
        labels, count = label(low_mask, strel)
        if count == 0:
            mask_final = low_mask
        else:
            sums = (np.array(ndi.sum(high_mask, labels,
                                 np.arange(count, dtype=np.int32) + 1),
                         copy=False, ndmin=1))
            good_label = np.zeros((count + 1,), bool)
            good_label[1:] = sums > 0
            output_mask = good_label[labels]
            mask_final = output_mask

        mask_final = mask_final.astype(np.float32)

        magnitude = magnitude * torch.FloatTensor(mask_final).cuda()
        test = magnitude[magnitude != 0] 
        magnitude = magnitude / magnitude.max().item()

        return s_0_45, s_45_90, s_90_135, s_135_180, local_maxima, test, magnitude


class FMNISTCanny(nn.Module):
    def __init__(self, sigma=3, high_threshold=0.2, low_threshold=0.1, thres=0., use_quantiles=False):
        super(FMNISTCanny, self).__init__()
        
        # see https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L202-L206
        truncate = 4  # default value in https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L211
        order = 0  # also default value
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)
        kernel = _gaussian_kernel1d(sigma, order, lw)[::-1].copy()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.thres = thres

        self.gaussian_conv = nn.Conv1d(28, 28, kernel.size, groups=28, padding=lw, bias=False)
        self.gaussian_conv.weight.data[:] = torch.FloatTensor(kernel)

        self.mask = nn.Parameter(torch.ones(1, 1, 28, 28))

        self.sobel_major_conv = nn.Conv1d(28, 28, 3, groups=28, padding=0, bias=False)
        self.sobel_major_conv.weight.data[:] = torch.FloatTensor([-1, 0, 1])

        self.sobel_minor_conv = nn.Conv1d(28, 28, 3, groups=28, padding=0, bias=False)
        self.sobel_minor_conv.weight.data[:] = torch.FloatTensor([1, 2, 1])

        # config
        self.eps = 1e-9  # add to sqrt() to prevent nan grad
        self.gamma = 0.005  # margin
        
    def gaussian(self, x):
        # x.shape: [N, C, H, W] (?, 1, 28, 28)
        
        # first 1d conv, axis = 0, vertical direction
        x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 28, 28)
        x = x.view(x.shape[0], 28, 28)  # x.shape: [N, W, H] (?, 28, 28)
        x = self.gaussian_conv(x)  # x.shape: [N, W, H] (?, 28, 28)
        x = x.view(x.shape[0], 1, 28, 28).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 28, 28)

        # second 1d conv, axis=1, horizontal direction
        x = x.view(x.shape[0], 28, 28)  # x.shape: [N, H, W] (?, 28, 28)
        x = self.gaussian_conv(x)  # x.shape: [N, H, W] (?, 28, 28)
        x = x.view(x.shape[0], 1, 28, 28)  # x.shape: [N, C, H, W] (?, 1, 28, 28)
        return x
    
    def sobel(self, x, axis):
        # x.shape: [N, C, H, W]
        
        if axis == 1:
            # horizontal first: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L363
            x = x.view(1, 28, 28)  # x.shape: [N, H, W] (?, 28, 28)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, H, W], (?, 28, 30)
            x = self.sobel_major_conv(x)  # x.shape: [N, H, W] (?, 28, 28)
            x = x.view(x.shape[0], 1, 28, 28)  # x.shape: [N, C, H, W] (?, 1, 28, 28)
            
            # then vertical: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L366
            x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 28, 28)
            x = x.view(x.shape[0], 28, 28)  # x.shape: [N, W, H] (?, 28, 28)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, W, H], (?, 28, 30)
            x = self.sobel_minor_conv(x)  # x.shape: [N, W, H] (?, 28, 28)
            x = x.view(x.shape[0], 1, 28, 28).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 28, 28)
        
        elif axis == 0:
            # vertical first
            x = x.transpose(2, 3)
            x = x.view(x.shape[0], 28, 28)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_major_conv(x)
            x = x.view(x.shape[0], 1, 28, 28).transpose(2, 3)
            
            # then horizontal
            x = x.view(1, 28, 28)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_minor_conv(x)
            x = x.view(x.shape[0], 1, 28, 28)
        else:
            raise NotImplementedError('Unknown axis {}'.format(axis))
        return x
        
        
    def forward(self, x):
        # x.shape: [N, C, H, W] (?, 1, 28, 28)
        
        # simulate https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py: canny()
        
        # L175-L180
        bleed_over = self.gaussian(self.mask)
        x = self.gaussian(x)
        x = x / (bleed_over + 1e-12)

        # debug_gaussian = x.data.clone()

        jsobel = self.sobel(x, axis=1)
        isobel = self.sobel(x, axis=0)
        abs_isobel = torch.abs(isobel)
        abs_jsobel = torch.abs(jsobel)
        magnitude2 = isobel ** 2 + jsobel ** 2
        magnitude = torch.sqrt(magnitude2 + self.eps)

        inter = (magnitude > self.thres).type(torch.FloatTensor).cuda()
        magnitude = magnitude * inter

        # L186-L188
        #
        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        #
        assert x.shape[0] == 1
        s = generate_binary_structure(2, 2)
        mask = self.mask.detach().cpu().numpy()[0, 0]  # mask.shape: [28, 28]
        eroded_mask = binary_erosion(mask, s, border_value=0)
        eroded_mask = eroded_mask & (magnitude2.detach().cpu().numpy()[0, 0] > 0)  # replace magnitude by magnitude2
        eroded_mask = torch.ByteTensor(eroded_mask.astype(np.uint8)).to(x.device)

        # L195-L212
        #
        #--------- Find local maxima --------------
        #
        local_maxima = torch.zeros(x.shape).byte().to(x.device)
        #----- 0 to 45 degrees ------
        pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
        pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, 1:, :][pts[:, :, :-1, :]]
        c2 = magnitude[:, :, 1:, 1:][pts[:, :, :-1, :-1]]
        m = magnitude[pts]
        w = abs_jsobel[pts] / (abs_isobel[pts] + self.eps)
        c_plus = c2 * w + c1 * (1 - w) <= m
        
        s_0_45_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, :-1, :][pts[:, :, 1:, :]]
        c2 = magnitude[:, :, :-1, :-1][pts[:, :, 1:, 1:]]
        c_minus = c2 * w + c1 * (1 - w) <= m
        
        s_0_45_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_0_45 = torch.max(s_0_45_1, s_0_45_2)
        
        local_maxima[pts] = c_plus & c_minus
        
        # L216-L228
        #----- 45 to 90 degrees ------
        # Mix diagonal and vertical
        #
        pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
        pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, :, 1:][pts[:, :, :, :-1]]
        c2 = magnitude[:, :, 1:, 1:][pts[:, :, :-1, :-1]]
        m = magnitude[pts]
        w = abs_isobel[pts] / abs_jsobel[pts]
        c_plus = c2 * w + c1 * (1 - w) <= m
        
        s_45_90_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, :, :-1][pts[:, :, :, 1:]]
        c2 = magnitude[:, :, :-1, :-1][pts[:, :, 1:, 1:]]
        c_minus = c2 * w + c1 * (1 - w) <= m
        
        s_45_90_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_45_90 = torch.max(s_45_90_1, s_45_90_2)
        
        local_maxima[pts] = c_plus & c_minus
        
        # L232-L244
        #----- 90 to 135 degrees ------
        # Mix anti-diagonal and vertical
        #
        pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
        pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1a = magnitude[:, :, :, 1:][pts[:, :, :, :-1]]
        c2a = magnitude[:, :, :-1, 1:][pts[:, :, 1:, :-1]]
        m = magnitude[pts]
        w = abs_isobel[pts] / abs_jsobel[pts]
        c_plus = c2a * w + c1a * (1.0 - w) <= m
        
        s_90_135_1 = F.relu(-m + self.gamma + (c2a * w + c1a * (1.0 - w)))
        
        c1 = magnitude[:, :, :, :-1][pts[:, :, :, 1:]]
        c2 = magnitude[:, :, 1:, :-1][pts[:, :, :-1, 1:]]
        c_minus = c2 * w + c1 * (1.0 - w) <= m
        
        s_90_135_2 = F.relu(-m + self.gamma + (c2a * w + c1a * (1.0 - w)))
        s_90_135 = torch.max(s_90_135_1, s_90_135_2)
        
        local_maxima[pts] = c_plus & c_minus
        
        # L248-L260
        #----- 135 to 180 degrees ------
        # Mix anti-diagonal and anti-horizontal
        #
        pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
        pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, :-1, :][pts[:, :, 1:, :]]
        c2 = magnitude[:, :, :-1, 1:][pts[:, :, 1:, :-1]]
        m = magnitude[pts]
        w = abs_jsobel[pts] / abs_isobel[pts]
        c_plus = c2 * w + c1 * (1 - w) <= m
        
        s_135_180_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, 1:, :][pts[:, :, :-1, :]]
        c2 = magnitude[:, :, 1:, :-1][pts[:, :, :-1, 1:]]
        c_minus = c2 * w + c1 * (1 - w) <= m
        
        s_135_180_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_135_180 = torch.max(s_135_180_1, s_135_180_2)
        
        local_maxima[pts] = c_plus & c_minus


        # Final part
        local_maxima_np = (local_maxima.data.clone().cpu().numpy()==1)[0][0]
        magnitude_np = magnitude.data.clone().cpu().numpy()[0][0]
        high_mask = local_maxima_np & (magnitude_np >= self.high_threshold)
        low_mask = local_maxima_np & (magnitude_np >= self.low_threshold)

        strel = np.ones((3, 3), bool)
        labels, count = label(low_mask, strel)
        if count == 0:
            mask_final = low_mask
        else:
            sums = (np.array(ndi.sum(high_mask, labels,
                                 np.arange(count, dtype=np.int32) + 1),
                         copy=False, ndmin=1))
            good_label = np.zeros((count + 1,), bool)
            good_label[1:] = sums > 0
            output_mask = good_label[labels]
            mask_final = output_mask

        mask_final = mask_final.astype(np.float32)

        # magnitude = magnitude * torch.FloatTensor(mask_final).cuda()
        magnitude = torch.FloatTensor(mask_final).cuda() + magnitude - magnitude.detach()

        test = torch.ones_like(magnitude).cuda()
        # test[magnitude != 0] = magnitude[magnitude !=0]
        # magnitude = magnitude / test.data

        return s_0_45, s_45_90, s_90_135, s_135_180, local_maxima, test, magnitude