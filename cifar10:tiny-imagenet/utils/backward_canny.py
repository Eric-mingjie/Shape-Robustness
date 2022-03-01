import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import generate_binary_structure, binary_erosion, label
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import _gaussian_kernel1d

class ThresholdF(Function):
    def __init__(self, threshold):
        super(ThresholdF, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        self.save_for_backward(input)
        a = torch.zeros_like(input)
        output = torch.where(input < self.threshold, a, input)
        return output

    def backward(self, grad_output):
        result, = self.saved_tensors
        return grad_output * result

def selfTF(threshold, input):
    return ThresholdF(threshold)(input)

class Canny_Net(nn.Module):
    def __init__(self, sigma=1.0, high_threshold=0.2, low_threshold=0.1, thres = 0.3, use_quantiles=False):
        super(Canny_Net, self).__init__()

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
        self.thres = thres

    def gaussian(self, x):
        # x.shape: [N, C, H, W] (?, 1, 32, 32)

        # first 1d conv, axis = 0, vertical direction
        x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 32, 32)
        x = x.view(x.shape[0], 32, 32)  # x.shape: [N, W, H] (?, 32, 32)
        x = self.gaussian_conv(x)  # x.shape: [N, W, H] (?, 32, 32)
        x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 32, 32)

        # second 1d conv, axis=1, horizontal direction
        x = x.view(x.shape[0], 32, 32)  # x.shape: [N, H, W] (?, 32, 32)
        x = self.gaussian_conv(x)  # x.shape: [N, H, W] (?, 32, 32)
        x = x.view(x.shape[0], 1, 32, 32)  # x.shape: [N, C, H, W] (?, 1, 32, 32)
        return x

    def sobel(self, x, axis):
        # x.shape: [N, C, H, W]

        if axis == 1:
            # horizontal first: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L363
            x = x.view(x.shape[0], 32, 32)  # x.shape: [N, H, W] (?, 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, H, W], (?, 32, 30)
            x = self.sobel_major_conv(x)  # x.shape: [N, H, W] (?, 32, 32)
            x = x.view(x.shape[0], 1, 32, 32)  # x.shape: [N, C, H, W] (?, 1, 32, 32)

            # then vertical: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L366
            x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 32, 32)
            x = x.view(x.shape[0], 32, 32)  # x.shape: [N, W, H] (?, 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, W, H], (?, 32, 30)
            x = self.sobel_minor_conv(x)  # x.shape: [N, W, H] (?, 32, 32)
            x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 32, 32)

        elif axis == 0:
            # vertical first
            x = x.transpose(2, 3)
            x = x.view(x.shape[0], 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_major_conv(x)
            x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)

            # then horizontal
            x = x.view(x.shape[0], 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_minor_conv(x)
            x = x.view(x.shape[0], 1, 32, 32)
        else:
            raise NotImplementedError('Unknown axis {}'.format(axis))
        return x

    def forward(self, x):
        # x.shape: [N, C, H, W] (?, 1, 32, 32)
        x = x * 0.5 + 0.5
        if x.shape[1] == 3:
            x = x[:,0:1,:,:] * 0.299 + x[:,1:2,:,:] * 0.587 + x[:,2:3,:,:] * 0.114

        # simulate https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py: canny()

        # L175-L180
        bleed_over = self.gaussian(self.mask)
        x = self.gaussian(x)
        x = x / (bleed_over + 1e-12)

        # debug_gaussian = x.data.clone()

        jsobel = self.sobel(x, axis=1)
        isobel = self.sobel(x, axis=0)

        # print(1, torch.max(jsobel))
        # print(2, torch.min(jsobel))

        abs_isobel = torch.abs(isobel)
        abs_jsobel = torch.abs(jsobel)
        magnitude2 = isobel ** 2 + jsobel ** 2
        magnitude = torch.sqrt(magnitude2 + self.eps)
        # magnitude = selfTF(self.thres, magnitude)

        # L186-L188
        #
        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        #
        # assert x.shape[0] == 1
        s = generate_binary_structure(2, 2)
        mask = self.mask.detach().cpu().numpy()[0, 0]  # mask.shape: [32, 32]
        eroded_mask = binary_erosion(mask, s, border_value=0)
        eroded_mask = eroded_mask & (magnitude2.detach().cpu().numpy()[0, 0] > 0)  # replace magnitude by magnitude2
        eroded_mask = torch.ByteTensor(eroded_mask.astype(np.uint8)).to(x.device)

        # L195-L212
        #
        # --------- Find local maxima --------------
        #
        local_maxima = torch.zeros(x.shape).byte().to(x.device)
        # ----- 0 to 45 degrees ------
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
        # ----- 45 to 90 degrees ------
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
        # ----- 90 to 135 degrees ------
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
        # ----- 135 to 180 degrees ------
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
        # local_maxima_np = (local_maxima.data.clone().cpu().numpy() == 1)[0][0]
        # magnitude_np = magnitude.data.clone().cpu().numpy()[0][0]
        local_maxima_np = (local_maxima.data.clone().cpu().numpy() == 1)
        magnitude_np = magnitude.data.clone().cpu().numpy()
        high_mask = local_maxima_np & (magnitude_np >= self.high_threshold)
        low_mask = local_maxima_np & (magnitude_np >= self.low_threshold)

        strel = np.ones((3, 3), bool)
        mask_final_list = []
        for i in range(x.shape[0]):
            labels, count = label(low_mask[i][0], strel)
            if count == 0:
                mask_final = low_mask[i][0]
            else:
                sums = (np.array(ndi.sum(high_mask[i][0], labels,
                                         np.arange(count, dtype=np.int32) + 1),
                                 copy=False, ndmin=1))
                good_label = np.zeros((count + 1,), bool)
                good_label[1:] = sums > 0
                output_mask = good_label[labels]
                mask_final = output_mask
            mask_final_list.append([mask_final])

        mask_final = np.concatenate((mask_final_list), 0)
        mask_final = np.reshape(mask_final.astype(np.float32),(x.shape[0], 1, 32, 32))

        # magnitude = magnitude * torch.FloatTensor(mask_final).cuda()
        magnitude = torch.FloatTensor(mask_final).cuda() + magnitude - magnitude.detach()
        test = magnitude[magnitude != 0]
        # magnitude = magnitude / magnitude.max().item()
        # if magnitude.max().item() == 0:
        #     print('yes')

        magnitude = (magnitude - 0.5) / 0.5
        # return s_0_45, s_45_90, s_90_135, s_135_180, local_maxima, test, magnitude
        return magnitude

    def vis_forward(self, x):
        # x.shape: [N, C, H, W] (?, 1, 32, 32)
        x = x * 0.5 + 0.5
        if x.shape[1] == 3:
            x = x[:,0:1,:,:] * 0.299 + x[:,1:2,:,:] * 0.587 + x[:,2:3,:,:] * 0.114

        # simulate https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py: canny()

        # L175-L180
        bleed_over = self.gaussian(self.mask)
        x = self.gaussian(x)
        x = x / (bleed_over + 1e-12)

        vis1 = x.clone().detach()

        # debug_gaussian = x.data.clone()

        jsobel = self.sobel(x, axis=1)
        isobel = self.sobel(x, axis=0)

        # print(1, torch.max(jsobel))
        # print(2, torch.min(jsobel))

        abs_isobel = torch.abs(isobel)
        abs_jsobel = torch.abs(jsobel)
        magnitude2 = isobel ** 2 + jsobel ** 2
        magnitude = torch.sqrt(magnitude2 + self.eps)
        # magnitude = selfTF(self.thres, magnitude)

        vis2 = magnitude.clone().detach()

        # L186-L188
        #
        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        #
        # assert x.shape[0] == 1
        s = generate_binary_structure(2, 2)
        mask = self.mask.detach().cpu().numpy()[0, 0]  # mask.shape: [32, 32]
        eroded_mask = binary_erosion(mask, s, border_value=0)
        eroded_mask = eroded_mask & (magnitude2.detach().cpu().numpy()[0, 0] > 0)  # replace magnitude by magnitude2
        eroded_mask = torch.ByteTensor(eroded_mask.astype(np.uint8)).to(x.device)

        # L195-L212
        #
        # --------- Find local maxima --------------
        #
        local_maxima = torch.zeros(x.shape).byte().to(x.device)
        # ----- 0 to 45 degrees ------
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
        # ----- 45 to 90 degrees ------
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
        # ----- 90 to 135 degrees ------
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
        # ----- 135 to 180 degrees ------
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
        # local_maxima_np = (local_maxima.data.clone().cpu().numpy() == 1)[0][0]
        # magnitude_np = magnitude.data.clone().cpu().numpy()[0][0]
        local_maxima_np = (local_maxima.data.clone().cpu().numpy() == 1)
        magnitude_np = magnitude.data.clone().cpu().numpy()
        high_mask = local_maxima_np & (magnitude_np >= self.high_threshold)
        low_mask = local_maxima_np & (magnitude_np >= self.low_threshold)

        vis3 = high_mask.copy()
        vis4 = low_mask.copy()

        strel = np.ones((3, 3), bool)
        mask_final_list = []
        for i in range(x.shape[0]):
            labels, count = label(low_mask[i][0], strel)
            if count == 0:
                mask_final = low_mask[i][0]
            else:
                sums = (np.array(ndi.sum(high_mask[i][0], labels,
                                         np.arange(count, dtype=np.int32) + 1),
                                 copy=False, ndmin=1))
                good_label = np.zeros((count + 1,), bool)
                good_label[1:] = sums > 0
                output_mask = good_label[labels]
                mask_final = output_mask
            mask_final_list.append([mask_final])

        mask_final = np.concatenate((mask_final_list), 0)
        mask_final = np.reshape(mask_final.astype(np.float32),(x.shape[0], 1, 32, 32))

        # magnitude = magnitude * torch.FloatTensor(mask_final).cuda()
        magnitude = torch.FloatTensor(mask_final).cuda() + magnitude - magnitude.detach()
        test = magnitude[magnitude != 0]
        # magnitude = magnitude / magnitude.max().item()
        # if magnitude.max().item() == 0:
        #     print('yes')
        magnitude = (magnitude - 0.5) / 0.5
        # return s_0_45, s_45_90, s_90_135, s_135_180, local_maxima, test, magnitude
        return magnitude, vis1, vis2, vis3, vis4

class CifarNet(nn.Module):
    def __init__(self, sigma=3, high_threshold=0.2, low_threshold=0.1, thres=0., use_quantiles=False):
        super(CifarNet, self).__init__()
        
        # see https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L202-L206
        truncate = 4  # default value in https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L211
        order = 0  # also default value
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)
        kernel = _gaussian_kernel1d(sigma, order, lw)[::-1].copy()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.thres = thres

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
        # x.shape: [N, C, H, W] (?, 1, 32, 32)
        
        # first 1d conv, axis = 0, vertical direction
        x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 32, 32)
        x = x.view(x.shape[0], 32, 32)  # x.shape: [N, W, H] (?, 32, 32)
        x = self.gaussian_conv(x)  # x.shape: [N, W, H] (?, 32, 32)
        x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 32, 32)

        # second 1d conv, axis=1, horizontal direction
        x = x.view(x.shape[0], 32, 32)  # x.shape: [N, H, W] (?, 32, 32)
        x = self.gaussian_conv(x)  # x.shape: [N, H, W] (?, 32, 32)
        x = x.view(x.shape[0], 1, 32, 32)  # x.shape: [N, C, H, W] (?, 1, 32, 32)
        return x
    
    def sobel(self, x, axis):
        # x.shape: [N, C, H, W]
        
        if axis == 1:
            # horizontal first: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L363
            x = x.view(1, 32, 32)  # x.shape: [N, H, W] (?, 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, H, W], (?, 32, 30)
            x = self.sobel_major_conv(x)  # x.shape: [N, H, W] (?, 32, 32)
            x = x.view(x.shape[0], 1, 32, 32)  # x.shape: [N, C, H, W] (?, 1, 32, 32)
            
            # then vertical: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L366
            x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 32, 32)
            x = x.view(x.shape[0], 32, 32)  # x.shape: [N, W, H] (?, 32, 32)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, W, H], (?, 32, 30)
            x = self.sobel_minor_conv(x)  # x.shape: [N, W, H] (?, 32, 32)
            x = x.view(x.shape[0], 1, 32, 32).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 32, 32)
        
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
        # x.shape: [N, C, H, W] (?, 1, 32, 32)
        x = x * 0.5 + 0.5
        # simulate https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py: canny()
        x = (x[:,0, :, :] + x[:,1, :, :] + x[:,2, :, :])/3
        x = x.view(x.shape[0], 1, 32, 32)
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
        mask = self.mask.detach().cpu().numpy()[0, 0]  # mask.shape: [32, 32]
        eroded_mask = binary_erosion(mask, s, border_value=0)
        eroded_mask = eroded_mask & (magnitude2.detach().cpu().numpy()[0, 0] > 0)  # replace magnitude by magnitude2
        eroded_mask = torch.ByteTensor(eroded_mask.astype(np.uint8)).to(x.device)

        # L195-L212
        #
        #--------- Find local maxima --------------
        #
        local_maxima = torch.zeros(x.shape).to(x.device)
        #----- 0 to 45 degrees ------
        pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
        pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, 1:, :][pts[:, :, :-1, :]]
        c2 = magnitude[:, :, 1:, 1:][pts[:, :, :-1, :-1]]
        m = magnitude[pts]
        w = abs_jsobel[pts] / (abs_isobel[pts] + self.eps)
        c_plus = (c2 * w + c1 * (1 - w) <= m)
        
        s_0_45_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, :-1, :][pts[:, :, 1:, :]]
        c2 = magnitude[:, :, :-1, :-1][pts[:, :, 1:, 1:]]
        c_minus = (c2 * w + 1 * (1 - w) <= m)
        
        s_0_45_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_0_45 = torch.max(s_0_45_1, s_0_45_2)
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
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
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
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
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
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
        
        local_maxima[pts] = (c_plus & c_minus).float()


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
        magnitude = (magnitude - 0.5) /0.5
        return magnitude


class ImageNet(nn.Module):
    def __init__(self, sigma=3, high_threshold=0.2, low_threshold=0.1, thres=0., use_quantiles=False):
        super(ImageNet, self).__init__()
        
        # see https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L202-L206
        truncate = 4  # default value in https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L211
        order = 0  # also default value
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)
        kernel = _gaussian_kernel1d(sigma, order, lw)[::-1].copy()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.thres = thres

        self.gaussian_conv = nn.Conv1d(224, 224, kernel.size, groups=224, padding=lw, bias=False)
        self.gaussian_conv.weight.data[:] = torch.FloatTensor(kernel)

        self.mask = nn.Parameter(torch.ones(1, 1, 224, 224))

        self.sobel_major_conv = nn.Conv1d(224, 224, 3, groups=224, padding=0, bias=False)
        self.sobel_major_conv.weight.data[:] = torch.FloatTensor([-1, 0, 1])

        self.sobel_minor_conv = nn.Conv1d(224, 224, 3, groups=224, padding=0, bias=False)
        self.sobel_minor_conv.weight.data[:] = torch.FloatTensor([1, 2, 1])

        # config
        self.eps = 1e-9  # add to sqrt() to prevent nan grad
        self.gamma = 0.005  # margin
        
    def gaussian(self, x):
        # x.shape: [N, C, H, W] (?, 1, 32, 32)
        
        # first 1d conv, axis = 0, vertical direction
        x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 32, 32)
        x = x.view(x.shape[0], 224, 224)  # x.shape: [N, W, H] (?, 224, 224)
        x = self.gaussian_conv(x)  # x.shape: [N, W, H] (?, 224, 224)
        x = x.view(x.shape[0], 1, 224, 224).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 224, 224)

        # second 1d conv, axis=1, horizontal direction
        x = x.view(x.shape[0], 224, 224)  # x.shape: [N, H, W] (?, 224, 224)
        x = self.gaussian_conv(x)  # x.shape: [N, H, W] (?, 224, 224)
        x = x.view(x.shape[0], 1, 224, 224)  # x.shape: [N, C, H, W] (?, 1, 224, 224)
        return x
    
    def sobel(self, x, axis):
        # x.shape: [N, C, H, W]
        
        if axis == 1:
            # horizontal first: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L363
            x = x.view(1, 224, 224)  # x.shape: [N, H, W] (?, 224, 224)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, H, W], (?, 224, 30)
            x = self.sobel_major_conv(x)  # x.shape: [N, H, W] (?, 224, 224)
            x = x.view(x.shape[0], 1, 224, 224)  # x.shape: [N, C, H, W] (?, 1, 224, 224)
            
            # then vertical: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L366
            x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 224, 224)
            x = x.view(x.shape[0], 224, 224)  # x.shape: [N, W, H] (?, 224, 224)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, W, H], (?, 224, 30)
            x = self.sobel_minor_conv(x)  # x.shape: [N, W, H] (?, 224, 224)
            x = x.view(x.shape[0], 1, 224, 224).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 224, 224)
        
        elif axis == 0:
            # vertical first
            x = x.transpose(2, 3)
            x = x.view(x.shape[0], 224, 224)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_major_conv(x)
            x = x.view(x.shape[0], 1, 224, 224).transpose(2, 3)
            
            # then horizontal
            x = x.view(1, 224, 224)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_minor_conv(x)
            x = x.view(x.shape[0], 1, 224, 224)
        else:
            raise NotImplementedError('Unknown axis {}'.format(axis))
        return x
        
        
    def forward(self, x):
        # x.shape: [N, C, H, W] (?, 1, 224, 224)
        x = x * 0.5 + 0.5
        # simulate https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py: canny()
        x = (x[:,0, :, :] + x[:,1, :, :] + x[:,2, :, :])/3
        x = x.view(x.shape[0], 1, 224, 224)
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
        mask = self.mask.detach().cpu().numpy()[0, 0]  # mask.shape: [224, 224]
        eroded_mask = binary_erosion(mask, s, border_value=0)
        eroded_mask = eroded_mask & (magnitude2.detach().cpu().numpy()[0, 0] > 0)  # replace magnitude by magnitude2
        eroded_mask = torch.ByteTensor(eroded_mask.astype(np.uint8)).to(x.device)

        # L195-L212
        #
        #--------- Find local maxima --------------
        #
        local_maxima = torch.zeros(x.shape).to(x.device)
        #----- 0 to 45 degrees ------
        pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
        pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, 1:, :][pts[:, :, :-1, :]]
        c2 = magnitude[:, :, 1:, 1:][pts[:, :, :-1, :-1]]
        m = magnitude[pts]
        w = abs_jsobel[pts] / (abs_isobel[pts] + self.eps)
        c_plus = (c2 * w + c1 * (1 - w) <= m)
        
        s_0_45_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, :-1, :][pts[:, :, 1:, :]]
        c2 = magnitude[:, :, :-1, :-1][pts[:, :, 1:, 1:]]
        c_minus = (c2 * w + 1 * (1 - w) <= m)
        
        s_0_45_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_0_45 = torch.max(s_0_45_1, s_0_45_2)
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
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
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
        # L2224-L244
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
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
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
        
        local_maxima[pts] = (c_plus & c_minus).float()


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
        magnitude = (magnitude-0.5) / 0.5
        return magnitude


class TinyNet(nn.Module):
    def __init__(self, sigma=3, high_threshold=0.2, low_threshold=0.1, thres=0., use_quantiles=False):
        super(TinyNet, self).__init__()
        
        # see https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L202-L206
        truncate = 4  # default value in https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L211
        order = 0  # also default value
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)
        kernel = _gaussian_kernel1d(sigma, order, lw)[::-1].copy()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.thres = thres

        self.gaussian_conv = nn.Conv1d(128, 128, kernel.size, groups=128, padding=lw, bias=False)
        self.gaussian_conv.weight.data[:] = torch.FloatTensor(kernel)

        self.mask = nn.Parameter(torch.ones(1, 1, 128, 128))

        self.sobel_major_conv = nn.Conv1d(128, 128, 3, groups=128, padding=0, bias=False)
        self.sobel_major_conv.weight.data[:] = torch.FloatTensor([-1, 0, 1])

        self.sobel_minor_conv = nn.Conv1d(128, 128, 3, groups=128, padding=0, bias=False)
        self.sobel_minor_conv.weight.data[:] = torch.FloatTensor([1, 2, 1])

        # config
        self.eps = 1e-9  # add to sqrt() to prevent nan grad
        self.gamma = 0.005  # margin
        
    def gaussian(self, x):
        # x.shape: [N, C, H, W] (?, 1, 32, 32)
        
        # first 1d conv, axis = 0, vertical direction
        x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 32, 32)
        x = x.view(x.shape[0], 128, 128)  # x.shape: [N, W, H] (?, 128, 128)
        x = self.gaussian_conv(x)  # x.shape: [N, W, H] (?, 128, 128)
        x = x.view(x.shape[0], 1, 128, 128).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 128, 128)

        # second 1d conv, axis=1, horizontal direction
        x = x.view(x.shape[0], 128, 128)  # x.shape: [N, H, W] (?, 128, 128)
        x = self.gaussian_conv(x)  # x.shape: [N, H, W] (?, 128, 128)
        x = x.view(x.shape[0], 1, 128, 128)  # x.shape: [N, C, H, W] (?, 1, 128, 128)
        return x
    
    def sobel(self, x, axis):
        # x.shape: [N, C, H, W]
        
        if axis == 1:
            # horizontal first: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L363
            x = x.view(1, 128, 128)  # x.shape: [N, H, W] (?, 128, 128)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, H, W], (?, 128, 30)
            x = self.sobel_major_conv(x)  # x.shape: [N, H, W] (?, 128, 128)
            x = x.view(x.shape[0], 1, 128, 128)  # x.shape: [N, C, H, W] (?, 1, 128, 128)
            
            # then vertical: https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py#L366
            x = x.transpose(2, 3)  # x.shape: [N, C, W, H] (?, 1, 128, 128)
            x = x.view(x.shape[0], 128, 128)  # x.shape: [N, W, H] (?, 128, 128)
            x = F.pad(x, (1, 1), mode='replicate')  # x.shape: [N, W, H], (?, 128, 30)
            x = self.sobel_minor_conv(x)  # x.shape: [N, W, H] (?, 128, 128)
            x = x.view(x.shape[0], 1, 128, 128).transpose(2, 3)  # x.shape: [N, C, H, W] (?, 1, 128, 128)
        
        elif axis == 0:
            # vertical first
            x = x.transpose(2, 3)
            x = x.view(x.shape[0], 128, 128)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_major_conv(x)
            x = x.view(x.shape[0], 1, 128, 128).transpose(2, 3)
            
            # then horizontal
            x = x.view(1, 128, 128)
            x = F.pad(x, (1, 1), mode='replicate')
            x = self.sobel_minor_conv(x)
            x = x.view(x.shape[0], 1, 128, 128)
        else:
            raise NotImplementedError('Unknown axis {}'.format(axis))
        return x
        
        
    def forward(self, x):
        # x.shape: [N, C, H, W] (?, 1, 128, 128)
        x = x * 0.5 + 0.5
        # simulate https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py: canny()
        x = (x[:,0, :, :] + x[:,1, :, :] + x[:,2, :, :])/3
        x = x.view(x.shape[0], 1, 128, 128)
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
        mask = self.mask.detach().cpu().numpy()[0, 0]  # mask.shape: [128, 128]
        eroded_mask = binary_erosion(mask, s, border_value=0)
        eroded_mask = eroded_mask & (magnitude2.detach().cpu().numpy()[0, 0] > 0)  # replace magnitude by magnitude2
        eroded_mask = torch.ByteTensor(eroded_mask.astype(np.uint8)).to(x.device)

        # L195-L212
        #
        #--------- Find local maxima --------------
        #
        local_maxima = torch.zeros(x.shape).to(x.device)
        #----- 0 to 45 degrees ------
        pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
        pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
        pts = pts_plus | pts_minus
        pts = eroded_mask & pts
        c1 = magnitude[:, :, 1:, :][pts[:, :, :-1, :]]
        c2 = magnitude[:, :, 1:, 1:][pts[:, :, :-1, :-1]]
        m = magnitude[pts]
        w = abs_jsobel[pts] / (abs_isobel[pts] + self.eps)
        c_plus = (c2 * w + c1 * (1 - w) <= m)
        
        s_0_45_1 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        
        c1 = magnitude[:, :, :-1, :][pts[:, :, 1:, :]]
        c2 = magnitude[:, :, :-1, :-1][pts[:, :, 1:, 1:]]
        c_minus = (c2 * w + 1 * (1 - w) <= m)
        
        s_0_45_2 = F.relu(-m + self.gamma + (c2 * w + c1 * (1 - w)))
        s_0_45 = torch.max(s_0_45_1, s_0_45_2)
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
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
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
        # L2128-L244
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
        
        local_maxima[pts] = (c_plus & c_minus).float()
        
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
        
        local_maxima[pts] = (c_plus & c_minus).float()


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
        magnitude = (magnitude-0.5) / 0.5
        return magnitude