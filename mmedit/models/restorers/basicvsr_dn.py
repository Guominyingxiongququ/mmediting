# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class BasicVSRDN(BasicRestorer):
    """BasicVSRDN model for video denoise.

    Note that this model is used for IconVSR.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    output_i = tensor2img(output[:, i, :, :, :])
                    gt_i = tensor2img(gt[:, i, :, :, :])
                    avg.append(self.allowed_metrics[metric](
                        output_i, gt_i, crop_border, convert_to=convert_to))
                eval_result[metric] = np.mean(avg)
            elif output.ndim == 4:  # an image: (n, c, t, w), for Vimeo-90K-T
                output_img = tensor2img(output)
                gt_img = tensor2img(gt)
                value = self.allowed_metrics[metric](
                    output_img, gt_img, crop_border, convert_to=convert_to)
                eval_result[metric] = value

        return eval_result

    def test_big_size_raw(self, input_data, patch_h=256, patch_w=256,
                          patch_h_overlap=64, patch_w_overlap=64):

        H = input_data.shape[3]
        W = input_data.shape[4]
        # input_data shape n, t, c, h, w 
        # output shape n, t, c, h, w
        t = input_data.shape[1]
        center_idx = int(t/2)
        test_result = np.zeros((input_data.shape[0], 3, H, W))
        h_index = 1
        while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
            test_horizontal_result = np.zeros((input_data.shape[0],
                                              3, patch_h, W))
            h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
            h_end = patch_h*h_index-patch_h_overlap*(h_index-1)
            w_index = 1
            w_end = 0
            while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
                w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
                w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
                test_patch = input_data[:, :, :, h_begin:h_end, w_begin:w_end]
                output_patch = self.generator(test_patch)
                output_patch = \
                    output_patch.cpu().detach().numpy().astype(np.float32)
                output_patch = output_patch[:, center_idx, :, :, :]
                if w_index == 1:
                    test_horizontal_result[:, :, :, w_begin:w_end] = \
                        output_patch
                else:
                    for i in range(patch_w_overlap):
                        test_horizontal_result[:, :, :, w_begin+i] = \
                            test_horizontal_result[:, :, :, w_begin + i]\
                            * (patch_w_overlap-1-i)/(patch_w_overlap-1)\
                            + output_patch[:, :, :, i] * i/(patch_w_overlap-1)
                    cur_begin = w_begin+patch_w_overlap
                    test_horizontal_result[:, :, :, cur_begin:w_end] = \
                        output_patch[:, :, :, patch_w_overlap:]
                w_index += 1
            test_patch = input_data[:, :, :, h_begin:h_end, -patch_w:]
            output_patch = self.generator(test_patch)
            output_patch = \
                output_patch.cpu().detach().numpy().astype(np.float32)
            output_patch = output_patch[:, center_idx, :, :, :]
            last_range = w_end-(W-patch_w)
            for i in range(last_range):
                term1 = test_horizontal_result[:, :, :, W-patch_w+i]
                rate1 = (last_range-1-i)/(last_range-1)
                term2 = output_patch[:, :, :, i]
                rate2 = i/(last_range-1)
                test_horizontal_result[:, :, :, W-patch_w+i] = \
                    term1*rate1+term2*rate2
            test_horizontal_result[:, :, :, w_end:] = \
                output_patch[:, :, :, last_range:]

            if h_index == 1:
                test_result[:, :, h_begin:h_end, :] = test_horizontal_result
            else:
                for i in range(patch_h_overlap):
                    term1 = test_result[:, :, h_begin+i, :]
                    rate1 = (patch_h_overlap-1-i)/(patch_h_overlap-1)
                    term2 = test_horizontal_result[:, :, i, :]
                    rate2 = i/(patch_h_overlap-1)
                    test_result[:, :, h_begin+i, :] = \
                        term1 * rate1 + term2 * rate2
                test_result[:, :, h_begin+patch_h_overlap:h_end, :] = \
                    test_horizontal_result[:, :, patch_h_overlap:, :]
            h_index += 1

        test_horizontal_result = np.zeros((input_data.shape[0], 3, patch_h, W))
        w_index = 1
        while (patch_w * w_index - patch_w_overlap * (w_index-1)) < W:
            w_begin = patch_w * (w_index-1) - patch_w_overlap * (w_index-1)
            w_end = patch_w * w_index - patch_w_overlap * (w_index-1)
            test_patch = input_data[:, :, :, -patch_h:, w_begin:w_end]
            output_patch = self.generator(test_patch)
            output_patch = \
                output_patch.cpu().detach().numpy().astype(np.float32)
            output_patch = output_patch[:, center_idx, :, :, :]
            if w_index == 1:
                test_horizontal_result[:, :, :, w_begin:w_end] = output_patch
            else:
                for i in range(patch_w_overlap):
                    term1 = test_horizontal_result[:, :, :, w_begin+i]
                    rate1 = (patch_w_overlap-1-i)/(patch_w_overlap-1)
                    term2 = output_patch[:, :, :, i]
                    rate2 = i/(patch_w_overlap-1)
                    test_horizontal_result[:, :, :, w_begin+i] = \
                        term1*rate1+term2*rate2
                cur_begin = w_begin+patch_w_overlap
                test_horizontal_result[:, :, :, cur_begin:w_end] = \
                    output_patch[:, :, :, patch_w_overlap:]
            w_index += 1
        test_patch = input_data[:, :, :,  -patch_h:, -patch_w:]
        output_patch = self.generator(test_patch)
        output_patch = output_patch.cpu().detach().numpy().astype(np.float32)
        output_patch = output_patch[:, center_idx, :, :, :]
        last_range = w_end-(W-patch_w)
        for i in range(last_range):
            term1 = test_horizontal_result[:, :, :, W-patch_w+i]
            rate1 = (last_range-1-i)/(last_range-1)
            term2 = output_patch[:, :, :, i]
            rate2 = i/(last_range-1)
            test_horizontal_result[:, :, :, W-patch_w+i] = \
                term1*rate1+term2*rate2
        test_horizontal_result[:, :, :, w_end:] = \
            output_patch[:, :, :, last_range:]

        last_last_range = h_end-(H-patch_h)
        for i in range(last_last_range):
            term1 = test_result[:, :, H-patch_w+i, :]
            rate1 = (last_last_range-1-i)/(last_last_range-1)
            term2 = test_horizontal_result[:, :, i, :]
            rate2 = i/(last_last_range-1)
            test_result[:, :, H-patch_w+i, :] = \
                term1*rate1+term2*rate2
        cur_result = test_horizontal_result[:, :, last_last_range:, :]
        test_result[:, :, h_end:, :] = cur_result
        return test_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        patch_h = 256
        patch_w = 256
        patch_h_overlap = 64
        patch_w_overlap = 64
        print(self.test_cfg)
        patch_h = self.test_cfg.get('patch_size', 256)
        patch_w = patch_h
        with torch.no_grad():
            output = self.test_big_size_raw(lq, patch_h, patch_w,
                                        patch_h_overlap, patch_w_overlap)
        # lq shape n, t, c, h, w
        # gt shape n, c, h, w
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(torch.tensor(output), gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        save_image = True
        if save_image:
            gt_path = meta[0]['gt_path'][0]
            folder_name = meta[0]['key'].split('/')[0]
            isp_name = meta[0]['key'].split('/')[1]
            frame_name = osp.splitext(osp.basename(gt_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name, isp_name,
                                     f'{frame_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, folder_name,
                                     f'{frame_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(torch.tensor(output)), save_path)

        return results
