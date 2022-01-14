# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
import numpy as np
import mmcv
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class EDVRDN(BasicRestorer):
    """EDVRDN model for video denoise.

    EDVRDN: Video Denoise with Enhanced Deformable Convolutional Networks.

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
        self.with_tsa = generator.get('with_tsa', False)
        self.step_counter = 0  # count training steps

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        if self.step_counter == 0 and self.with_tsa:
            if self.train_cfg is None or (self.train_cfg is not None and
                                          'tsa_iter' not in self.train_cfg):
                raise KeyError(
                    'In TSA mode, train_cfg must contain "tsa_iter".')
            # only train TSA module at the beginging if with TSA module
            for k, v in self.generator.named_parameters():
                if 'fusion' not in k:
                    v.requires_grad = False

        if self.with_tsa and (self.step_counter == self.train_cfg.tsa_iter):
            # train all the parameters
            for v in self.generator.parameters():
                v.requires_grad = True

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        Args:
            imgs (Tensor): Input images.

        Returns:
            Tensor: Restored image.
        """
        out = self.generator(imgs)
        return out

    def test_big_size_raw(self, input_data, patch_h=256, patch_w=256,
                          patch_h_overlap=64, patch_w_overlap=64):

        H = input_data.shape[3]
        W = input_data.shape[4]
        # input_data shape n, t, c, h, w
        # output shape n, c, h, w
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
                if w_index == 1:
                    test_horizontal_result[:, :, :, w_begin:w_end] = \
                        output_patch
                else:
                    for i in range(patch_w_overlap):
                        test_horizontal_result[:, :, :, w_begin+i] = \
                            test_horizontal_result[:, :, :, w_begin + i]\
                            * (patch_w_overlap-1-i)/(patch_w_overlap-1)\
                            + output_patch[:, :, :, i] * i/(patch_w_overlap-1)
                    test_horizontal_result[:, :, :, w_begin+patch_w_overlap:w_end] = \
                    output_patch[:, :, :, patch_w_overlap:]
                w_index += 1
            test_patch = input_data[:, :, :, h_begin:h_end,-patch_w:] 
            output_patch = self.generator(test_patch)
            output_patch = \
            output_patch.cpu().detach().numpy().astype(np.float32)
            last_range = w_end-(W-patch_w)       
            for i in range(last_range):
                test_horizontal_result[:, :, :, W-patch_w+i] = \
                test_horizontal_result[:, :, :, W-patch_w+i]\
                *(last_range-1-i)/(last_range-1)\
                +output_patch[:, :, :, i]*i/(last_range-1)
            test_horizontal_result[:, :, :, w_end:] = output_patch[:, :, :, last_range:]       

            if h_index == 1:
                test_result[:, :, h_begin:h_end, :] = test_horizontal_result
            else:
                for i in range(patch_h_overlap):
                    test_result[:, :, h_begin+i, :] = \
                    test_result[:, :, h_begin+i, :]*(patch_h_overlap-1-i)\
                    /(patch_h_overlap-1)+test_horizontal_result[:, :, i, :] * i/(patch_h_overlap-1)
                test_result[:, :, h_begin+patch_h_overlap:h_end, :] = test_horizontal_result[:, :, patch_h_overlap:, :]
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
            if w_index == 1:
                test_horizontal_result[:, :, :, w_begin:w_end] = output_patch
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:, :, :, w_begin+i] = \
                    test_horizontal_result[:, :, :, w_begin+i]*\
                    (patch_w_overlap-1-i)/(patch_w_overlap-1)\
                    +output_patch[:, :, :, i]*i/(patch_w_overlap-1)
                test_horizontal_result[:, :, :, w_begin+patch_w_overlap:w_end] = output_patch[:, :, :, patch_w_overlap:]  
            w_index += 1
        test_patch = input_data[:, :, :,  -patch_h:, -patch_w:]         
        output_patch = self.generator(test_patch)
        output_patch = output_patch.cpu().detach().numpy().astype(np.float32)
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:, :, :, W-patch_w+i] = \
                test_horizontal_result[:, :, :, W-patch_w+i]*(last_range-1-i)\
                /(last_range-1)+output_patch[:, :, :, i]*i/(last_range-1)
        test_horizontal_result[:, :, :, w_end:] = output_patch[:, :, :, last_range:] 

        last_last_range = h_end-(H-patch_h)
        for i in range(last_last_range):
            test_result[:, :, H-patch_w+i, :] = \
                test_result[:, :, H-patch_w+i, :]*(last_last_range-1-i)/(last_last_range-1)\
                +test_horizontal_result[:, :, i, :]*i/(last_last_range-1)
        test_result[:, :, h_end:, :] = test_horizontal_result[:, :, last_last_range:, :]
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
        output = self.test_big_size_raw(lq, patch_h, patch_w, \
            patch_h_overlap, patch_w_overlap)
        # lq shape n, t, c, h, w
        # gt shape n, c, h, w
        print(self.test_cfg)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(torch.tensor(output), gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        save_image=True
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
