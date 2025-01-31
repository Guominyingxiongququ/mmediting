import glob
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np

from .base_dn_dataset import BaseDNDataset
from .registry import DATASETS


@DATASETS.register_module()
class DNFolderVideoDataset(BaseDNDataset):
    """General dataset for video DN, used for sliding-window framework.

    The dataset loads several LQ (Low-Quality) frames and one GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    This dataset takes an annotation file specifying the sequences used in
    training or test. If no annotation file is provided, it assumes all video
    sequences under the root directory are used for training or test.

    In the annotation file (.txt), each line contains:

        1. image name (no file extension);
        2. number of frames in the sequence (in the same folder)

    Examples:

    ::

        calendar/00000000 41
        calendar/00000001 41
        ...
        calendar/00000040 41
        city/00000000 34
        ...


    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        ann_file (str): The path to the annotation file. If None, we assume
            that all sequences in the folder is used. Default: None.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{:08d}'.
        metric_average_mode (str): The way to compute the average metric.
            If 'clip', we first compute an average value for each clip, and
            then average the values from different clips. If 'all', we
            compute the average of all frames. Default: 'clip'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `True`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 num_input_frames,
                 pipeline,
                 scale,
                 ann_file=None,
                 filename_tmpl='{:08d}',
                 metric_average_mode='clip',
                 test_mode=True):
        super().__init__(pipeline, scale, test_mode)

        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        if metric_average_mode not in ['clip', 'all']:
            raise ValueError('metric_average_mode can only be "clip" or '
                             f'"all", but got {metric_average_mode}.')

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.ann_file = ann_file
        self.filename_tmpl = filename_tmpl
        self.metric_average_mode = metric_average_mode

        self.data_infos = self.load_annotations()

    def _load_annotations_from_file(self):
        self.folders = {}
        data_infos = []

        ann_list = mmcv.list_from_file(self.ann_file)
        for ann in ann_list:
            key, max_frame_num = ann.strip().rsplit(' ', 1)
            sequence = osp.basename(key)
            if sequence not in self.folders:
                self.folders[sequence] = int(max_frame_num)

            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    num_input_frames=self.num_input_frames,
                    max_frame_num=int(max_frame_num)))

        return data_infos

    def load_annotations(self):
        """Load annoations for the dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """

        if self.ann_file:
            return self._load_annotations_from_file()

        self.folders = {}
        data_infos = []

        sequences = sorted(glob.glob(osp.join(self.lq_folder, '*')))
        sequences = [s.split('/')[-1] for s in sequences]

        for sequence in sequences:
            seq_dir = osp.join(self.lq_folder, sequence)

            max_frame_num = len(list(mmcv.utils.scandir(seq_dir)))
            self.folders[sequence] = max_frame_num

            for i in range(0, max_frame_num):
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        key=f'{sequence}/{self.filename_tmpl.format(i)}',
                        num_input_frames=self.num_input_frames,
                        max_frame_num=max_frame_num))

        return data_infos

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_results = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_results[metric].append(val)
        for metric, val_list in eval_results.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        if self.metric_average_mode == 'clip':
            for metric, values in eval_results.items():
                start_idx = 0
                metric_avg = 0
                for _, num_img in self.folders.items():
                    end_idx = start_idx + num_img
                    folder_values = values[start_idx:end_idx]
                    metric_avg += np.mean(folder_values)
                    start_idx = end_idx

                eval_results[metric] = metric_avg / len(self.folders)
        else:
            eval_results = {
                metric: sum(values) / len(self)
                for metric, values in eval_results.items()
            }

        return eval_results
