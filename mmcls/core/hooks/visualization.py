import os.path as osp

import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from torchvision.utils import save_image
from mmcv.image import tensor2imgs
import numpy as np
import cv2
import os
import copy
@HOOKS.register_module()
class WeightVisualizationHook(Hook):
    """Visualization hook.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        output_dir (str): The file path to store visualizations.
        res_name_list (str): The list contains the name of results in outputs
            dict. The results in outputs dict must be a torch.Tensor with shape
            (n, c, h, w).
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
    """

    def __init__(self,
                 output_dir,
                 interval=-1,
                 filename_tmpl='iter_{:08d}',
                 rerange=True,
                 bgr2rgb=True,
                 nrow=1,
                 padding=4):
        self.output_dir = output_dir
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

    def concat_tile(self, im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    def norm(self, data, imgs):
        data = data.detach().cpu().numpy().astype('float32')
        heatmaps = [0] * data.shape[0]        
        for i in range(data.shape[0]):
            # for j in range(data.shape[1]):
            j=0
            maxv = data[i, j, :, :].max()
            minv = data[i, j, :, :].min()
            data[i, j, :, :] = (data[i, j, :, :] - minv) / (maxv - minv) * 255
            gray = data[i, j, :, :].astype(np.uint8)
            gray = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            heatmaps[i] = (0.5 * imgs[i] + 0.5 * gray).astype(np.uint8)
        return heatmaps

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if runner.iter == 1:
            rm_pre = osp.join(runner.work_dir, self.output_dir, '*')
            os.popen('rm -r ' + rm_pre)
        if not self.every_n_iters(runner, self.interval):
            return
        results = runner.outputs['results']

        filename = self.filename_tmpl.format(runner.iter + 1)

        # img_list = [x for k, x in results.items() if k in self.res_name_list]
        weights = results['weight']

        if weights.shape[1] > 1:
            weights = torch.split(weights, 1, dim=1)
            weights = [np.uint8(weights[i].squeeze(1).detach().cpu().numpy().astype('float32') * 255) for i in range(len(weights))]
        else:
            weights = weights.squeeze(1).detach().cpu().numpy().astype('float32') * 255
            weights = np.uint8(weights)

        pred = results['seg_res'].squeeze(1).detach().cpu().numpy().astype('float32') * 255 / (results['num_classes'] - 1)
        pred = np.uint8(pred)

        gt = results['ori']['gt_semantic_seg'].squeeze(1).detach().cpu().numpy().astype('float32') * 255 / (results['num_classes'] - 1)
        gt = np.uint8(gt)

        gt_meta = results['meta']['gt_semantic_seg'].squeeze(1).detach().cpu().numpy().astype('float32') * 255 / (results['num_classes'] - 1)
        gt_meta = np.uint8(gt_meta)

        imgs = tensor2imgs(results['ori']['img'], **results['ori']['img_metas'][0]['img_norm_cfg'])
        imgs_gt = copy.deepcopy(imgs)
        imgs_pred = copy.deepcopy(imgs)        

        imgs_meta = tensor2imgs(results['meta']['img'], **results['meta']['img_metas'][0]['img_norm_cfg'])
        imgs_weight = [0 for i in range(len(imgs))]
        if 'box_semantic_seg' in results['ori']:
            imgs_box = copy.deepcopy(imgs)
            gt_box = results['ori']['box_semantic_seg'].squeeze(1).detach().cpu().numpy().astype('float32') * 255 / (results['num_classes'] - 1)
            gt_box = np.uint8(gt_box)

        # if 'aux_semantic_seg' in results['ori']:
        #     heatmaps = self.norm(results['ori']['aux_semantic_seg'], imgs)

        for i in range(len(imgs)):
            if isinstance(weights, list):
                gray = [cv2.applyColorMap(weights[j][i,:,:], cv2.COLORMAP_JET) for j in range(len(weights))]
            else:
                gray = cv2.applyColorMap(weights[i,:,:], cv2.COLORMAP_JET)
            gt_gray = cv2.applyColorMap(gt[i,:,:], cv2.COLORMAP_JET)                
            pred_gray = cv2.applyColorMap(pred[i,:,:], cv2.COLORMAP_JET)
            if 'box_semantic_seg' in results['ori']:
                gt_box_gray = cv2.applyColorMap(gt_box[i,:,:], cv2.COLORMAP_JET)
                imgs_box[i] = 0.5 * imgs_box[i] + 0.5 * gt_box_gray
            if isinstance(weights, list):
                imgs_weight[i] = [0.5 * imgs[i] + 0.5 * gray[j] for j in range(len(gray))]
                # imgs[i] = 0.5 * imgs[i] + 0.5 * gray
            else:
                imgs_weight[i] = 0.5 * imgs[i] + 0.5 * gray
            imgs_gt[i] = 0.5 * imgs_gt[i] + 0.5 * gt_gray
            imgs_pred[i] = 0.5 * imgs_pred[i] + 0.5 * pred_gray
            if i < gt_meta.shape[0]:
                gt_meta_gray = cv2.applyColorMap(gt_meta[i,:,:], cv2.COLORMAP_JET)
                imgs_meta[i] = 0.5 * imgs_meta[i] + 0.5 * gt_meta_gray
            
            if isinstance(weights, list):
                
                if 'box_semantic_seg' in results['ori'] and runner.iter % 2 == 0:
                    imgs_weight[i].extend([imgs_gt[i], imgs_pred[i], imgs_box[i]])
                else:
                    imgs_weight[i].extend([imgs_gt[i], imgs_pred[i], imgs_meta[i % gt_meta.shape[0]]])
                concat_img = self.concat_tile([imgs_weight[i][:8], 
                                                    imgs_weight[i][8:16], 
                                                    imgs_weight[i][16:]])
            else:
                if 'box_semantic_seg' in results['ori'] and runner.iter % 2 == 0:
                    concat_img = self.concat_tile([[imgs_weight[i], imgs_gt[i]], [imgs_pred[i], imgs_box[i]]])
                else:
                    concat_img = self.concat_tile([[imgs_weight[i], imgs_gt[i]], [imgs_pred[i], imgs_meta[i % gt_meta.shape[0]]]])

            # print(concat_img.shape)
            # print(imgs_weight[0][0].shape)
            # print(len(imgs_weight[i][:7].append(imgs_gt[i])))
            # import pdb
            # pdb.set_trace()
            basename = self.filename_tmpl.format(runner.iter + 1) + '_' + results['ori']['img_metas'][i]['filename'].split('/')[-1]
            filename = osp.join(runner.work_dir, self.output_dir, basename)
            mmcv.imwrite(concat_img, filename)
