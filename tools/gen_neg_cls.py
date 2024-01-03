import os
import os.path as osp
import imageio
import pandas as pd
import datetime
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import multiprocessing
import cv2

import torch
from pathlib import Path
import mmcv
import time
from mmcls.apis import init_model, inference_model
import glob

rois = {
    'big': [441, 1, 1278, 720],  # july video
    'small': [156, 40, 698, 527],
    '20191009_0900_0915': [156, 33, 699, 439],
    '20191009_0900_0915': [156, 33, 699, 439]
}
SKIP_FRAME = 1
CONTINUE_FRAME = 1
LABEL = ['erosive','ulcer']
COLOR = [(255,0,0),(0,255,0)]

class Evaluator:
    def __init__(self, opt):

        self.opt = opt
        self.saving_root = opt.save_path
        self.images_root = opt.images_path
        self.det_summary = osp.join(self.saving_root, 'summary.txt')
        self.save_train_images = opt.save_train_images
        os.popen('rm -r ' + osp.join(self.saving_root, '*'))
        os.makedirs(self.saving_root, exist_ok=True)

    def _init_detector(self):
        self.device = self.opt.device

        # Load model
        model = init_model(self.opt.config, checkpoint=self.opt.weights, device='cpu')
        self.names = model.CLASSES
        self.model = model.to(self.device)

    def test_images(self):
        self._init_detector()
        class_dirs = glob.glob(osp.join(self.images_root,'*'))
        for index,  class_dir in enumerate(class_dirs):
            class_dir = osp.basename(class_dir)
            image_path = os.path.join(self.images_root, class_dir)

            image_files = glob.glob(osp.join(image_path, '*.jpg'))
            length = len(image_files)
            pbar = tqdm(range(length))
            count = 0 
            start_time = time.time()
            p = Path(image_path)  # to Path
            print(p.stem)
            save_path = osp.join(self.saving_root, p.stem)  # img.jpg
            print(save_path)
            vid_save_path = osp.join(save_path, p.stem)
            fp_save_dirs = []
            for name in self.names:
                fp_save_dirs.append(osp.join(save_path, name))
                os.makedirs(osp.join(save_path, name), exist_ok=True)
            os.makedirs(save_path, exist_ok=True)
            for frame in pbar:
                torch.cuda.empty_cache()
                if not os.path.isfile(image_files[frame]):
                    print('{} not exist'.format(image_files[frame]))
                    continue
                img_ori = mmcv.imread(image_files[frame])
                # Inference
                result = inference_model(self.model, img_ori)
                img = self.model.show_result(img_ori, result, show=False)
                if result['pred_class'] != p.stem:
                    cv2.imwrite(os.path.join(self.saving_root, p.stem, osp.basename(image_files[frame])), img_ori)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="video_evaluation")
    parser.add_argument('--csv_file', type=str, default='neg0615.csv', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--images_path', type=str, default='/data2/dataset/gastric_3cls_0625/test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='/data3/zzhang/tmp/gastric_3cls_0921/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_train_images', default=None, help='source')  # /data3/zzhang/tmp/classification/train/non_cancer/
    # parser.add_argument('--det_summary', type=str, default='/data3/zzhang/tmp/erosive_ulcer_videos0615/summary.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--start', default=0, type=int,  help="video index to start")
    parser.add_argument('--end', default=0, type=int, help="video index to end")
    parser.add_argument('--config', type=str, default='configs/diseased/resnet50_diseased3.py', help='model.pt path(s)')
    parser.add_argument('--weights', type=str, default='work_dirs/resnet50_diseased3/latest.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    args = parser.parse_args()
    evaluator = Evaluator(args)

    evaluator.test_images()


