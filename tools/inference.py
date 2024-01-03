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
class Evaluator:
    def __init__(self, opt):

        self.opt = opt
        self.saving_root = opt.save_path
        self.image_root = opt.image_path
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
        for index, image_file in tqdm(enumerate(glob.glob(self.image_root + '*.jpg'))):
            img = mmcv.imread(image_file)
            # Inference
            result = inference_model(self.model, img)
            img = self.model.show_result(img, result, show=False)
            
            if result['pred_label'] != 0:
                filename = image_file.replace(self.image_root, self.saving_root)
                mmcv.imwrite(img, filename)            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="video_evaluation")
    parser.add_argument('--image_path', type=str, default='/data3/zzhang/tmp/gastro_cancer/test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='/data3/zzhang/tmp/gastro_cancer_0901/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--config', nargs='+', type=str, default='/data3/zzhang/mmclassification/configs/diseased/resnet50_cancer.py', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='/data3/zzhang/mmclassification/work_dirs/resnet50_cancer/latest.pth', help='model.pt path(s)')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.test_images()


