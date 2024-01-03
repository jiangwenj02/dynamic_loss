# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number
# from hamcrest import has_key
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import mmcv
import math
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import DistSamplerSeedHook, build_runner
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcls.utils import get_root_logger
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier, build_meta
import os.path as osp
from mmcls.core import DistOptimizerHook, build_optimizers
# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (7.5, 5)

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def plot_rank(rank, cfg, name):
    x = np.arange(0.0, rank.shape[0], 1.0)

    # plt.scatter(x, rank, s, c="g", alpha=0.5, marker=r'$\clubsuit$',
    #             label="Luck")
    fig, ax = plt.subplots()
    # ax.plot(x, rank, 'o-', color='skyblue', label='weight', linewidth=3, markersize=6)
    ax.scatter(x, rank, c='tab:orange', s=40, marker='*', edgecolors='none')
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([0, rank.shape[0]])
    # import pdb
    # pdb.set_trace()
    if 'rank_num' in cfg:
        rank_num = cfg.rank_num
        for i in range(rank_num, rank.shape[0], rank_num):
            ax.axvline(x=i, color='black', linestyle='solid')
    if 'noise_rate' in cfg:
        noise_rate = cfg.noise_rate
        noise_step = rank_num - rank_num * noise_rate
        for i in range(0, rank.shape[0], rank_num):
            ax.axvline(x=i + noise_step, color='r', linestyle='dotted')

    plt.ylabel(r'$g(r)$', fontsize=20)
    ax.set_xlabel(r'Rank $r$', fontsize=20)
    # plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(osp.join(cfg.work_dir, name + '_rank.jpg'))

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def plot_rank_class(rank, cfg, name):
    num_classes = rank.shape[0] //  cfg.rank_num
    colormap = matplotlib.cm.gist_ncar
    colors = [np.array(colormap(i)).reshape(1,-1) for i in np.linspace(0, 1, num_classes * 2+3)]
    x = np.arange(0.0, cfg.rank_num, 1.0)
    if num_classes == 14:
        cols = 7
    elif num_classes == 10:
        # cfg['noise_rate']=0.08
        cols = 5
    elif num_classes == 50 or num_classes >= 100:
        cols = 10
    rows = int(math.ceil(num_classes / cols))
    print(rows, cols)
    figsize = (3.2*cols, 2.5*rows)
    axes = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
    axes = trim_axs(axes, num_classes)
    # ax.plot(x, rank, 'o-', color='skyblue', label='weight', linewidth=3, markersize=6)
    iter_num = range(num_classes)
    for index, i in enumerate(iter_num):
        axes[i].scatter(x, rank[i*cfg.rank_num:(i+1)*cfg.rank_num], c=colors[i], s=40, marker='*', edgecolors='none')
        axes[i].set_ylim([-0.1, 1.1])
        axes[i].set_xlim([0, cfg.rank_num])

        axes[i].set_title('class_' + str(i), fontsize=15)

        if 'noise_rate' in cfg:
            noise_rate = cfg.noise_rate
            # print(noise_rate)
            axes[i].axvline(x = cfg.rank_num * (1-noise_rate), color='r', linestyle='dotted')

        # axes[i].legend(fontsize=15, loc='center left')
        axes[i].set_ylabel(r'$g(r, {})$'.format(i))
        axes[i].set_xlabel(r'Rank $r$')
    # plt.legend(loc='center left')
    plt.savefig(osp.join(cfg.work_dir, name + '_rank_class.jpg'), bbox_inches='tight')

def plot_weight(weight, cfg, name):
    x = np.arange(0.0, weight.shape[0], 1.0)
    fig, ax = plt.subplots()
    ax.plot(x, weight, 'o-', color='skyblue', label='weight', linewidth=3, markersize=6)
    plt.xlabel(r'class')
    plt.ylabel(r'Margin $q$')
    # plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(osp.join(cfg.work_dir, name + '_weight.jpg'))

def main():
    args = parse_args()
    
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])


    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the model and load checkpoint
    model = build_meta(
        cfg.meta,
        classifier=cfg.get('model'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    name = osp.splitext(osp.basename(args.checkpoint))[0]

    model.eval()
    if model.metanet.use_LC:
        rank_gate = model.metanet.get_rank_weight()        
        if not model.metanet.LC_classwise:
            np.save(osp.join(cfg.work_dir, name + '_rank.npy'), rank_gate)
            plot_rank(rank_gate, cfg, name)
        else:
            num_classes = rank_gate.shape[0] //  cfg.rank_num
            rank_class = rank_gate.reshape((num_classes, cfg.rank_num))
            np.save(osp.join(cfg.work_dir, name + '_rank_class.npy'), rank_class)
            plot_rank_class(rank_gate, cfg, name)

    if model.metanet.use_MG: 
        margin = model.metanet.get_margin()
        np.save(osp.join(cfg.work_dir, name + '_weight.npy'), margin)
        plot_weight(margin, cfg, name)


if __name__ == '__main__':
    main()
