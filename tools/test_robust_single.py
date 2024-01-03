import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import copy
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.core import wrap_fp16_model
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.metrics:
            threshold_list = np.arange(1.00, -0.01, -0.01).tolist()
            threshold_list = tuple(threshold_list)
            args.metric_options['thrs'] = threshold_list   
            cls = outputs[0].shape[0]
            results_all = []
            for i in range(cls):
                print('-----------cls------------', i)
                outputs_s = copy.deepcopy(outputs)
                for j in range(len(outputs_s)):
                    outputs_s[j][:i] = -0.1
                    outputs_s[j][i+1:] = -0.1
                args.metric_options['average_mode'] = 'none'
                results = dataset.evaluate(outputs_s, args.metrics,
                                        args.metric_options)
                results_all.append(results)
            results = copy.deepcopy(results_all[0])
            for idx, item in enumerate(results_all):
                for k,v in item.items():
                    if type(v) is np.ndarray:
                        results[k][idx] = v[idx]
                    # if v is not float and v.shape[0] > 1:
                    #     results[k][idx] = v[idx]
            for k,v in results.items():
                if type(v) is np.ndarray:
                    print(k,v)
                    results[k] = np.mean(results[k])
            f1_best = 0
            f1_best_thr = 0
            recall_best = 0
            recall_best_thr = 0
            TPRs = []
            FPRs = []
            for k, v in results.items():
                if 'f1_score' in k and v > f1_best:
                    f1_best = v
                    f1_best_str = f'\n{k} : {v:.2f}'
                    f1_best_thr = float(k.split('_')[-1])
                if 'TPR' in k:
                    TPRs.append(v)
                if 'FPR' in k:
                    FPRs.append(v)
                if 'recall' in k and v > recall_best:
                    recall_best = v
                    recall_best_str = f'\n{k} : {v:.2f}'
                    recall_best_thr = float(k.split('_')[-1])
                if 'TPR' in k:
                    TPRs.append(v)
                if 'FPR' in k:
                    FPRs.append(v)
                # print(k, " : ", v)
                # print(f'\n{k} : {v:.2f}')
            if 'TPR' in args.metrics:      
                print(FPRs)          
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,10))
                plt.plot(FPRs, TPRs, color='darkorange',
                    lw=2, label='ROC curve')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.savefig(args.checkpoint.replace('.pth', '.jpg'))

            if 'recall' in args.metrics:
                print('-------------recall-----------------')
                print('best: ', recall_best_str)
                args.metric_options['thrs'] = recall_best_thr
                results = dataset.evaluate(outputs, args.metrics,
                                        args.metric_options)
                for k, v in results.items():
                    print(k, " : ", v)
                args.metric_options['average_mode'] = 'none'
                results = dataset.evaluate(outputs, args.metrics,
                                        args.metric_options)
                for k, v in results.items():
                    print(k, " : ", v)
                # print('best: ', f1_best_str)

            if 'recall' in args.metrics:
                print('-------------recall-single-----------------')
                print('best: ', recall_best_str)
                cls = outputs[0].shape[0]
                for i in range(cls):
                    print('-----------cls------------', i)
                    outputs_s = copy.deepcopy(outputs)
                    for j in range(len(outputs_s)):
                        outputs_s[j][:i] = 0
                        outputs_s[j][i+1:] = 0
                    args.metric_options['thrs'] = recall_best_thr
                    results = dataset.evaluate(outputs_s, args.metrics,
                                            args.metric_options)
                    for k, v in results.items():
                        print(k, " : ", v)
                    args.metric_options['average_mode'] = 'none'
                    results = dataset.evaluate(outputs_s, args.metrics,
                                            args.metric_options)
                    for k, v in results.items():
                        print(k, " : ", v)
            
            if 'f1_score' in args.metrics:
                print('-------------f1best-----------------')
                print('best: ', f1_best_str)
                args.metric_options['thrs'] = f1_best_thr
                results = dataset.evaluate(outputs, args.metrics,
                                        args.metric_options)
                for k, v in results.items():
                    print(k, " : ", v)
                args.metric_options['average_mode'] = 'none'
                results = dataset.evaluate(outputs, args.metrics,
                                        args.metric_options)
                for k, v in results.items():
                    print(k, " : ", v)
        else:
            warnings.warn('Evaluation metrics are not specified.')
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if not args.out:
                print('\nthe predicted result for the first element is '
                      f'pred_score = {pred_score[0]:.2f}, '
                      f'pred_label = {pred_label[0]} '
                      f'and pred_class = {pred_class[0]}. '
                      'Specify --out to save all results to files.')
    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
