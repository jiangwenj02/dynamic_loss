from pickle import NONE
import time
import warnings
from functools import partial
import torch
import mmcv
import torch.distributed as dist
from mmcv.parallel import collate, is_module_wrapper
from mmcv.runner import HOOKS, RUNNERS, EpochBasedRunner, get_host_info
from torch.utils.data import DataLoader, Sampler
import numpy as np
from mmcls.datasets.builder import build_sub_dataloader
from torch.optim import Optimizer

class IterLoader:
    """Iteration based dataloader.

    This wrapper for dataloader is to matching the iter-based training
    proceduer.

    Args:
        dataloader (object): Dataloader in PyTorch.
        runner (object): ``mmcv.Runner``
    """

    def __init__(self, dataloader, runner):
        self._dataloader = dataloader
        self.runner = runner
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        """The number of current epoch.

        Returns:
            int: Epoch number.
        """
        return self._epoch

    def update_dataloader(self, curr_scale):
        """Update dataloader.

        Update the dataloader according to the `curr_scale`. This functionality
        is very helpful in training progressive growing GANs in which the
        dataloader should be updated according to the scale of the models in
        training.

        Args:
            curr_scale (int): The scale in current stage.
        """
        # update dataset, sampler, and samples per gpu in dataloader
        if hasattr(self._dataloader.dataset, 'update_annotations'):
            update_flag = self._dataloader.dataset.update_annotations(
                curr_scale)
        else:
            update_flag = False
        if update_flag:
            # the sampler should be updated with the modified dataset
            assert hasattr(self._dataloader.sampler, 'update_sampler')
            samples_per_gpu = None if not hasattr(
                self._dataloader.dataset, 'samples_per_gpu'
            ) else self._dataloader.dataset.samples_per_gpu
            self._dataloader.sampler.update_sampler(self._dataloader.dataset,
                                                    samples_per_gpu)
            # update samples per gpu
            if samples_per_gpu is not None:
                if dist.is_initialized():
                    # samples = samples_per_gpu
                    # self._dataloader.collate_fn = partial(
                    #     collate, samples_per_gpu=samples)
                    self._dataloader = DataLoader(
                        self._dataloader.dataset,
                        batch_size=samples_per_gpu,
                        sampler=self._dataloader.sampler,
                        num_workers=self._dataloader.num_workers,
                        collate_fn=partial(
                            collate, samples_per_gpu=samples_per_gpu),
                        shuffle=False,
                        worker_init_fn=self._dataloader.worker_init_fn)

                    self.iter_loader = iter(self._dataloader)
                else:
                    raise NotImplementedError(
                        'Currently, we only support dynamic batch size in'
                        ' ddp, because the number of gpus in DataParallel '
                        'cannot be obtained easily.')

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class MaskSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

@RUNNERS.register_module()
class DynamicEpochBasedRunner(EpochBasedRunner):
    """Dynamic EpochBased Runner.

    In this Dynamic Iterbased Runner, we will pass the ``reducer`` to the
    ``train_step`` so that the models can be trained with dynamic architecture.
    More details and clarification can be found in this [tutorial](docs/tutorials/ddp_train_gans.md).  # noqa

    Args:
        is_dynamic_ddp (bool, optional): Whether to adopt the dynamic ddp.
            Defaults to False.
        pass_training_status (bool, optional): Whether to pass the training
            status. Defaults to False.
    """
    def __init__(self, cfg=None,
                 first_random_select_ratio=0.5,
                 second_random_select_ratio=0.5,
                 no_noise=False,
                rank_num=0,
                step=1,
                max_iterations=None,
                warm_epoch=None,
                **kwargs):
        super(DynamicEpochBasedRunner, self).__init__(**kwargs)

        self.warm_epoch = warm_epoch
        self.first_random_select_ratio = first_random_select_ratio
        self.second_random_select_ratio = second_random_select_ratio
        self.cfg = cfg
        self.rank_num = rank_num
        self.max_iterations = max_iterations
        self.no_noise = no_noise
        self.step = step
        

    def run_iter_warmup(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(stage='warmup',
                                            data=data_batch,
                                            optimizer_meta=self.optimizer['metanet'],
                                            iter_num=self._iter,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
    
    def run_iter_finetune(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(stage='finetune',
                                            data=data_batch,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def run_iter_meta(self, data_batch, meta_data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(stage='meta_train',
                                                data=data_batch,
                                                meta_data=meta_data_batch,
                                                optimizer=self.optimizer['classifier'],
                                                optimizer_meta=self.optimizer['metanet'],
                                                skip = not(self._epoch % self.step),
                                                **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    ### compute and set rank for each sample
    def compute_rank(self, return_sel_loss=False):
        self.model.eval()
        self.min_cls_num = self.data_loader.dataset.min_number
        selectPerclsMask = self.data_loader.dataset.randomSelect(int(self.first_random_select_ratio * self.min_cls_num))

        perclsLosses = dict()
        perclsIdx = dict()
        selectPerclsLosses = dict()
        selectPerclsIdx = dict()

        for key in range(self.data_loader.dataset.class_num):
            perclsLosses[key] = []
            perclsIdx[key] = []
            selectPerclsLosses[key] = []
            selectPerclsIdx[key] = []
        
        prog_bar = mmcv.ProgressBar(len(self.data_loader.dataset))
        for _, data_batch in enumerate(self.data_loader):
            if self.no_noise:
                if 'strong' in data_batch:
                    data_batch = data_batch['strong']
                outputs['loss'] = torch.rand(data_batch['gt_label'].shape).to(data_batch['gt_label'].device)
            else:
                outputs = self.model.train_step(stage='get_loss', data=data_batch)
                if 'strong' in data_batch:
                    data_batch = data_batch['strong']

            for j in range(data_batch['idx'].shape[0]):
                perclsLosses[data_batch['gt_label'][j].item()].append(outputs['loss'][j].item())
                perclsIdx[data_batch['gt_label'][j].item()].append(data_batch['idx'][j].item())

                if data_batch['idx'][j].item() in selectPerclsMask[data_batch['gt_label'][j].item()]:
                    selectPerclsLosses[data_batch['gt_label'][j].item()].append(outputs['loss'][j].item())
                    selectPerclsIdx[data_batch['gt_label'][j].item()].append(data_batch['idx'][j].item())

            batch_size = data_batch['gt_label'].shape[0]
            for _ in range(batch_size):
                prog_bar.update()
        
        if self.rank_num > 0:
            self.data_loader.dataset.clear_rank()
            for key, value in perclsLosses.items():
                value = np.array(value)
                perclsIdx_sort = np.array(perclsIdx[key])
                idx = np.argsort(value)
                
                rank_idx = 1.0 *  np.arange(len(perclsIdx_sort)) / (len(perclsIdx_sort) + 1) * self.rank_num 
                rank_idx = rank_idx.astype(int)
                perclsIdx_sort = perclsIdx_sort[idx]
                self.data_loader.dataset.set_rank(perclsIdx_sort, rank_idx)            
            self.data_loader.dataset.verify_rank(self.rank_num)

        if return_sel_loss:
            return selectPerclsLosses, selectPerclsIdx
    
    ### Hierarchical Sampling
    def hier_sampling(self):        
        sub_mask = np.zeros(0)
        percls_number = int(self.first_random_select_ratio * self.min_cls_num * self.second_random_select_ratio)
        selectPerclsLosses, selectPerclsIdx = self.compute_rank(return_sel_loss=True)
        for key, value in selectPerclsLosses.items():
            value = np.array(value)
            selectPerclsIdx[key] = np.array(selectPerclsIdx[key])
            if self.no_noise:
                idx = np.random.choice(np.argsort(value), percls_number, replace=False)
            else:
                idx = np.argsort(value)[:percls_number]
            sub_mask = np.concatenate((sub_mask, selectPerclsIdx[key][idx]), axis=0)

        sub_mask = sub_mask.astype(np.int64)
        if hasattr(self.data_loader.dataset, 'check_acc'):
            self.data_loader.dataset.check_acc(sub_mask)
        not_sub_mask = np.setdiff1d(self.full_mask, sub_mask)
        return sub_mask, not_sub_mask
    
    def gen_subset(self):        
        self.min_cls_num = self.data_loader.dataset.min_number        

        metaMask, trainMask = self.hier_sampling()
        trainSampler = MaskSampler(trainMask)
        metaSampler = MaskSampler(metaMask)
        trainloader = build_sub_dataloader(self.data_loader.dataset,
                                                samples_per_gpu=self.cfg.data.samples_per_gpu,
                                                workers_per_gpu=self.cfg.data.workers_per_gpu,
                                                sampler=trainSampler,
                                                num_gpus=len(self.cfg.gpu_ids),
                                                dist=self.cfg.dist,
                                                round_up=True,
                                                seed=self.cfg.seed)
        metaloader = build_sub_dataloader(self.data_loader.dataset,
                                                samples_per_gpu=self.cfg.data.samples_per_gpu,
                                                workers_per_gpu=self.cfg.data.workers_per_gpu,
                                                sampler=metaSampler,
                                                num_gpus=len(self.cfg.gpu_ids),
                                                dist=self.cfg.dist,
                                                round_up=True,
                                                seed=self.cfg.seed)

        iter_metaloader = IterLoader(metaloader, self)            
        return trainloader, metaloader, iter_metaloader

    def train(self, data_loader, **kwargs):
        self.data_loader = data_loader        
        self.warm_stage = self.epoch < self.warm_epoch        

        self.full_mask = self.data_loader.dataset.full_mask
        self.perClsNum = self.data_loader.dataset.get_perClsNum()
        
        self.compute_rank(self.data_loader)
        if self.warm_stage:
            self.trainloader = self.data_loader
            self.meta_data_loader = None                
        else:
            self.trainloader, self.meta_data_loader, self.iter_metaloader = self.gen_subset()

        self.model.train()
        self.mode = 'train'
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.trainloader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            if self.warm_stage:
                # This stage is used for initializing the meta net.
                self.run_iter_warmup(data_batch, train_mode=True, **kwargs)
            else:   
                meta_data_batch = next(self.iter_metaloader)    
                self.run_iter_meta(data_batch, meta_data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        # After meta train, using the meta data loader finetuning the model
        if self.meta_data_loader is not None:
            for i, data_batch in enumerate(self.meta_data_loader):
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter_finetune(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        if self.max_iterations is not None:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if mode == 'train':
                    self._max_epochs = self.max_iterations // len(data_loaders[i])

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')
        # data_loaders[0][1] = IterLoader(data_loaders[0][1], self)

        self.data_loader = data_loaders[0]
        
        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        self.logger.info('Finished training of %s', self.work_dir)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if self.meta is None:
            self.meta = {}
        self.meta.setdefault('hook_msgs', {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        self.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

        # Re-calculate the number of iterations when resuming
        # models with different number of GPUs
        if 'config' in checkpoint['meta']:
            config = mmcv.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
                    previous_gpu_ids) != self.world_size:
                self._iter = int(self._iter * len(previous_gpu_ids) /
                                 self.world_size)
                self.logger.info('the iteration number is changed due to '
                                 'change of GPU number')

        # resume meta information meta
        self.meta = checkpoint['meta']

        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

        # if self.rank_num>0:
        #     self.gen_rank = True
        
        if self._epoch > 0:
            self.init = False

    def register_lr_hook(self, lr_config):
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            is_dict_of_dict = True
            for key, cfg in lr_config.items():
                if not isinstance(cfg, dict):
                    is_dict_of_dict = False

            def register_lr_hook_fun(lr_config_part):
                assert 'policy' in lr_config_part
                policy_type = lr_config_part.pop('policy')
                # if lr_config_part is not None:
                #     lr_config_part.setdefault('by_epoch', False)
                # If the type of policy is all in lower case, e.g., 'cyclic',
                # then its first letter will be capitalized, e.g., to be 'Cyclic'.
                # This is for the convenient usage of Lr updater.
                # Since this is not applicable for `
                # CosineAnnealingLrUpdater`,
                # the string will not be changed if it contains capital letters.
                if policy_type == policy_type.lower():
                    policy_type = policy_type.title()
                hook_type = policy_type + 'MetaLrUpdaterHook'
                lr_config_part['type'] = hook_type
                hook = mmcv.build_from_cfg(lr_config_part, HOOKS)
                self.register_hook(hook, priority='VERY_HIGH')

            if is_dict_of_dict:
                for key, lr_config_part in lr_config.items():
                    lr_config_part['name'] = key
                    register_lr_hook_fun(lr_config_part)
            else:
                register_lr_hook_fun(lr_config)
            return
        else:
            hook = lr_config
        self.register_hook(hook, priority='VERY_HIGH')

