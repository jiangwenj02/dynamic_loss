import torch.nn as nn
import torch
from ..builder import METAS, build_classifier, build_metanet, build_loss
from .base import BaseMeta
import torch.nn.functional as F
from mmcls.core import add_prefix
import time
import higher
from ..utils.augment import Augments
from mmcls.models.losses import accuracy

@METAS.register_module()
class DyanamicMeta(BaseMeta):
    def __init__(self,
                 classifier,
                 metanet=None,
                 mixup=False,
                 use_weak=True,
                 mixup_cfg=dict(type='BatchSoftMixup', alpha=0.5, prob=1.),
                 loss=dict(type='CrossEntropyLoss',
                            # loss_margin=1.0,
                            reduction='none',
                            use_soft=True),
                 frozen_stages=-1):
        super(DyanamicMeta, self).__init__()

        self.classifier = build_classifier(classifier)
        if 'num_classes' in classifier.head:
            self.num_classes = classifier.head.num_classes
        else:
            self.num_classes = classifier.backbone.num_classes

        self.frozen_stages = frozen_stages
        self.mixup = mixup
        self.use_weak = use_weak

        self.mixup = mixup
        if self.mixup:
            mixup_cfg['num_classes'] = self.num_classes
            self.augments = Augments(mixup_cfg)

        if metanet is not None:
            self.with_meta = True
            metanet['num_classes'] = self.num_classes
            self.metanet = build_metanet(metanet)
            self.compute_loss = build_loss(loss)

    def init_margins(self):
        self.classifier.init_margins()
        if self.with_meta:
            if isinstance(self.metanet, nn.Sequential):
                for m in self.metanet:
                    m.init_margins()
            else:
                self.metanet.init_margins()

    def froze(self, fmodel):
        if self.frozen_stages >= 0:
            fmodel.backbone.bn1.eval()
            for m in [fmodel.backbone.conv1,
                    fmodel.backbone.bn1]:
                for param in m.parameters():                    
                    param.requires_grad = False

            for i in range(1, self.frozen_stages + 1):
                    m = getattr(fmodel.backbone, f'layer{i}')
                    # m.eval()
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
    
    def meta_train(self, data, meta_data, optimizer, optimizer_meta, skip, **kwargs):
        if 'strong' in data:
            if self.use_weak:
                weak_data = data['weak']
            else:
                weak_data = data['strong']
            data = data['strong']
            meta_data = meta_data['strong']
        else:
            weak_data = data

        if skip:
            ### optimize dynamic loss
            with higher.innerloop_ctx(self.classifier, optimizer) as (fmodel, diffopt):
                self.froze(fmodel)
                cls_score = fmodel.meta_forward_train(**weak_data)
                correct_label, margin = self.metanet.forward_train(cls_score, weak_data)

                first_loss = self.compute_loss(cls_score=cls_score + margin, label=correct_label).mean()
                diffopt.step(first_loss)
                meta_first_log_vars={'loss':first_loss.item()}

                meta_loss_sw, _ = fmodel(**meta_data, meta=True)
                meta_loss, meta_second_log_vars = self._parse_losses(meta_loss_sw)
                    
                t1 = time.time()
                optimizer_meta.zero_grad()
                meta_loss.backward()
                optimizer_meta.step()
                t2 = time.time()
        else:
            cls_score = self.classifier.meta_forward_train(**weak_data)

        losses = {}        
        ### optimize classifier    
        with torch.no_grad():
            # compute soft label and margin
            correct_label, margin = self.metanet.forward_train(cls_score, weak_data)

            if self.metanet.use_LC and 'true_label' in data.keys():
                ### clean rate between soft label and given label of clean samples
                clean_idx = data['gt_label'] == data['true_label']
                correct_label_clean_acc = accuracy(correct_label[clean_idx], data['true_label'][clean_idx]).item()
                ### clean rate between soft label and given label of noise samples
                noise_idx = data['gt_label'] != data['true_label']
                correct_label_noise_acc = accuracy(correct_label[noise_idx], data['true_label'][noise_idx]).item()
                ### clean rate between soft label and given label of all samples
                correct_label_all_acc = accuracy(correct_label, data['true_label']).item()        

        
        if self.mixup:
            img, correct_label_mix = self.augments(data['img'], correct_label)
        else:
            img = data['img']
            correct_label_mix = correct_label
            
        cls_score = self.classifier.meta_forward_train(img, correct_label_mix)
        second_loss = self.compute_loss(cls_score=cls_score+margin, label=correct_label_mix)

        losses['loss'] = second_loss.mean()
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        outputs['log_vars'].update(add_prefix(meta_first_log_vars, 'first'))
        outputs['log_vars'].update(add_prefix(meta_second_log_vars, 'second'))
        outputs['log_vars']['meta_time'] = (t2 - t1)*1000

        if self.metanet.use_LC and 'true_label' in data.keys():
            outputs['log_vars']['correct_label.clean.acc'] = correct_label_clean_acc
            outputs['log_vars']['correct_label.noise.acc'] = correct_label_noise_acc
            outputs['log_vars']['correct_label.all.acc'] = correct_label_all_acc
            
        return outputs

    def warmup(self, data, optimizer_meta, iter_num=0, **kwargs):
        if 'strong' in data:
            if self.use_weak:
                weak_data = data['weak']
            else:
                weak_data = data['strong']
            data = data['strong']
        else:
            weak_data = None        
        ### optimize classifier
        label = F.one_hot(data['gt_label'], num_classes=self.num_classes).float()
        if self.mixup:
            img, correct_label_mix = self.augments(data['img'], label)
        else:
            img = data['img']
            correct_label_mix = label

        cls_score = self.classifier.meta_forward_train(img, correct_label_mix)
        classifer_loss = self.compute_loss(cls_score=cls_score, label=correct_label_mix).mean()
        classifer_log_vars = {'classifer_loss':classifer_loss.item()}


        ### inintialize label corrector
        
        if self.metanet.use_LC and iter_num < 2000:
            if weak_data is not None: 
                with torch.no_grad():
                    cls_score = self.classifier.meta_forward_train(**weak_data)
            correct_label, _ = self.metanet.forward_train(cls_score, weak_data, warm=True)
            meta_loss = (-label * torch.log(correct_label+1e-8)).sum(dim=-1).mean()
            meta_log_vars={'meta_loss':meta_loss.item()}

            optimizer_meta.zero_grad()
            meta_loss.backward(retain_graph=True)
            optimizer_meta.step()
        else:
            meta_loss = 0
            meta_log_vars={'meta_loss':0}
            

        with torch.no_grad():
            if 'true_label' in data.keys():
                ### clean rate between soft label and given label of clean samples
                clean_idx = data['gt_label'] == data['true_label']
                correct_label_clean_acc = accuracy(correct_label[clean_idx], data['true_label'][clean_idx]).item()
                ### clean rate between soft label and given label of noise samples
                noise_idx = data['gt_label'] != data['true_label']
                correct_label_noise_acc = accuracy(correct_label[noise_idx], data['true_label'][noise_idx]).item()
                ### clean rate between soft label and given label of all samples
                correct_label_all_acc = accuracy(correct_label, data['true_label']).item()
        
        outputs = dict(
            loss=classifer_loss, log_vars=classifer_log_vars, num_samples=len(data['img'].data))
        if self.metanet.use_LC:
            outputs['log_vars'].update(meta_log_vars)
            if 'true_label' in data.keys():
                outputs['log_vars']['correct_label.clean.acc'] = correct_label_clean_acc
                outputs['log_vars']['correct_label.noise.acc'] = correct_label_noise_acc
                outputs['log_vars']['correct_label.all.acc'] = correct_label_all_acc
        return outputs
    
    def finetune(self, data, **kwargs):
        if 'strong' in data:
            if self.use_weak:
                weak_data = data['weak']
            else:
                weak_data = data['strong']
            data = data['strong']
        else:
            weak_data = data

        label = F.one_hot(data['gt_label'], num_classes=self.num_classes)
        if self.mixup:
            img, correct_label = self.augments(data['img'], label)
        else:
            img, correct_label = data['img'], label

        cls_score = self.classifier.meta_forward_train(img, correct_label)
        
        with torch.no_grad():
            _, margin = self.metanet.forward_train(cls_score, weak_data)
        classifer_loss = self.compute_loss(cls_score=cls_score + margin, label=correct_label).mean()
        classifer_log_vars={'classifier_loss':classifer_loss.item()}
        outputs = dict(loss=classifer_loss, log_vars=classifer_log_vars, num_samples=len(data['img'].data))

        return outputs

    def get_loss(self, data):
        if 'weak' in data:
            data = data['weak']
        with torch.no_grad():
            cls_score = self.classifier.meta_forward_train(**data)
            label = F.one_hot(data['gt_label'], num_classes=self.num_classes)
            loss = self.compute_loss(cls_score=cls_score, label=label)
        log_vars = dict()
        outputs = dict(
            loss=loss, log_vars=log_vars, cls_score=cls_score, num_samples=len(data['img'].data))
        outputs['log_vars']['stage'] = 'get_loss'
        return outputs

    def train_step(self, stage, **kwargs):
        if stage == 'get_loss':
            return self.get_loss(**kwargs)
        elif stage == 'warmup':
            return self.warmup(**kwargs)
        elif stage == 'meta_train':
            return self.meta_train(**kwargs)
        elif stage == 'finetune':
            return self.finetune(**kwargs)

    def forward_train(self, imgs, **kwargs):
        return super().forward_train(imgs, **kwargs)

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        return self.classifier.simple_test(img, img_metas)
