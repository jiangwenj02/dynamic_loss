import os
import os.path
import pickle
import copy
import numpy as np
from .pipelines import Compose
from .base_dataset import BaseDataset
from .builder import DATASETS
from .utils import check_integrity, download_and_extract_archive
import mmcv
import torch

from mmcls.utils import get_root_logger

@DATASETS.register_module()
class CIFAR10Cor(BaseDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py  # noqa: E501
    """

    base_folder = 'cifar-10-batches-py'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self,
                 data_prefix,
                 pipeline,
                 loss_file=None,
                 weak_pipeline=None,
                 weak2=None,
                 noise_rate=0.0,
                 imb_ratio=1,
                 classes=None,
                 noise='sym',
                 ann_file=None,
                 test_mode=False):
        # super(CIFAR10Cor, self).__init__()
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.noise_rate = noise_rate
        self.imb_ratio = imb_ratio
        self.noise = noise
        self.logger = get_root_logger()
        self.data_infos = self.load_annotations()
        self.use_weak = None       
        self.weak2 = weak2        
        
        if weak_pipeline is not None:
            self.use_weak = True
            self.weak_pipeline = Compose(weak_pipeline)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        if self.use_weak:
            results_weak = copy.deepcopy(self.data_infos[idx])
            data = {'strong':self.pipeline(results), 'weak':self.weak_pipeline(results_weak)}
            if self.weak2 is not None:
                results_weak2 = copy.deepcopy(self.data_infos[idx])
                data['weak2'] = self.weak_pipeline(results_weak2)
        else:
            data = self.pipeline(results)
        return data
    
    def load_annotations(self):

        if not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self._load_meta()
        self.class_num = len(self.CLASSES)

        data_dirs = os.path.join('work_dirs', 'cifar' + str(self.class_num),
                                self.noise + '_cor' + str(self.noise_rate) + '_imb' +str(self.imb_ratio))
        imgs_path = os.path.join(data_dirs, 'imgs.npy')
        gt_labels_path = os.path.join(data_dirs, 'gt_labels.npy')
        true_labels_path = os.path.join(data_dirs, 'true_labels.npy')
        if os.path.exists(imgs_path) and os.path.exists(gt_labels_path) and os.path.exists(true_labels_path) and not self.test_mode:
            self.imgs = np.load(imgs_path)
            self.gt_labels = np.load(gt_labels_path)
            self.true_labels = np.load(true_labels_path)
            self.true_labels2 = []
             # load the picked numpy arrays
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.data_prefix, self.base_folder,
                                        file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    if 'labels' in entry:
                        self.true_labels2.extend(entry['labels'])
                    else:
                        self.true_labels2.extend(entry['fine_labels'])
            self.true_labels2 = np.array(self.true_labels2)
            self.logger.info('Load Data Succeed')
        else:
            self.imgs = []
            self.gt_labels = []

            # load the picked numpy arrays
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.data_prefix, self.base_folder,
                                        file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.imgs.append(entry['data'])
                    if 'labels' in entry:
                        self.gt_labels.extend(entry['labels'])
                    else:
                        self.gt_labels.extend(entry['fine_labels'])

            self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
            self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

            self.true_labels = copy.deepcopy(self.gt_labels)

            # if self.imb_ratio < 1:
            self.gen_imbalance()
            
            if self.noise_rate > 0:
                if self.noise == 'sym':
                    if self.imb_ratio < 1:
                        self.uniform_mix_C_imb()
                    else:
                        self.uniform_mix_C()
                elif self.noise == 'asym':
                    self.flip_mix_C()
                
            if not self.test_mode:
                mmcv.mkdir_or_exist(data_dirs)
                np.save(imgs_path, self.imgs)
                np.save(gt_labels_path, self.gt_labels)
                np.save(true_labels_path, self.true_labels)
                self.logger.info('Generate Data Succeed')

        count_all, noise_rate = self.verify_datas(self.gt_labels, self.true_labels)
        self.logger.info(f'The amount of each classes: {count_all}')
        self.logger.info(f'The noise rates of each classes: {noise_rate}')
        self.min_number = min(count_all)

        data_infos = []
        self.per_cls = dict()
        self.full_mask = []
        for i in range(self.class_num):
            self.per_cls[i] = []
        self.clear_rank()
        for idx, (img, gt_label, true_label, loss_rank) in enumerate(zip(self.imgs, self.gt_labels, self.true_labels, self.loss_ranks)):
            self.per_cls[gt_label].append(idx)
            self.full_mask.append(idx)
            gt_label = np.array(gt_label, dtype=np.int64)
            true_label = np.array(true_label, dtype=np.int64)            
            info = {'img': img, 'gt_label': gt_label, 'true_label': true_label, 'idx': idx, 'rank': loss_rank}
            data_infos.append(info)

        self.full_mask = np.array(self.full_mask)
        for i in range(self.class_num):
            self.per_cls[i] = np.array(self.per_cls[i])
        return data_infos
    
    def randomSelect(self, random_num):
        select_mask = dict()
        for i in range(self.class_num):
            np.random.shuffle(self.per_cls[i])
            percls_select_idx = self.per_cls[i][:random_num]
            select_mask[i] = percls_select_idx
        return select_mask
    
    def allSelect(self):
        select_mask = dict()
        for i in range(self.class_num):
            np.random.shuffle(self.per_cls[i])
            percls_select_idx = self.per_cls[i]
            select_mask[i] = percls_select_idx
        return select_mask
    
    def preSelect(self, selNum):
        selectPerclsIdx = dict()
        for key in range(self.class_num):
            selectPerclsIdx[key] = []
            np.random.shuffle(self.per_cls[key])
            for img_idx in self.per_cls[key]:
                if self.data_infos[img_idx]['gt_label'] == self.data_infos[img_idx]['true_label']:
                    selectPerclsIdx[key].append(img_idx)
                if len(selectPerclsIdx[key]) >= selNum:
                    break                
        return selectPerclsIdx
    
    def clear_rank(self):
        self.loss_ranks = np.zeros(len(self.gt_labels)) - 1

    def clean_per_cls(self):
        self.per_cls = dict()
        for i in range(self.class_num):
            self.per_cls[i] = []    

    def set_rank(self, idxes, ranks):
        for idx, img_idx in enumerate(idxes):
            self.data_infos[img_idx]['rank'] = 1.0 * ranks[idx]
    
    def uniform_mix_C(self):
        C = self.noise_rate * np.full((self.class_num, self.class_num), 1 / self.class_num) + \
            (1 - self.noise_rate) * np.eye(self.class_num)
        for i in range(len(self.gt_labels)):
            self.gt_labels[i] = np.random.choice(self.class_num, p=C[self.gt_labels[i]])

    def uniform_mix_C_imb(self):
        C = np.full((self.class_num, self.class_num), 0) + (1 - self.noise_rate) * np.eye(self.class_num)        
        total_training_sample = sum(self.img_num_per_cls)
        for i in range(self.class_num):
            for j in range(self.class_num):
                C[i,j] += self.img_num_per_cls[j] / (total_training_sample) * self.noise_rate
        print(C)
        for i in range(len(self.gt_labels)):
            self.gt_labels[i] = np.random.choice(self.class_num, p=C[self.gt_labels[i]])
    
    def flip_mix_C(self):
        C = np.eye(self.class_num)
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}
        for gt, noise in self.transition.items():
            C[gt, noise] += self.noise_rate
            C[gt, gt] -= self.noise_rate
        print(C)

        for i in range(len(self.gt_labels)):
            self.gt_labels[i] = np.random.choice(self.class_num, p=C[self.gt_labels[i]])

    def gen_imbalance(self):
        img_max = self.imgs.shape[0] / self.class_num

        img_num_per_cls = []
        for cls_idx in range(self.class_num):
            num = img_max * (self.imb_ratio**(cls_idx / (self.class_num - 1.0)))
            img_num_per_cls.append(int(num))
        
        idx_to_train_dict = {}        
        data_list_val = {}
        for j in range(self.class_num):
            data_list_val[j] = [i for i, label in enumerate(self.gt_labels) if label == j]

        for cls_idx, img_id_list in data_list_val.items():
            np.random.shuffle(img_id_list)
            img_num = img_num_per_cls[int(cls_idx)]            
            idx_to_train_dict[cls_idx] = img_id_list[:img_num]

        idx_to_train = []
        for cls_idx, img_id_list in idx_to_train_dict.items():
            idx_to_train.extend(img_id_list)

        self.imgs = self.imgs[idx_to_train]
        self.gt_labels = np.array(self.gt_labels)[idx_to_train]
        self.true_labels = np.array(self.true_labels)[idx_to_train]
        self.img_num_per_cls = img_num_per_cls
                
    def verify_rank(self, rank_num):
        ranks_true = {}
        ranks_all = {}
        rank_clean_samples = []
        rank_all_samples = []
        for j in range(rank_num):
            rank_clean_samples.append(0)
            rank_all_samples.append(0)
        for i in range(self.class_num):
            ranks_true[i] = {}
            ranks_all[i] = {}
            for j in range(rank_num):
                ranks_true[i][j] = 0
                ranks_all[i][j] = 0

        for item in self.data_infos:
            if item['rank'] != -1:
                key = item['gt_label'].item()
                ranks_all[key][item['rank']] += 1
                rank_all_samples[int(item['rank'])] += 1
                if key == item['true_label']:
                    ranks_true[key][item['rank']] += 1
                    rank_clean_samples[int(item['rank'])] += 1
        
        rank_clean_rates = [rank_clean_samples[i] / rank_all_samples[i] for i in range(rank_num)]
        display_idx = [i for i in range(0, rank_num, rank_num // 10)]
        display_idx.append(rank_num-1)
        display_cls = [i for i in range(0, self.class_num, self.class_num // 10)]
        display_cls.append(self.class_num - 1)
        for cls, ranks in ranks_all.items():
            if cls not in display_cls:
                continue
            out_string = 'cls {}: '.format(cls)
            for rank_idx, value in ranks.items():
                if rank_idx in display_idx:                    
                    if value > 0:
                        rate = ranks_true[cls][rank_idx] / (value) * 100
                        out_string = out_string + 'rank_{} {:.2f}, '.format(rank_idx, rate)
                    else:
                        out_string = out_string + 'rank_{} -1, '.format(rank_idx)
            self.logger.info(out_string)
        rank_clean_rates = [ '%.2f' % elem for elem in rank_clean_rates ]
        self.logger.info(f'Rank Clean Rates: {rank_clean_rates}')
        return rank_clean_rates

    def get_cls_rank_rates(self, rank_num):
        ranks_true = {}
        ranks_all = {}
        rank_clean_samples = []
        rank_all_samples = []
        for j in range(rank_num):
            rank_clean_samples.append(0)
            rank_all_samples.append(0)
        for i in range(self.class_num):
            ranks_true[i] = {}
            ranks_all[i] = {}
            for j in range(rank_num):
                ranks_true[i][j] = 0
                ranks_all[i][j] = 0

        for item in self.data_infos:
            if item['rank'] != -1:
                key = item['gt_label'].item()
                ranks_all[key][item['rank']] += 1
                rank_all_samples[int(item['rank'])] += 1
                if key == item['true_label']:
                    ranks_true[key][item['rank']] += 1
                    rank_clean_samples[int(item['rank'])] += 1
        
        rank_rates = np.zeros((self.class_num, rank_num))
        for cls, ranks in ranks_all.items():
            for rank_idx, value in ranks.items():
                rank_rates[cls, rank_idx] = ranks_true[cls][rank_idx] / (value) * 100
        return rank_rates

    def verify_datas(self, gt_label, true_label):
        count_true = [0] * self.class_num
        count_false = [0] * self.class_num

        for gt, true in zip(gt_label, true_label):
            if gt == true:
                count_true[gt] += 1
            else:
                count_false[gt] += 1
        
        count_all = [item_true + item_false for item_true, item_false in zip(count_true, count_false)]
        noise_rate = [round(100 * item_false / (item_all + 1e-6), 2) for item_false, item_all in zip(count_false, count_all)]
        return count_all, noise_rate

    def get_perClsNum(self):
        perClsNum = [0 for _ in range(self.class_num)]
        for _, item in enumerate(self.data_infos):
            perClsNum[item['gt_label']] += 1
        perClsNum = torch.tensor(perClsNum)
        return perClsNum
    
    def verify_all_labels(self, mask=None):
        count_true = [0] * self.class_num
        count_all = [0] * self.class_num
        count_true_all = 0
        count_all_all = 0
        for idx, item in enumerate(self.data_infos):
            if mask is not None:
                if idx not in mask:
                    continue
            if item['gt_label'] == item['true_label']:
                count_true[item['gt_label']] += 1
                count_true_all += 1
            count_all[item['gt_label']] += 1
            count_all_all += 1

        clean_rate = [round(100 * item_false / (item_all + 1e-6), 2) for item_false, item_all in zip(count_true, count_all)]
        clean_rate.append(round(100 * count_true_all / (count_all_all + 1e-6), 2))
        return clean_rate, count_all

    def check_acc(self, idxs):
        sub_gt_labels = np.array(self.gt_labels)[idxs]
        sub_true_labels = np.array(self.true_labels)[idxs]
        sub_count_all, sub_noise_rate= self.verify_datas(sub_gt_labels, sub_true_labels)
        self.logger.info(f'Count All: {sub_count_all}')
        self.logger.info(f'Noise Rates: {sub_noise_rate}')

    def _load_meta(self):
        path = os.path.join(self.data_prefix, self.base_folder,
                            self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError(
                'Dataset metadata file not found or corrupted.' +
                ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.CLASSES = data[self.meta['key']]

    def _check_integrity(self):
        root = self.data_prefix
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


@DATASETS.register_module()
class CIFAR100Cor(CIFAR10Cor):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """

    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
