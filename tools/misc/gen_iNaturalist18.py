"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Usage:
1. Change "root" to your data path
2. python gen_lists.py
"""

import os
import json
from tqdm import tqdm

root = '/data3/zzhang/ours/iNaturalist'
save_root = './work_dirs/iNaturalist'
os.makedirs(save_root, exist_ok=True)
json2txt = {
    'train2018.json': 'iNaturalist18_train.txt',
    'val2018.json': 'iNaturalist18_val.txt',
    'test2018.json': 'iNaturalist18_test.txt',
}

def convert(json_file, txt_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    lines = []
    for i in tqdm(range(len(data['images']))):
        assert data['images'][i]['id'] == data['annotations'][i]['id']
        img_name = data['images'][i]['file_name']
        label = data['annotations'][i]['category_id']
        lines.append(img_name + ' ' + str(label) + '\n')
    
    with open(txt_file, 'w') as ftxt:
        ftxt.writelines(lines)

for k, v in json2txt.items():
    print('===> Converting {} to {}'.format(k, v))
    srcfile = os.path.join(root, k)
    detfile = os.path.join(save_root, v)
    convert(srcfile, detfile)
