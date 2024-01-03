import os
import numpy as np

def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def gen_train_list():
    root_data_path = 'data/Food101N/meta/imagelist.tsv'
    class_list_path = 'data/Food101N/meta/classes.txt'

    # file_path_prefix = 'data/Food101N/images'

    map_name2cat = dict()
    with open(class_list_path) as fp:
        for i, line in enumerate(fp):
            row = line.strip()
            map_name2cat[row] = i
    num_class = len(map_name2cat)
    print('Num Classes: ', num_class)

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        fp.readline()  # skip first line

        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            targets.append(map_name2cat[class_name])
            img_list.append(line.strip())

    targets = np.array(targets)
    img_list = np.array(img_list)
    print('Num Train Images: ', len(img_list))

    save_dir = check_folder('./work_dirs/food101n')
    np.save(os.path.join(save_dir, 'train_images'), img_list)
    np.save(os.path.join(save_dir, 'train_targets'), targets)

    return map_name2cat


def gen_test_list(arg_map_name2cat):
    map_name2cat = arg_map_name2cat
    root_data_path = 'data/Food101/meta/test.txt'

    # file_path_prefix = 'work_dirs/food-101/images'

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            targets.append(map_name2cat[class_name])
            img_list.append(os.path.join(line.strip() + '.jpg'))

    targets = np.array(targets)
    img_list = np.array(img_list)

    save_dir = check_folder('./work_dirs/food101n')
    np.save(os.path.join(save_dir, 'test_images'), img_list)
    np.save(os.path.join(save_dir, 'test_targets'), targets)

    print('Num Test Images: ', len(img_list))


if __name__ == '__main__':
    map_name2cat = gen_train_list()
    gen_test_list(map_name2cat)