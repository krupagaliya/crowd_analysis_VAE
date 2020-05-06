import os
from os.path import expanduser

import cv2
import numpy as np
import h5py


def get_logger(name):
    import logging
    path = os.path.join(expanduser('~'), "Desktop")
    # Create and configure logger
    logging.basicConfig(filename=os.path.join(path, 'CA_log_information.txt'),
                        format='%(asctime)s %(message)s',
                        filemode='w')

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    return logger


def make_txt(jpg_img_dir, filename, limit=1000):
    file = open(filename, 'w')
    for j, i in enumerate(os.listdir(jpg_img_dir)):
        if i.endswith('jpg'):
            file.write(os.path.join(jpg_img_dir, i) + '\n')
        if j > limit:
            break
    file.close()


def load_img(path, resizedata =True):
    print("path is ", path)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if resizedata:
        img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    # img[:, :, 0]=(img[:, :, 0] - 0.5) / 1
    # img[:, :, 1]=(img[:, :, 1] - 0.5) / 1
    # img[:, :, 2]=(img[:, :, 2] - 0.5) / 1
    return img.astype(np.float32)


def img_from_h5(path, resizedata=True):
    gt_file = h5py.File(path, 'r')
    density_map = np.asarray(gt_file['density'])
    if resizedata:
        density_map = cv2.resize(density_map, (int(density_map.shape[1] / 3), int(density_map.shape[0] / 3)), interpolation=cv2.INTER_CUBIC) * 9
    stride = 1
    if stride > 1:
        density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int) // stride).tolist(), dtype=np.float32)
        for r in range(density_map_stride.shape[0]):
            for c in range(density_map_stride.shape[1]):
                density_map_stride[r, c] = np.sum(density_map[r * stride:(r + 1) * stride, c * stride:(c + 1) * stride])
    else:
        density_map_stride = density_map
    return density_map_stride


def fix_singular_shape(tensor):
    # Append 0 lines or colums to fix the shapes as integers times of 8, since there are 3 pooling layers.
    for idx_sp in [0, 1]:
        remainder = tensor.shape[idx_sp] % 8
        if remainder != 0:
            fix_len = 8 - remainder
            pad_list = []
            for idx_pdlst in range(len(tensor.shape)):
                if idx_pdlst != idx_sp:
                    pad_list.append([0, 0])
                else:
                    pad_list.append([int(fix_len / 2), fix_len - int(fix_len / 2)])
            tensor = np.pad(tensor, pad_list, 'constant')
    return tensor


if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')
    make_txt('/home/krupa/VirtualDrive/8th-BE/Project/SANet-Keras/data/part_B/test_data/images',
             'data/testing.txt')