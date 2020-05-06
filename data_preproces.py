import os
import random

import cv2
import scipy

from utility import *

def read_txt(txt_file, limit=1000):
    with open(txt_file) as f:
        testing = f.readlines()

    img_paths = []

    for i in testing[:limit]:
        i = i.replace("'", '')
        i = i.replace(',', '')
        i = i.strip()
        img_paths.append(i)

    # img = cv2.imread(img_paths[0])
    # print(img.shape)
    return img_paths


def get_density_map_gaussian(im, points, adaptive_mode=False, fixed_value=15, fixed_values=None):
    density_map = np.zeros(im.shape[:2], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_mode == True:
        fixed_values = None
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances, locations = tree.query(points, k=4)
    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        if num_gt > 1:
            if adaptive_mode == 1:
                sigma = int(np.sum(distances[idx][1:4]) * 0.1)
            elif adaptive_mode == 0:
                sigma = fixed_value
        else:
            sigma = fixed_value
        sigma = max(1, sigma)
        gaussian_radius_no_detection = sigma * 3
        gaussian_radius = gaussian_radius_no_detection

        if fixed_values is not None:
            grid_y, grid_x = int(p[0]//(h/3)), int(p[1]//(w/3))
            grid_idx = grid_y * 3 + grid_x
            gaussian_radius = fixed_values[grid_idx] if fixed_values[grid_idx] else gaussian_radius_no_detection
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        gaussian_map[gaussian_map < 0.0003] = 0
        if np.sum(gaussian_map):
            gaussian_map = gaussian_map / np.sum(gaussian_map)
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(density_map.shape[0], p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(density_map.shape[1], p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    # density_map[density_map < 0.0003] = 0
    density_map = density_map / (np.sum(density_map / num_gt))
    return density_map


def gen_x_y(img_paths, train_val_test='train', augmentation_methods=['ori']):
    x, y = [], []
    for i in img_paths:
        x_ = load_img(i)
        y_ = img_from_h5(i.replace('.jpg', '.h5').replace('images', 'ground'))
        x_, y_ = fix_singular_shape(x_), fix_singular_shape(y_)
        if 'ori' in augmentation_methods:
            x.append(np.expand_dims(x_, axis=0))
            y.append(np.expand_dims(np.expand_dims(y_, axis=0), axis=-1))
        if 'flip' in augmentation_methods and train_val_test == 'train':
            x.append(np.expand_dims(cv2.flip(x_, 1), axis=0))
            y.append(np.expand_dims(np.expand_dims(cv2.flip(y_, 1), axis=0), axis=-1))
    if train_val_test == 'train':
        random_num = random.randint(7, 77)
        random.seed(random_num)
        random.shuffle(x)
        random.seed(random_num)
        random.shuffle(y)
        random.seed(random_num)
        random.shuffle(img_paths)
    return x, y, img_paths


if __name__ == '__main__':
    paths = read_txt('data/testing.txt')
    x,y, img_paths = gen_x_y(paths[:5])
    x_train = np.squeeze(x, axis=1)
    y_train = np.squeeze(y, axis=1)
    print(x_train.shape)
    print(y_train.shape)