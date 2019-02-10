import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os
import scipy.ndimage as sc


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def parse_minute_file(mnt_file_path):
    with open(mnt_file_path, 'r') as mnt_file:
        mnt_list = [line.strip().split(' ') for line in mnt_file.readlines()]
        mnt_out = []
        for mnt in mnt_list:
            p1 = (int(mnt[1]), int(mnt[2]))
            p2 = (int(mnt[1]) + int(mnt[4]), int(mnt[2]) + int(mnt[4]))
            type = int(mnt[0])
            mnt_dict = {"type": type,
                        "p1": p1,
                        "p2": p2}
            mnt_out.append(mnt_dict)
        return mnt_out


def plot_minute_on_image(mnt_dict_list, img_path, plot_rect=False):
    size_dict = {1: 3, 2: 3, 4: 10, 5: 10}
    color_dict = {1: 'r', 2: 'b', 4: 'g', 5: 'g'}
    img = plt.imread(img_path)
    for mnt_dict in mnt_dict_list:
        p1 = mnt_dict['p1']
        p2 = mnt_dict['p2']
        type = mnt_dict['type']
        plt.scatter(p1[0], p1[1], s=size_dict[type], c=color_dict[type])
        if plot_rect:
            cv2.rectangle(img, p1, p2, color=3, thickness=3)
    plt.imshow(img)
    plt.show()


def create_map_scipy(mnt_dict_list, size=(768, 832), num_of_maps=3, do_show=False):
    maps = []
    types = [[1], [2], [4, 5]]
    for idx in range(num_of_maps):
        map = np.zeros(size)
        x = [mnt_dict['p1'][0] for mnt_dict in mnt_dict_list if mnt_dict['type'] in types[idx]]
        y = [mnt_dict['p1'][1] for mnt_dict in mnt_dict_list if mnt_dict['type'] in types[idx]]
        map[[y, x]] = 1
        maps.append(sc.gaussian_filter(map, sigma=np.sqrt(3))[:, :, np.newaxis] * 20)
    output = np.concatenate(maps, axis=-1)
    return output


def create_map(mnt_dict_list, img_path, do_show=False):
    img = plt.imread(img_path)
    size = max(img.shape)
    maps = [np.zeros((size, size), np.float32), np.zeros((size, size), np.float32), np.zeros((size, size), np.float32)]
    # start_time = time.time()
    for mnt_dict in mnt_dict_list:
        p1 = mnt_dict['p1']
        mnt_type = mnt_dict['type'] - 1 if mnt_dict['type'] in [1, 2] else 2
        maps[mnt_type] += make_gaussian(size, fwhm=9, center=p1)
    # print("time ro crate all gaussians: {}".format((time.time() - start_time)/len(mnt_dict_list)))
    mask = np.ones_like(maps[0][:img.shape[0], :img.shape[1]])
    total_map = np.zeros_like(maps[0])
    for map in maps:
        total_map += map
        if do_show:
            mask[map[:img.shape[0], :img.shape[1]] > 0.1] = 0
            plt.imshow(map)
            plt.figure()
    if do_show:
        plt.imshow(total_map[:img.shape[0], :img.shape[1]] * 150 + img * mask)
        plt.show()
    for i in range(len(maps)):
        maps[i] = maps[i][:img.shape[0], :img.shape[1], np.newaxis]
    maps = np.concatenate(maps, axis=-1)
    return maps, total_map


def create_label_map(mnt_file_path):
    mnt_dict_list = parse_minute_file(mnt_file_path)
    label_map = create_map_scipy(mnt_dict_list)
    return label_map


def main():
    output_path = r'D:\users\rafael\finger_print\Data\Fmaps'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    imgs_path = r'D:\users\rafael\finger_print\Data\NIST14\test\Fimages'
    txts_path = r'D:\users\rafael\finger_print\Data\NIST14\test\Fiso'
    imgs_name_list = os.listdir(imgs_path)
    for idx, img_name in enumerate(imgs_name_list):
        img_path = os.path.join(imgs_path, img_name)
        mnt_file_path = os.path.join(txts_path, img_name.replace('tif', 'txt'))
        # start_time = time.time()
        mnt_dict_list = parse_minute_file(mnt_file_path)
        # print("time to parse file: {}".format(time.time() - start_time))
        # start_time = time.time()
        # map, total_map = create_map(mnt_dict_list, img_path)
        map = create_map_scipy(mnt_dict_list)
        img = plt.imread(img_path)
        plt.imshow(map * 150 + img[:, :, np.newaxis])
        plt.show()
        # print("time to create map: {}".format(time.time() - start_time))
        np.save(os.path.join(output_path, img_name[:-4] + '_map.npy'), map)
        plt.imsave(os.path.join(output_path, img_name[:-4] + '_map_{}.jpg'.format('total')), map)
        if idx % 100 == 0:
            print("Save {}/{} maps".format(idx, len(imgs_name_list)))
        # print("time to create maps: {}".format(time.time() - start_time))


if __name__ == "__main__":
    main()