import numpy as np
import cv2
import scipy.io as sio


# load color map
def load_color_LUT_21(fn):
    contents = sio.loadmat(fn)
    return contents.values()[0]


# read data image and label from txt file
# return a list [[data, label] ...]
def get_list(fn):
    l = []
    with open(fn) as f:
        for line in f:
            data, label = line.split(' ')[0: 2]
            if label[-1] == '\n':
                label = label[: -1]
            l.append((data, label))
    return l


# data augmentation
def resize(im, resize_scale, interpolation):
    return cv2.resize(im, (im.shape[1] * resize_scale, im.shape[0] * resize_scale), interpolation=interpolation)


# pad image if the image size is smaller than require size
# pad_kind can be corner or middle
def padding(im, size, pad_kind, pad_value):
    assert pad_kind in ('corner', 'middle')

    crop_height = size[0]
    crop_width = size[1]
    im = np.float32(im)
    pad_height = max(crop_height - im.shape[0], 0)
    pad_width = max(crop_width - im.shape[1], 0)

    if pad_height > 0 or pad_width > 0:
        if pad_kind == 'corner':
            im = cv2.copyMakeBorder(im, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=pad_value)
        elif pad_kind == 'middle':
            im = cv2.copyMakeBorder(im, pad_height / 2, pad_height - pad_height / 2, pad_width / 2, pad_width - pad_width / 2, cv2.BORDER_CONSTANT, value=pad_value)

    return im


# crop image if the image size is bigger than require size
# crop_kind can be corner or middle
def crop(im, size, crop_kind):
    assert crop_kind in ('corner', 'middle')

    crop_height = size[0]
    crop_width = size[1]
    if crop_kind == 'corner':
        h_off = np.random.randint(im.shape[0] - crop_height + 1)
        w_off = np.random.randint(im.shape[1] - crop_width + 1)
    elif crop_kind == 'middle':
        h_off = (im.shape[0] - crop_height) / 2
        w_off = (im.shape[1] - crop_width) / 2

    im = im[h_off: h_off + crop_height, w_off: w_off + crop_width, :]

    return im


# flip image
def mirror(im):
    if np.random.randint(0, 2) == 0:
        return cv2.flip(im, 1)
    else:
        return im


# get mask
def get_mask(im_label):
    ignore = im_label == 255
    im_label[ignore] = 0

    assert im_label.max() < 21

    im_label_w = im_label.copy()
    im_label_w[ignore] = 0
    im_label_w[~ignore] = 1

    return im_label_w


# confusion matrix (only count index whose value below n)
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


#  metrics
def cal_accuracy(hist):
    # class accuracy
    c_acc = np.diag(hist) / (hist.sum(1))
    print '>>>', 'per class acc:\n', c_acc
    show_acc = ["{:.2f}".format(i*100) for i in c_acc]
    print '>>>', show_acc

    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', 'overall accuracy', acc

    # mean accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', 'mean accuracy', np.nanmean(acc)


# IoU
def cal_IoU(hist):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', 'per class IU:\n', iu
    show_iu = ["{:.2f}".format(i*100) for i in iu]
    print '>>>', show_iu
    print '>>>', 'mean IU', np.nanmean(iu)




   
