import numpy as np
import cv2
from util import load_color_LUT_21, get_list, resize, padding, crop, mirror, get_mask


# a generator which generate data and corresponding label for batch_size numer
def preprocess(source, data_root, label_root, mean, crop_height, crop_width, is_shuffle, is_mirror, show):
    # get_list
    l = get_list(source)

    while True:
        # shuffle
        if is_shuffle:
            np.random.shuffle(l)

        for data_source, label_source in l:
            # load image
            im_data = cv2.imread('{}/{}'.format(data_root, data_source))
            im_label = cv2.imread('{}/{}'.format(label_root, label_source))[:, :, 0]

            if show:
                print 'im_data shape is', im_data.shape
                cv2.imshow('ori', im_data)

            # resize image
            scale = [0.5, 0.75, 1, 1.25, 1.5]
            resize_scale = np.random.choice(scale, 1)
            im_data = resize(im_data, resize_scale, cv2.INTER_LINEAR)
            im_label = resize(im_label, resize_scale, cv2.INTER_NEAREST)

            if show:
                print 'after resize, im_data shape is', im_data.shape
                cv2.imshow('after_resize', im_data)

            # convert to np.float32
            im_data = np.float32(im_data)

            # pad image
            im_data = padding(im_data, (crop_height, crop_width), 'corner', mean)
            im_label = padding(im_label, (crop_height, crop_width), 'corner', 255)

            if show:
                print 'after pad, im_data shape is', im_data.shape
                cv2.imshow('after_pad', np.uint8(im_data))

            # crop
            im_data = crop(im_data, (crop_height, crop_width), 'corner')
            im_label = im_label[..., np.newaxis]
            im_label = crop(im_label, (crop_height, crop_width), 'corner')

            if show:
                print 'after crop, im_data shape is', im_data.shape
                cv2.imshow('after_crop', np.uint8(im_data))

            # mirror image
            if is_mirror:
                im_data = mirror(im_data)
                im_label = mirror(im_label)

            if show:
                print 'after mirror, im_data.shape is', im_data.shape
                cv2.imshow('after_mirror', np.uint8(im_data))
                cv2.waitKey(0)

            # substract mean
            im_data -= mean

            im_data = im_data.transpose((2, 0, 1))
            im_label = np.array(im_label, dtype=np.uint8)
            im_label = im_label[np.newaxis, :, :]
            im_label_w = get_mask(im_label)

            yield [im_data.T, im_label.T, im_label_w.T]
