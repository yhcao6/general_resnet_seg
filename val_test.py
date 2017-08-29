import cv2
import numpy as np
import sys
import os
from util import fast_hist, load_color_LUT_21, padding, cal_accuracy, cal_IoU
from run_parrots import model_and_session
from tester import Tester

sys.path.append('/home/yhcao6')
import ext_layer

from parrots import base
base.set_debug_log(True)


def test_eval_seg(model, session, param, test_list, data_root, gt_root, query, mean=np.array([104.008, 116.669, 122.675]), batch_size=1, uniform_size=513, show=False, save=False):
    # get file list
    with open(test_list) as infile:
        img_list = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

    base_size = [uniform_size, uniform_size]

    tester = Tester(model, session, param)

    hist = np.zeros((21, 21))
    # for i in range(0, 5, batch_size):
    for i in range(0, len(img_list), batch_size):
        if i % (batch_size*100) == 0:
            print 'Processing: %d/%d' % (i, len(img_list))
        true_batch_size = min(batch_size, len(img_list)-i)

        batch_data = np.zeros((batch_size, 3, base_size[0], base_size[1]), dtype=np.float)
        for k in range(true_batch_size):
            img = cv2.imread(data_root + img_list[i + k].split(' ')[0])
            img = np.float32(img)

            # padding
            img = padding(img, base_size, 'corner', mean)

            # substract by mean
            img -= mean

            # N x C x H x W
            batch_data[k, ...] = img.transpose((2, 0, 1))

        # feed data
        data = {}
        data['data'] = batch_data.T

        # predict
        pred = tester.predict(data, query)

        for k in range(0, true_batch_size):
            origin_img = cv2.imread(data_root + img_list[i + k].split(' ')[0])
            origin_shape = origin_img.shape
            
            gt_img = cv2.imread(data_root + img_list[i + k].split(' ')[1])[:, :, 0]

            tmp = pred[k, ...].transpose((1, 2, 0))
            tmp = tmp.argmax(axis=2)
            # get the roi
            cls_map = np.array(tmp, dtype=np.uint8)[0: origin_shape[0], 0: origin_shape[1]]

            # show image
            if show:
                LUT = load_color_LUT_21('./VOC_color_LUT_21.mat')
                tmp = np.zeros((256, 3))
                tmp[0:21, ...] = LUT
                tmp[255] = [0, 1, 1]
                LUT = tmp

                # visualize
                out_map = np.uint8(LUT[cls_map] * 255)
                gt_map = np.uint8(LUT[gt_img] * 255)

                both = np.hstack((out_map, gt_map))
                cv2.imshow('seg result', both)
                cv2.waitKey(0)

            # save image in res dir
            if save:
                cwd = os.getcwd()
                if not os.path.isdir(cwd+'/res'):
                    os.makedirs(cwd+'/res')

                LUT = load_color_LUT_21('./VOC_color_LUT_21.mat')
                tmp = np.zeros((256, 3))
                tmp[0:21, ...] = LUT
                tmp[255] = [0, 1, 1]
                LUT = tmp

                # visualize
                out_map = np.uint8(LUT[cls_map] * 255)
                gt_map = np.uint8(LUT[gt_img] * 255)

                cv2.imwrite(cwd + '/res/' + 'gt_' + str(i) + '.png', gt_map)
                cv2.imwrite(cwd + '/res/' + 'predict_' + str(i) + '.png', out_map)
                cv2.imwrite(cwd + '/res/' + 'ori_' + str(i) + '.png', origin_img)

            hist += fast_hist(gt_img.flatten(), cls_map.flatten(), 21)

    # cal metrics
    cal_accuracy(hist)
    cal_IoU(hist)


if __name__ == '__main__':
    test_eval_seg(model='/home/yhcao6/resnet_seg/val_model.yaml', session='/home/yhcao6/resnet_seg/val_session.yaml', param='/home/yhcao6/resnet_seg/work_dir/snapshots/iter.00020000.parrots', test_list='/home/yhcao6/val.txt', data_root='/home/yhcao6/VOC_arg', gt_root='/home/yhcao6/VOC_arg/SegmentationClass_label', query='fc_fusion', show=False, save=False, batch_size=1)
