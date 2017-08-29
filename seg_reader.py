from preprocess import preprocess


class SegReader:
    support_keys = ['source', 'data_root', 'label_root', 'crop_height',
                    'crop_width', 'is_shuffle', 'is_mirror']

    def config(self, cfg):
        source = cfg['source']
        data_root = cfg['data_root']
        label_root = cfg['label_root']
        mean = (104.00698793, 116.66876762, 122.67891434)
        crop_height = cfg.get('crop_height', 321)
        crop_width = cfg.get('crop_width', 321)
        is_shuffle = cfg.get('is_shuffle', True)
        is_mirror = cfg.get('is_mirror', True)
        self.gen = preprocess(source, data_root, label_root, mean, crop_height, crop_width, is_shuffle, is_mirror, False)

    def read(self):
        return self.gen.next()


def test():

    reader = SegReader()
    reader.config({
        'source': '/home/yhcao6/VOC_arg/train.txt',
        'data_root': '/home/yhcao6/VOC_arg',
        'label_root': '/home/yhcao6/VOC_arg'
        })

    for i in range(10):
        im_data, im_label, im_label_w = reader.read()
        print im_data.shape, im_label.shape, im_label_w.shape


if __name__ == '__main__':
    test()
else:
    from parrots.dnn import reader
    reader.register_pyreader(SegReader, 'seg_reader')
