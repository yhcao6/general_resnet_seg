# general_resnet_seg
General Version \\
\\
\\
copy_params.py:       transform .caffemodel to h5 format \\
preprocess.py:        data preprocess for training \\
run_parrots.py:       load model and session config \\
seg_reader.py:        the Reader, generate a turple of format (data, label) \\
test_resnet.py:       test net definition \\
tester.py:            tester class \\
train_resnet.py:      train net definition \\
train_session.yaml:   training parameters \\
util.py:              necessary functions in training and testing \\
val_session.yaml:     test parameters \\
val_test.py:          test the model \\
