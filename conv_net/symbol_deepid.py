'''
This is symbol for deepid1.
'''
import mxnet as mx


def get_symbol(num_classes):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data,
        kernel=(4, 4),
        stride=(1, 1),
        num_filter=20,
        name='conv1')
    relu1 = mx.symbol.Activation(
        data=conv1,
        act_type="relu",
        name='relu1')
    pool1 = mx.symbol.Pooling(
        data=relu1,
        pool_type="max",
        kernel=(2, 2),
        stride=(2, 2),
        name='pool1')

    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1,
        kernel=(3, 3),
        stride=(1, 1),
        num_filter=40,
        name='conv2')
    relu2 = mx.symbol.Activation(
        data=conv2,
        act_type="relu",
        name='relu2')
    pool2 = mx.symbol.Pooling(
        data=relu2,
        kernel=(2, 2),
        stride=(1, 1),
        pool_type="max",
        name='pool2')

    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2,
        kernel=(3, 3),
        stride=(1, 1),
        num_filter=60,
        name='conv3')
    relu3 = mx.symbol.Activation(
        data=conv3,
        act_type="relu",
        name='relu3')
    pool3 = mx.symbol.Pooling(
        data=relu3,
        kernel=(2, 2),
        stride=(2, 2),
        pool_type="max",
        name='pool3')

    conv4 = mx.symbol.Convolution(
        data=pool3,
        kernel=(2, 2),
        stride=(1, 1),
        num_filter=80,
        name='conv4')

    # stage 4
    flatten_layer3 = mx.symbol.Flatten(data=pool3, name='ft3')
    flatten_layer4 = mx.symbol.Flatten(data=conv4, name='ft4')
    f3 = mx.symbol.FullyConnected(data=flatten_layer3, num_hidden=160, name='fc1')
    f4 = mx.symbol.FullyConnected(data=flatten_layer4, num_hidden=160, name='fc2')
    deepid = mx.symbol.ElementWiseSum(f3,f4, name='fc3')
    # deepid_out = mx.symbol.Activation(data=deepid, act_type='relu', name='relu5')
    dropout = mx.symbol.Dropout(data=deepid, p=0.4, name='drop')
    fc_class = mx.symbol.FullyConnected(data=dropout, num_hidden=num_classes, name='fc4')
    # stage 6
    softmax = mx.symbol.SoftmaxOutput(data=fc_class, name='softmax')
    return softmax
