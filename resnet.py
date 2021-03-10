import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Add, ZeroPadding2D


# Deep nn is good. but hard to train.
# Resnet can handle that problem!

# when input/ output dimension matches up
def identy_block(x, f, filters, stage, block):
    # define name base
    conv_name_base = 'res' + str(stage) + block + '_'
    bn_name_base = 'bn' + str(stage) + block + '_'

    f1, f2, f3 = filters
    x_shortcut = x

    x = Conv2D(f1, 1, name=conv_name_base + 'a')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + 'a')(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, f, padding='same', name=conv_name_base + 'b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + 'b')(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, 1, name=conv_name_base + 'c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + 'c')(x)

    x = Add()([x_shortcut, x])
    x = Activation('relu')(x)
    return x

# when input/ output dimension doesnt match up
# use conv layer to adjust dim of shortcut path
def conv_block(x, f, filters, stage, block, s=2):
    # define name base
    conv_name_base = 'res' + str(stage) + block + '_'
    bn_name_base = 'bn' + str(stage) + block + '_'

    f1, f2, f3 = filters
    x_shortcut = x

    x = Conv2D(f1, 1, strides=s, name=conv_name_base + 'a')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + 'a')(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, f, padding='same', name=conv_name_base + 'b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + 'b')(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, 1, name=conv_name_base + 'c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + 'c')(x)

    x_shortcut = Conv2D(f3, 1, strides=s, name=conv_name_base + '1')(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(x_shortcut)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    x_input = Input(shape=input_shape)

    # zero pad input
    x = ZeroPadding2D(3)(x_input)

    # stage1
    # output shape: (batch_size, 64, 64, 64)
    x = Conv2D(filters=64, kernel_size=7, strides=2, name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # stage2
    # output shape: (batch_size, 64, 64, 256)
    x = conv_block(x=x, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    x = identy_block(x=x, f=3, filters=[64, 64, 256], stage=2, block='b')
    x = identy_block(x=x, f=3, filters=[64, 64, 256], stage=2, block='c')

    # stage3
    # output shape: (batch_size, 32, 32, 512)
    x = conv_block(x=x, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    x = identy_block(x=x, f=3, filters=[128, 128, 512], stage=3, block='b')
    x = identy_block(x=x, f=3, filters=[128, 128, 512], stage=3, block='c')
    x = identy_block(x=x, f=3, filters=[128, 128, 512], stage=3, block='d')

    # stage4
    # output shape: (batch_size, 16, 16, 1024)
    x = conv_block(x=x, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    x = identy_block(x=x, f=3, filters=[256, 256, 1024], stage=4, block='b')
    x = identy_block(x=x, f=3, filters=[256, 256, 1024], stage=4, block='c')
    x = identy_block(x=x, f=3, filters=[256, 256, 1024], stage=4, block='d')
    x = identy_block(x=x, f=3, filters=[256, 256, 1024], stage=4, block='e')
    x = identy_block(x=x, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # stage5
    # output shape: (batch_size, 16, 16, 1024)
    x = conv_block(x=x, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    x = identy_block(x=x, f=3, filters=[512, 512, 2048], stage=5, block='b')
    x = identy_block(x=x, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # avg pool and flatten to FC layer
    x = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(units=classes, activation='softmax', name='fc')(x)

    # create model
    model = Model(inputs=x_input, outputs=x, name='ResNet50')
    return model

resnet = ResNet50()
resnet.summary()