from tensorflow.keras.layers import (
    Input, Conv2D, Relu, Activation, Maxpoll2D, concatenate, add, BatchNormalization
    )
from tensorflow.keras.models import Model

def make_net(input_shape):
    img_layer = Input(shape=input_shape)

def conv_layer(filters, kernes_size, input0):
    conv = Conv2D(filters=filters, kernes_size=kernes_size)(input0)
    conv = BatchNormalization()(conv)
    conv = ReLu(conv)
    return conv

def custom_block(input0, input1):
    conv = conv_layer(32, (1, 1), input0)
    upsamp = UpSampling2D()(conv)
    conv_befor_add = conv_layer(32, (1, 1), input1)
    add = add()([conv_befor_add, upsamp])
    return add

def make_net(input_shape):
    img_layer = Input(shape = input_shape)
    
    #tiny darknet
    conv0 = conv_layer(16, (3, 3), img_layer)
    pool1 = MaxPool2d(pool_size=(2,2), strides=(2, 2))(conv0)
    conv2 = conv_layer(32, (3, 3), pool1)
    pool3 = MaxPool2d(pool_size=(2,2), strides=(2, 2))(conv2)
    conv4 = conv_layer(16, (1, 1), pool3)
    conv5 = conv_layer(128, (3, 3), conv4)
    conv6 = conv_layer(16, (1, 1), conv5)
    conv7 = conv_layer(128, (3, 3), conv6)
    pool8 = MaxPool2d(pool_size=(2,2), strides=(2, 2))(conv7)
    conv9 = conv_layer(32, (1, 1), pool8)
    conv10 = conv_layer(256, (3, 3), conv9)
    conv11 = conv_layer(32, (1, 1), conv10)
    conv12 = conv_layer(256, (3, 3), conv11)
    pool13 = MaxPool2d(pool_size=(2,2), strides=(2, 2))(conv11)
    conv14 = conv_layer(64, (1, 1), pool13)
    conv15 = conv_layer(512, (3, 3), conv14)
    conv16 = conv_layer(64, (1, 1), conv15)
    conv17 = conv_layer(512, (3, 3), conv16)
    conv18 = conv_layer(128, (1, 1), conv17)

    #FPN
    conv19 = conv_layer(32, (1, 1), conv18)

    add20 = custam_block(conv19, conv12)
    add21 = custam_block(add20, conv7)
    add22 = custam_block(add21, conv2)

    conv23_1 = conv_layer(16, (3, 3), add20)
    conv23_2 = conv_layer(16, (3, 3), add21)
    conv23_3 = conv_layer(16, (3, 3), add22)

    conv24_1 = conv_layer(16, (3, 3), conv23_1)
    conv24_2 = conv_layer(16, (3, 3), conv23_2)
    conv24_3 = conv_layer(16, (3, 3), conv23_3)

    upsamp25_1 = UpSampling2D(size=(4, 4))(conv24_1)
    upsamp25_2 = UpSampling2D()(conv24_2)

    concatted26 = Concatenate()([upsamp25_1, upsamp25_2, conv24_3])
    conv27 = conv_layer(16, (3, 3), concatted26)
    upsamp28 = UpSampling2D()(conv27)
    conv_out = Conv2D(filters=2, kernel_size=(3,3), activation="sigmoid")(upsamp28)
    
    model = Model(inputs=[img_layer], outputs=[conv_out])
    return model

if __name__ == '__main__':
    net = make_net((224, 224, 3))
    net.compile('sgd', 'categorical_crossentropy')
    net.summary()
    test = np.zeros((1000, 224, 224, 3), np.float32)
    net.predict(test, 1, verbose=1)




 
