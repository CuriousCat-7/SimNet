import simnets.keras as sk
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, AveragePooling2D, Lambda, Conv2D
from keras import backend as K
import numpy as np
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


#batch_size = 64
batch_size = 128
num_classes = 10
sim_kernel = 2
sim_channels = 32
mex_channels = sim_channels
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_last':
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.transpose(0,3,1,2)
    x_test = x_test.transpose(0,3,1,2)
    input_shape = (1, img_rows, img_cols)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train = x_train / 255.0 - 0.5
#x_test = x_test / 255.0 - 0.5
x_train /= 255.0
x_test /= 255.0

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def sum_pooling_layer(x, pool_size):
   x = AveragePooling2D(pool_size=pool_size, padding='valid', data_format='channels_first')(x)
   x = Lambda(lambda x: x * pool_size[0] * pool_size[1])(x)
   return x


a = Input(shape=(1, img_rows, img_cols))
b = Conv2D(9, 3, padding='same',strides=[1,1], data_format='channels_first', activation='relu', use_bias=False)(a) # attention! the strides cannot be 1
b = sk.Similarity(sim_channels,
                 blocks=[2, 2], strides=[2, 2], similarity_function='L2',
                 normalization_term=True, padding=[2, 2], out_of_bounds_value=np.nan, ignore_nan_input=True)(b)
while b.shape[-2:] != (1, 1):
   mex_channels *= 2
   b = sk.Mex(mex_channels,
              blocks=[int(b.shape[-3]), 1, 1], strides=[int(b.shape[-3]), 1, 1],
              softmax_mode=True, normalize_offsets=True,
              use_unshared_regions=True, unshared_offset_region=[2])(b)
   b = sum_pooling_layer(b, pool_size=(2, 2))

b = sk.Mex(num_classes,
          blocks=[mex_channels, 1, 1], strides=[mex_channels, 1, 1],
          softmax_mode=True, normalize_offsets=True,
          use_unshared_regions=True, shared_offset_region=[1])(b)
b = Flatten()(b)
model = Model(inputs=[a], outputs=[b])

print(model.summary())

def softmax_loss(y_true, y_pred):
   #return K.categorical_crossentropy(y_pred, y_true, True)
   return keras.losses.categorical_crossentropy(y_pred, y_true)

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.nadam(lr=5e-1, epsilon=1e-6),
             metrics=['accuracy'])

sk.perform_unsupervised_init(model, 'kmeans', layers=None, data=x_train, batch_size=100)

model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
