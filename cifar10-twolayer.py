'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, InputLayer, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf
import simnets.keras as sk
import numpy as np
import os
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0,3,1,2)
    x_test = x_test.transpose(0,3,1,2)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(InputLayer(input_shape=(3,32,32)))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', use_bias=None,\
    data_format='channels_first', strides=(1,1)))
model.add(sk.Similarity(32, blocks=[1,1], strides=[1,1], similarity_function='L2',normalization_term=True, padding=[1,1],\
    out_of_bounds_value=np.nan, ignore_nan_input=True ))
model.add(sk.Mex(32,
              blocks=[1, 3, 3], strides=[32, 2, 2],
              softmax_mode=False, normalize_offsets=True,
              use_unshared_regions=True, unshared_offset_region=[2]))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', use_bias=None,\
    data_format='channels_first', strides=(1,1)))
model.add(sk.Similarity(64, blocks=[1,1], strides=[1,1], similarity_function='L2',normalization_term=True, padding=[1,1],\
    out_of_bounds_value=np.nan, ignore_nan_input=True ))
model.add(sk.Mex(64,
              blocks=[1, 3, 3], strides=[64, 2, 2],
              softmax_mode=False, normalize_offsets=True,
              use_unshared_regions=True, unshared_offset_region=[2])) #w/4 = 8
model.add(sk.Mex(10,
              blocks=[1, 8, 8], strides=[64, 8, 8],
              softmax_mode=True, normalize_offsets=True,
              use_unshared_regions=True, unshared_offset_region=[2])) #max pooling
#model.add(keras.layers.AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid', data_format='channels_first'))
model.add(Flatten(data_format='channels_first'))
model.summary()
# initiate SGD with nesterov optimizer
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=True)

# Let's train the model using the optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train -= 0.5
x_test -= 0.5

if not data_augmentation:
    print('Not using data augmentation.')
    # init
    sk.perform_unsupervised_init(model, 'kmeans', layers=None, data=x_train, batch_size=100)
    # train
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    # init
    sk.perform_unsupervised_init(model, 'kmeans', layers=None, data=x_train, batch_size=100)
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
