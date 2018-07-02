#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from keras.initializers import Constant
from keras import regularizers
from keras.regularizers import l2
import keras.backend as K
import pickle, time, random
from random import randint
from copy import deepcopy
from shutil import copy
import argparse

from softrank import SoftRankRegularizer

parser = argparse.ArgumentParser(description='Inputs for Mini AlexNet variants.')
parser.add_argument('--batch_size', metavar='b', type=int, default=1, help='batch size')
parser.add_argument('--long_run', metavar='l', type=bool, default=False, help='long run')
parser.add_argument('--regularize', metavar='w', type=bool, default=False, help='weight regularizer')
parser.add_argument('--batch_norm', metavar='n', type=bool, default=False, help='batch normalization')
parser.add_argument('--random', metavar='r', type=int, default=0, help='% labels randomized')
parser.add_argument('--id', metavar='i', type=int, default=0, help='id of run')
parser.add_argument('--save', metavar='s', type=bool, default=False, help='save intermediate model files')
parser.add_argument('--softrank_k', metavar='k', type=int, default=0.1, help='k value for softrank regularization')

args = parser.parse_args()

filename = "weights/alexnet.b{}".format(args.batch_size)
if args.regularize:
        filename+=".wd"
if args.random > 0:
    filename+=".rand{}".format(args.random)
if args.id > 0:
    filename+=".id{}".format(args.id)
        
print(filename)
print(args)

                

model = Sequential()
model.add(Conv2D(96, (5, 5), input_shape=(28, 28, 3), kernel_initializer=
                 'glorot_normal', bias_initializer=Constant(0.1), padding=
                 'same', activation='relu',kernel_regularizer=SoftRankRegularizer(args.softrank_k))) #Applying SoftRank regularization with C=0.01
 
model.add(MaxPooling2D((3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (5, 5), kernel_initializer='glorot_normal',
                 bias_initializer=Constant(0.1), padding='same',
                 activation='relu',kernel_regularizer=SoftRankRegularizer(args.softrank_k))) 
model.add(MaxPooling2D((3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Flatten())

if args.regularize:
        model.add(Dense(384, kernel_initializer='glorot_normal', kernel_regularizer=l2(1e-4),
                bias_initializer=Constant(0.1), activation='relu'))
        model.add(Dense(192, kernel_initializer='glorot_normal', kernel_regularizer=l2(1e-4),
                        bias_initializer=Constant(0.1), activation='relu'))
else:
        model.add(Dense(384, kernel_initializer='glorot_normal',
                        bias_initializer=Constant(0.1), activation='relu'))
        model.add(Dense(192, kernel_initializer='glorot_normal',
                        bias_initializer=Constant(0.1), activation='relu'))



model.add(Dense(10, kernel_initializer='glorot_normal',
        bias_initializer=Constant(0.1), activation='softmax'))

# use early stopping
# unless using 75 or more epochs
min_delta = 0.0001
prev_loss = 1e4
epochs = 100
if args.long_run is True:
        min_delta = 0.0000001
        prev_loss = 1e10
early_stop = EarlyStopping(monitor='loss', min_delta=min_delta, patience=5)


now = str(time.time())
tb_callback = TensorBoard(log_dir='../Tensorboard/alexnet/' + now)

img = tf.placeholder(tf.float32, [28, 28, 3])
norm_image = tf.image.per_image_standardization(img)

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

cifar10_train_images = []
cifar10_train_labels = []
print "Loading training images..."
for i in range(1, 6):
    train_file = open('../../cifar-10-batches-py/data_batch_' + str(i), 'r')
    train_dict = pickle.load(train_file)
    for image, label in zip(train_dict['data'], train_dict['labels']):
        image_red = np.reshape(image[:1024], (32, 32))[2:-2, 2:-2] / 255.0
        image_red = np.reshape(image_red, (28, 28, 1))
        image_green = np.reshape(image[1024:2048], (32, 32))[2:-2,
                                                             2:-2] / 255.0
        image_green = np.reshape(image_green, (28, 28, 1))
        image_blue = np.reshape(image[2048:3072], (32, 32))[2:-2, 2:-2] / 255.0
        image_blue = np.reshape(image_blue, (28, 28, 1))
        image = np.concatenate([image_red, image_green, image_blue], axis=-1)
        image = norm_image.eval(feed_dict={img:image})
        cifar10_train_images.append(image)
        
        label = np.identity(10)[label]
        if args.random > 0:
                if randint(0,100) < args.random:
                        label = np.identity(10)[randint(0, 9)]
                        
        cifar10_train_labels.append(label)
    train_file.close()

batch_size = args.batch_size


patience = deepcopy(early_stop.patience)

model.save("{}.e{}.h5".format(filename,0))
        
for epoch in range(epochs):
    hist = model.fit(np.array(cifar10_train_images), np.array(
                     cifar10_train_labels), epochs=(epoch + 1),
                     batch_size=batch_size, initial_epoch=epoch,
                     callbacks=[tb_callback])

    if args.save:
            model.save("{}.e{}.h5".format(filename,epoch))
            
    K.set_value(opt.lr, 0.95 * K.get_value(opt.lr))
    if hist.history[early_stop.monitor][0] - prev_loss > early_stop.min_delta:
        patience -= 1
    else:
        patience = deepcopy(early_stop.patience)
    if patience <= 0:
        #break
        prev_loss = hist.history[early_stop.monitor][0]
    else:
        prev_loss = hist.history[early_stop.monitor][0]

del cifar10_train_images, cifar10_train_labels
print "Loading test images..."
cifar10_test_images = []
cifar10_test_labels = []
test_file = open('../../cifar-10-batches-py/test_batch', 'r')
test_dict = pickle.load(test_file)
for image, label in zip(test_dict['data'], test_dict['labels']):
    image_red = np.reshape(image[:1024], (32, 32))[2:-2, 2:-2] / 255.0
    image_red = np.reshape(image_red, (28, 28, 1))
    image_green = np.reshape(image[1024:2048], (32, 32))[2:-2,
                                                            2:-2] / 255.0
    image_green = np.reshape(image_green, (28, 28, 1))
    image_blue = np.reshape(image[2048:3072], (32, 32))[2:-2, 2:-2] / 255.0
    image_blue = np.reshape(image_blue, (28, 28, 1))
    image_blue = np.reshape(image_blue, (28, 28, 1))
    image = np.concatenate([image_red, image_green, image_blue], axis=-1)
    image = norm_image.eval(feed_dict={img:image})
    cifar10_test_images.append(image)
    label = np.identity(10)[label]
    cifar10_test_labels.append(label)
test_file.close()

print(model.evaluate(np.array(cifar10_test_images),
    np.array(cifar10_test_labels), batch_size=256))

model.save("{}.final.h5".format(filename))



