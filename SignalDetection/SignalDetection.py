# -*- coding: utf-8 -*-

#%%
import numpy as np
from skimage import color, exposure
from skimage.transform import rescale, resize
NUM_CLASSES = 43
IMG_SIZE = 48

#%%

#%%
#Helper function for preprocessing the image

def preprocessing(img):
    ## Histogram equalization in HSV color space for v channel
    
    hsv = color. rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    
    ## central square crop
    min_side  = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]
    img = resize(img, (IMG_SIZE, IMG_SIZE))
    
    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    
    return img



#%%
    
#%%

from skimage import io
import os
import glob

## helper function to get the class id of the image ranging from 0 to 42
def get_class(img_path):
    return int(img_path.split('\\')[-2])

root_dir = 'GTSRB\Final_Training\Images\'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*\*.ppm'))
np.random.shuffle(all_img_paths)

print (len(all_img_paths))
#%%

#%%
for img_path in all_img_paths:
    print (img_path)
    img = preprocessing(io.imread(img_path))
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

## convert the preprocess image into aa numpy array

X = np.array(imgs, dtype='float32')

## Make one hot encoding od the targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
#print (labels)
#%%

#%%

## Keras Deep Learning Model

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_data_format('channels_first')


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model



#%%

#%%
from keras.optimizers import SGD    
model = cnn_model()

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#%%

#%%
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))    

batch_size = 32
epochs = 30

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('model.h5', save_best_only=True)]
          )
#%%













