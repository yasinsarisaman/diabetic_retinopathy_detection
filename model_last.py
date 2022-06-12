# -- coding: utf-8 --
"""
Created on Thu Mar 31 15:48:00 2022

@author: yasin
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from keras import backend as K
from glob import glob
import json

train_path = "../DATASET/DR_DATA/TRAIN/TRAIN_RESIZED"
test_path = "../DATASET/DR_DATA/TEST/TEST_RESIZED"

img_size = 150

class_names = glob(train_path + '/*')
numb_of_classes = len(class_names)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_size, img_size)
else:
    input_shape = (img_size, img_size, 3)

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(units = numb_of_classes))
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

batch_size = 32


# DATA AUGMENTATION
train_datagen = ImageDataGenerator(rescale = 1./255,
                   shear_range = 0.3,
                   horizontal_flip = (True),
                   zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical')


test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical')

model_hist = model.fit_generator(generator = train_generator,
                    steps_per_epoch = (800//batch_size),
                    epochs = 15,
                    validation_data = test_generator,
                    validation_steps = 400//batch_size)

print(model_hist.history.keys())
# summarize history for accuracy
plt.plot(model_hist.history['accuracy'])
plt.plot(model_hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("model.h5")