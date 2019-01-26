from keras import models
from keras import layers
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

import os
import pandas as pd
import shutil

base_dir = './data/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

validation_rate = 0.2

class_csv_file = os.path.join(base_dir, 'labels.csv')

seed = 1

def class_id_dictionary():
    # open csv and get dictionary of id and class
    class_id_dict = pd.read_csv(class_csv_file, index_col='id')
    return class_id_dict

def create_dataset_directory():
    train_data_list = os.listdir(train_dir)
    class_id_dict = class_id_dictionary()
    for train_data in train_data_list:
        print(train_data)
        class_name = class_id_dict.at[train_data.replace('.jpg', ''), 'breed']
        dist_path = os.path.join(train_dir, class_name)
        if not os.path.exists(dist_path):
            os.mkdir(dist_path)
        shutil.move(os.path.join(train_dir, train_data), dist_path)

input_size = 150
output_size = 120
def simple_cnn_model():
    model = models.Sequential()
    model.add(layers.Convolution2D(32, (3, 3), input_shape=(input_size, input_size, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(128, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(256, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    # model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    return model

batch_size = 20
validation_split = 0.2
def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        horizontal_flip=True
    )
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(input_size, input_size),
        subset='training',
        batch_size=batch_size,
        seed=seed,
        class_mode='categorical',
    )
    val_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(input_size, input_size),
        subset='validation',
        batch_size=32,
        seed=seed,
        class_mode='categorical',
    )
    return train_generator, val_generator


model = simple_cnn_model()

epochs = 30
train_generator, val_generator = load_data()

model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epochs, validation_data=val_generator, validation_steps=50)


