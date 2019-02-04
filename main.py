from keras import models
from keras import layers
from keras.applications import resnet50
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import shutil
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

input_size = 224
output_size = 120

base_dir = './data/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


class_csv_file = os.path.join(base_dir, 'labels.csv')

seed = 1470

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

def get_model():
    conv_base = resnet50.ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(input_size, input_size, 3))
    main_input = conv_base.input
    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    for layer in conv_base.layers:
        layer.trainable = False;
    # main_input = conv_base.input
    # embedding = conv_base.output
    # features = layers.Flatten()(embedding)
    # features2 = layers.GlobalMaxPooling2D()(embedding)
    # hid = layers.Concatenate()([features, features2])
    # hid = layers.Dense(2048)(hid)
    # hid = layers.BatchNormalization()(hid)
    # hid = layers.Activation('relu')(hid)
    # new or known
    # hid1 = layers.Dense(512)(hid)
    x = layers.Dense(120)(x)
    x = layers.Activation('softmax')(x)
    model = models.Model(inputs=[main_input], outputs=[x])
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                        metrics=['accuracy'])
    return model

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
    model.add(layers.Convolution2D(512, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    # model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    return model

def simple_cnn_model2():
    model = models.Sequential()
    model.add(layers.Convolution2D(16, (3, 3), input_shape=(input_size, input_size, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(48, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    return model

def vgg_cnn_model():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    conv_base.trainable = True
    for layer in conv_base.layers[:15]:
        layer.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['acc'])
    return model

def vgg_cnn_model2():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(input_size, input_size, 3))
    conv_base.trainable = True
    for layer in conv_base.layers[:15]:
        layer.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['acc'])
    return model

batch_size = 32
validation_split = 0.1

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
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

def train_model(model, train_generator, val_generator, model_name):
    base_dir = 'models/'
    model_path = os.path.join(base_dir, model_name)
    if not os.path.exists(model_path):
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        os.mkdir(model_path)
    classes= train_generator.class_indices
    df = pd.DataFrame.from_dict(classes, orient='index')
    df.to_csv(os.path.join(model_path, 'train_classes.cls'))
    epochs = 100
    # eary stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=50,
        callbacks=[early_stopping]
    )
    model_file_path = os.path.join(model_path, 'model.h5')
    model.save(model_file_path)
    plt.plot(history.history['val_loss'], 'b', label='val_loss')
    plt.plot(history.history['loss'], 'r', label='loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'loss.png') )
    plt.clf()
    plt.plot(history.history['val_acc'], 'b', label='val_acc')
    plt.plot(history.history['acc'], 'r', label='acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'acc.png') )
    plt.clf()

model = get_model()
train_generator, val_generator = load_data()
history = train_model(model, train_generator, val_generator, 'single_resnet')

