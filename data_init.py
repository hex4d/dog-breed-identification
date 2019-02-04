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


# def create_test_directory():
#     test_data_list = os.listdir(test_dir)
#     os.mkdir(os.path.join(test_dir, '0')) # for predict 
#     for test_data in test_data_list:
#         shutil.move(os.path.join(test_data, ))

create_dataset_directory()
