from keras import models
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os


test_dir = './data/test'
input_size = 128

def load_test_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    generator = datagen.flow_from_directory(
        test_dir,
        target_size=(input_size, input_size),
        class_mode=None,
        shuffle = False,
        batch_size=1,
    )
    return generator

def evaluate(model):
    test_generator = load_test_data()
    filenames = test_generator.filenames

    # model.summary()
    predictions = model.predict_generator(
        test_generator,
        steps=len(filenames)
    )

    df = pd.read_csv('train_classes.cls')
    columns = ['id']
    columns.extend(df.values[:,0])

    test_data_frame = pd.DataFrame(columns=columns)

    for i, prediction in enumerate(predictions):
        row_array = [filenames[i].replace('0/','').replace('.jpg','')]
        row_array.extend(prediction)
        row_data = pd.DataFrame([row_array], columns=columns)
        test_data_frame = test_data_frame.append(row_data)

    test_data_frame.to_csv('test_output.csv', index=False)


base_dir = 'models/'
def load_model(model_name):
    return models.load_model(os.path.join(base_dir + model_name, 'model.h5'))

# model_name = 'cnn_model2'
model_name = 'single_resnet'
model = load_model(model_name)
evaluate(model)
