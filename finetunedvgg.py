import random

import keras.applications.imagenet_utils
import numpy as np
import matplotlib.pyplot as plt
import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import glob
from keras.datasets import cifar10
from keras.utils import to_categorical,image_dataset_from_directory
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass
from mnistmlp import plot_results_mlp
from keras.applications import vgg16,resnet,inception_v3

SEED_VALUE = 41

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

def processImages(model,size,preprocessmethod,imagepaths,topk = 2,top_k = False):
    predicted_labels = []
    confidences = []
    for idx,imagepath in enumerate(imagepaths):
        image = tf.io.read_file(imagepath)
        decoded_image = tf.image.decode_image(image)
        resized_image = tf.image.resize(decoded_image)
        expanded = tf.expand_dims(resized_image,axis=0)
        preprocessedimage = preprocessmethod(expanded)
        predicted = model(preprocessedimage)
        postprocessed_image = keras.applications.imagenet_utils.decode_predictions(predicted,5)
        label = postprocessed_image[0][0][1]
        confidence = postprocessed_image[0][0][2]
        predicted_labels.append(label)
        confidences.append(confidence)
    return predicted_labels,confidences


def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

def preprocess_image(image):
    # Decode and resize image.
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [DatasetConfig.IMAGEHEIGHT, DatasetConfig.IMAGEWIDTH])
    return image
def load_and_preprocess_image(path):
    # Read image into memory as a byte string.
    image = tf.io.read_file(path)
    return preprocess_image(image)
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

URL = r"https://www.dropbox.com/s/uzgh5g2bnz40o13/dataset_traffic_signs_40_samples_per_class.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "TF-Keras-Bootcamp-NB07-assets.zip")
dataset_path   = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class")
# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 32
    EPOCHS: int =101
    LEARNINING_RATE: float = 0.0001
    DROPOUT: float = 0.6
    LAYERS_FINETUNE: int = 8
@dataclass(frozen=True)
class DatasetConfig:
    IMAGEWIDTH: int = 224
    IMAGEHEIGHT: int = 224
    IMAGECHANNELS: int = 3
    NUMCLASSES: int = 43

    DATA_ROOT_TRAIN: str = os.path.join(dataset_path, "Train")
    DATA_ROOT_VALID: str = os.path.join(dataset_path, "Valid")
    DATA_ROOT_TEST: str = os.path.join(dataset_path, "Test")
    DATA_TEST_GT: str = os.path.join(dataset_path, "Test.csv")


training_data = image_dataset_from_directory(DatasetConfig.DATA_ROOT_TRAIN,batch_size=TrainingConfig.BATCH_SIZE,label_mode='int',image_size = (DatasetConfig.IMAGEHEIGHT,DatasetConfig.IMAGEWIDTH),seed=SEED_VALUE,shuffle=True)
validation_data = image_dataset_from_directory(DatasetConfig.DATA_ROOT_VALID,batch_size=TrainingConfig.BATCH_SIZE,label_mode='int',image_size=(DatasetConfig.IMAGEHEIGHT,DatasetConfig.IMAGEWIDTH),seed=SEED_VALUE,shuffle=True)

import pandas as pd

input_file = DatasetConfig.DATA_TEST_GT

dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
cols = [6]
df = df[df.columns[cols]]
ground_truth_ids = df["ClassId"].values.tolist()
print("Total number of Test labels: ", len(ground_truth_ids))
print(ground_truth_ids[0:10])

# Convert train/valid class names to integers.
class_names_int = list(map(int, training_data.class_names))

# Create a dictionary mapping ground truth IDs to class name IDs.
gtid_2_cnidx = dict(zip(class_names_int, range(0, DatasetConfig.NUMCLASSES)))

label_ids = []
for idx in range(len(ground_truth_ids)):
    label_ids.append(gtid_2_cnidx[ground_truth_ids[idx]])

image_paths = sorted(glob.glob(DatasetConfig.DATA_ROOT_TEST + os.sep + "*.png"))
test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_ids))

# Apply the functions above to the test dataset.
test_dataset = test_dataset.map(load_and_preprocess_from_path_label)

# Set the batch size for the dataset.
test_dataset = test_dataset.batch(TrainingConfig.BATCH_SIZE)

featureext_model = vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(DatasetConfig.IMAGEHEIGHT,DatasetConfig.IMAGEWIDTH,DatasetConfig.IMAGECHANNELS))
featureext_model.trainable = True
start_layer = TrainingConfig.LAYERS_FINETUNE
end_layer = len(featureext_model.layers)

for idx in featureext_model.layers[:end_layer-start_layer]:
    idx.trainable = False

input = keras.Input(shape=(DatasetConfig.IMAGEHEIGHT,DatasetConfig.IMAGEWIDTH,DatasetConfig.IMAGECHANNELS))
x = vgg16.preprocess_input(input)
x = featureext_model(x)
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(TrainingConfig.DROPOUT)(x)

outputs = Dense(43,activation='softmax')(x)

finetuned_vgg16 = keras.Model(input,outputs)
finetuned_vgg16.compile(optimizer=keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNINING_RATE),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=["accuracy"])
results = finetuned_vgg16.fit(training_data,epochs=TrainingConfig.EPOCHS,validation_data=validation_data)
training_loss = results.history["training_loss"]
training_accuracy = results.history["training_accuracy"]
validation_loss = results.history["val_loss"]
validation_accuracy = results.history["val_accuracy"]
plot_results_mlp([training_loss,validation_loss],"Training vs validation loss",ylabel="loss",y_lim=[0.0,5.0],metric_name=["Training Loss","Validation loss"],color=["b","g"])
plot_results_mlp([training_accuracy,validation_accuracy],"Training vs validation accuracy",ylabel="Accuracy",y_lim=[0.0,1.0],metric_name = ["Training Accuracy","validation Accuracy"],color=["b","g"])

loss,accuracy = finetuned_vgg16.evaluate(test_dataset[1])
predictions = finetuned_vgg16.predict(test_dataset[1])
cm = tf.math.confusion_matrix(test_dataset[0],predictions)
import seaborn as sn
sn.heatmap(cm,annot=True, fmt="d", annot_kws={"size": 12})
plt.show()