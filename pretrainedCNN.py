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
from keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass
from mnistmlp import plot_results_mlp
from keras.applications import vgg16,resnet,inception_v3

def processImages(model,size,preprocessmethod,imagepaths,topk = 2,top_k = False):
    predicted_labels = []
    confidences = []
    for idx,imagepath in enumerate(image_paths):
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

URL = r"https://www.dropbox.com/s/uzgh5g2bnz40o13/dataset_traffic_signs_40_samples_per_class.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "TF-Keras-Bootcamp-NB07-assets.zip")
dataset_path  = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class")
# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)



""" For pre-trained models 
1. Load Image dataset
2. preprocessimage dataset(resize,expand,model.preprocess() etc)
3. make a prediction for each image using model.predict
4. Use imagenet_utilis.decode(prediction)
5. for labels use [0][0][1], for probability of the [0][0][2]
"""
image_paths = sorted(glob.glob("images" + os.sep + "*.png"))
model_VGG = vgg16.VGG16()
model_resnet = resnet.ResNet50()
model_incpetion = inception_v3.InceptionV3()

pre_process_vgg = vgg16.preprocess_input()
pre_process_resnet = resnet.preprocess_input()
pre_process_inception = inception_v3.preprocess_input()
predicted_labels_vgg,confidences_vgg = processImages(model_VGG,(224,224),pre_process_vgg,image_paths)
predicted_labels_resnet , confidences_resnet = processImages(model_resnet,(224,224),pre_process_resnet,image_paths)
predicted_labels_inception,confidences_inception = processImages(model_incpetion,(224,224),pre_process_inception,image_paths)
