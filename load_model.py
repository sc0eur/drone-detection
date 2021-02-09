import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.utils import *

from PIL import Image
import requests
from io import BytesIO
import os
import random
import pickle
import tqdm
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
x = vgg_conv.output
x = GlobalAveragePooling2D()(x)
# x_2 = Dense(2, activation="softmax")(x)
x_6 = Dense(6, activation="softmax")(x)
# x_5 = Dense(5, activation="softmax")(x)
x_2 = Dense(2, activation="softmax")(x)

model = Model(vgg_conv.input, x_2)
model_6 = Model(vgg_conv.input, x_6)
# model_station = Model(vgg_conv.input, x_2)
# model_trans = Model(vgg_conv.input, x_2)
# model_trash = Model(vgg_conv.input, x_2)
# model_holes = Model(vgg_conv.input, x_2)

# model_station.load_weights("./models/model.h5")
# model_trans.load_weights("./models/model_trans.h5")
# # model_trash.load_weights("./models/model.h5")
# model_holes.load_weights("./models/model_holes.h5")
# model_2 = Model(vgg_conv.input, x_2)
# model_2 = Model(vgg_conv.input, x_2)
# model_2.load_weights("./models/model_stations_with_full.h5")
# model.load_weights("./models/model_classes_weights_25ep.h5")
# model_2.load_weights("./models/model.h5")
# model.load_weights("./models/model_6_with_full.h5")
