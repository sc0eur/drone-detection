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

from time import clock
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from load_model import model, model_6

from PIL import Image

def shred_img(path_to_pic, shape=224):
    im = Image.open(path_to_pic)
    print(im.size[0])
    check = int(im.size[0]) < 4000
    # if check:
    #     im = im.resize((shape, shape))
    #     return np.array([img_to_array(im)/255])
    columns = 4
    rows = 4
    width, height = im.size

    height_i = int(height /  rows)
    width_i = int(width / columns)

    imgs = [img_to_array(im.resize((shape, shape)))/255]

    for i in range(0, columns * rows):

        row = i // columns
        column = i % columns

        left_corner = column * width_i
        top_corner = row * height_i

        target_width = width_i * 1.5
        target_height = height_i * 1.5
        if row == rows - 1:
            top_corner = row * height_i - height_i * 0.5

        if column == columns - 1:
            left_corner = column * width_i - width_i * 0.5

        im1 = im.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
        im1 = im1.resize((shape, shape))
        imgs.append(img_to_array(im1)/255)
        # save_img(f"unsorted_holes/{j*100+i}.jpg", array_to_img(im1))
    print(np.array(imgs).shape)
    return np.array(imgs)


def generate_pic(path_to_pic):
    classes = {"station": "./models/model.h5",
                "transport": "./models/model_trans.h5",
                "trash": "./models/model_trash_no.h5",
                "holes": "./models/model_holes.h5",
                "other": "./models/model.h5",
                "station_2": "./models/model_stations_with_full.h5"
                }
    names = ["station", "holes", "trash", "transport", "other"]
    start = clock()
    base = os.path.basename(path_to_pic)
    imgs = shred_img(path_to_pic)
    # print(imgs)
    # fig, (((())))
    # fig, ax = plt.subplots(2,2, figsize=(400,400))
    for i, img in enumerate(imgs):
        # img = img.resize((640, 640))
        # img = img_to_array(img)/255
        # img = img/255
        # print(img)
        plt.figure(i)
        plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
        plt.axis('off')
        preds = []
        for name in names:
            model.load_weights(classes[name])


            pred = model.predict(img[np.newaxis,:,:,:])
            if name=="other":
                pred[0] = pred[0][::-1]
            preds.append(pred[0][1])
            # print(pred)
            # # pred = preds[i]
            # if name=="other":
            #     pred[0] = pred[0][::-1]
            # pred_class = np.argmax(pred[0])
        preds[3] -= 0.2
        if max(preds) > 0.5:
            model.load_weights(classes[names[preds.index(max(preds))]])
            weights = model.layers[-1].get_weights()[0]
            pred_class = 1

            class_weights = weights[:, pred_class]

            intermediate = Model(model.input, model.get_layer("block5_conv3").output)
            conv_output = intermediate.predict(img[np.newaxis,:,:,:])
            conv_output = np.squeeze(conv_output)

            h = int(img.shape[0]/conv_output.shape[0])
            w = int(img.shape[1]/conv_output.shape[1])

            activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
            out = np.dot(activation_maps.reshape((img.shape[0]*img.shape[1], 512)), class_weights).reshape(img.shape[0],img.shape[1])
            plt.imshow(out, cmap='jet', alpha=0.35)
            addition = ""
            if names[preds.index(max(preds))]=="station":
                model.load_weights(classes['station_2'])
                pred = model.predict(img[np.newaxis,:,:,:])[0]
                model_6.load_weights("./models/model_classes_weights_25ep.h5")
                pred = model_6.predict(img[np.newaxis,:,:,:])[0][1:3]
                pred_class = np.argmax(pred)

                stat = {0: " leak",
                        1: " no leak"}
                addition = stat[pred_class]
            plt.title(names[preds.index(max(preds))]+addition)

        else:
            plt.title("no")

        base,ext = os.path.splitext(base)
        dig = random.randint(0,1000)
        plt.savefig("./static/imgs/"+base+str(dig)+'.png')
        print(str(i)+' ', clock()-start)
        yield base+str(dig)+".png"

    # print(os.path.abspath("./static/imgs/foo.png")
    # return base+'.png'
