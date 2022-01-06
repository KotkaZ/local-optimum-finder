import ast

import cv2 as cv
import numpy as np
import os

from numpy import ndarray
from sklearn.linear_model import *

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "processed/marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "processed/unmarked/"
BUILD_COORDINATES_FILE = BUILD_DIR + "processed/coordinates.txt"
BUILD_MODEL = BUILD_DIR + "model.something"
DEBUG_NR_PICTURES = 100

#############################
# Trainer
#############################


def load_image(path: str) -> float:
    img = cv.imread(path)

    # Normaliseerime pildi
    # norm_img = np.zeros(shape = (130,210,1))
    norm_img = [px[0] * 1000000000 + px[1] * 1000000 + px[2]* 1000 + px[3] for px in [row for row in img]]
    return norm_img


def load_from_dir(path: str) -> ndarray:
    if not os.path.isdir(path):
        print("Error: {} does not exist".format(path))
        raise ValueError
    data = []
    for idx in range(DEBUG_NR_PICTURES):#len(os.listdir(path))):
        fpath = path + str(idx) + ".png"
        img = cv.imread(fpath)
        if img is None:
            #data.append(None)
            print("There is no {}".format(path))
            continue
        # Normaliseerime pildi
        #norm_img = np.zeros(shape = (130,210,1))
        norm_img = [px[0] * 1000000000 + px[1] * 1000000 + px[2]* 1000 + px[3] for px in [row for row in img]]
        data.append(norm_img)
    return data


def get_train_data():
    if not os.path.isfile(BUILD_DIR+"coordinates.txt"):
        print("Error: {} does not exist".format("coordinates.txt"))
        raise ValueError

    unmarked_imgs = load_from_dir(BUILD_DIR_UNMARKED)
    coordinates = []
    with open(BUILD_COORDINATES_FILE, "r", encoding="utf-8") as coord_file:
        for line in coord_file:
            array = ast.literal_eval(line)
            coordinates.append(0 if len(array) == 0 else array[0][0])

    return np.array(unmarked_imgs), np.array(coordinates[:DEBUG_NR_PICTURES])


def train_model(X_train, y_train):
    nsamples, nx, ny = X_train.shape
    d2_train_dataset = X_train.reshape((nsamples, nx * ny))

    linearModel = LinearRegression()
    linearModel.fit(d2_train_dataset, y_train)

    return linearModel


def train():
    X_train, y_train = get_train_data()
    linearModel = train_model(X_train, y_train)
    return linearModel


model = train()
#############################
# Predicter
#############################

test = np.array([load_image(BUILD_DIR_UNMARKED + "547.png")])
nsamples, nx, ny = test.shape
d2_train_dataset = test.reshape((nsamples, nx * ny))

print(model.predict(d2_train_dataset))


def model_load():
    model = None
    return model


def model_predict(model):
    locs = []
    return []


# Adds red dots to the locations, and shows the final answer
def show_answer(img, locs):
    return


def predict(img, model):
    locs = model_predict(model, img)
    show_answer(img, locs)
