import cv2 as cv
import numpy as np
import os
import random as rn
from PIL import Image

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "unmarked/"
BUILD_COORDINATES_FILE = BUILD_DIR_MARKED + "coordinates.txt"
BUILD_MODEL = BUILD_DIR + "model.something"


#############################
# Trainer
#############################

def load_from_dir(path:str, file_type:str)-> list:
    if not os.path.isdir(path):
        print("Error: {} does not exist".format(path))
        raise ValueError
    data = []
    for idx in range(len(os.listdir(path))):
        fpath = path + str(idx) + file_type
        img = cv.imread(fpath)
        if img is None:
            data.append(None)
            continue
        data.append(np.array(img))
    return data

def get_train_data():
    if not os.path.isfile(BUILD_DIR+"coordinates.txt"):
        print("Error: {} does not exist".format("coordinates.txt"))
        raise ValueError

    marked_imgs = load_from_dir(BUILD_DIR_MARKED, ".png")
    unmarked_imgs = load_from_dir(BUILD_DIR_UNMARKED, ".jpg")
    coordinates = []
    with open(BUILD_COORDINATES_FILE, "r", encoding="utf-8") as coord_file:
        lines = coord_file.readlines()
        for line in lines:
            splitted_line = line.strip()[2:-2].replace(" ", "").split("),(")
            locs = []
            for x in splitted_line:
                locs.append(tuple(int(y) for y in x.split(",")))

            coordinates.append(locs)

    return (marked_imgs, unmarked_imgs, coordinates)

get_train_data()

def train_model():
    model = None
    return model

def save_model(model):
    model = None

def train():
    get_train_data()
    model = train_model()
    return model

#############################
# Predicter
#############################


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
