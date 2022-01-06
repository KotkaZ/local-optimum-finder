import cv2 as cv
import numpy as np
import os

from keras.layers import MaxPooling2D, BatchNormalization, Dropout
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "processed/marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "processed/unmarked/"
BUILD_COORDINATES_FILE = BUILD_DIR + "processed/coordinates.txt"
BUILD_MODEL = BUILD_DIR + "model.something"
DEBUG_NR_PICTURES = 10

#############################
# Trainer
#############################

def load_from_dir(path:str)-> list:
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
        norm_img = np.zeros(shape = (130,210,1))
        for ri, row in enumerate(img):
            for pi, px in enumerate(row):
                norm_img[ri][pi][0] = px[0]/255
        data.append(norm_img)

    return np.array(data)

def get_train_data():
    if not os.path.isfile(BUILD_DIR+"coordinates.txt"):
        print("Error: {} does not exist".format("coordinates.txt"))
        raise ValueError
    #marked_imgs = load_from_dir(BUILD_DIR_MARKED, ".png")
    unmarked_imgs = load_from_dir(BUILD_DIR_UNMARKED)
    coordinates = []
    with open(BUILD_COORDINATES_FILE, "r", encoding="utf-8") as coord_file:
        lines = coord_file.readlines()[:DEBUG_NR_PICTURES]
        for line in lines:
            splitted_line = line.strip()[2:-2].replace(" ", "").split("),(")
            locs = []
            for x in splitted_line:
                locs.append(tuple(int(y) for y in x.split(",")))

            data = [0 for el in range(210*130)]
            for loc in locs:
                data[loc[0]*loc[1]]= 1
            coordinates.append(data)

    return (unmarked_imgs, coordinates)

def train_model(X_train, y_train):
    input_shape = X_train[0].shape
    x = Input(shape=input_shape)
    c1 = Conv2D(input_shape[0], (3, 3), strides=1, padding="same")(x)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    c2 = Conv2D(input_shape[0], (3, 3), strides=1, padding="valid")(a1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    p2 = MaxPooling2D(pool_size=(4,4))(a2)
    d2 = Dropout(0.25)(p2)
    f2 = Flatten()(d2)
    #h3 = Dense(input_shape[0] * input_shape[1],activation='softmax')(f2)
    """ b3 = BatchNormalization()(h3)
    a3 = Activation('relu')(b3)
    d3 = Dropout(0.25)(a3)
    z = Dense(input_shape[0] + input_shape[1],activation='softmax')(d3) """

    model = Model(inputs=x, outputs=f2)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, batch=256, epochs=1)

    return model, history


def save_model(model):
    model = None

def train():
    X_train, y_train = get_train_data()
    model, history = train_model(X_train, y_train)
    return model, history


model, history = train()
model, history
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
