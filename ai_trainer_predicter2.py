import ast

import cv2 as cv
import numpy as np
import os

from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Flatten, Dense, MaxPooling2D, MaxPool2D, GlobalAveragePooling2D, Dropout, Convolution2D, BatchNormalization, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "processed/marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "processed/unmarked/"
BUILD_COORDINATES_FILE = BUILD_DIR + "processed/coordinates.txt"
BUILD_MODEL = BUILD_DIR + "model.something"
DEBUG_NR_PICTURES = 200

#############################
# Trainer
#############################

def load_from_dir(path:str)-> list:
    if not os.path.isdir(path):
        print("Error: {} does not exist".format(path))
        raise ValueError
    imgs_norm = []
    imgs = []
    for idx in range(DEBUG_NR_PICTURES):#len(os.listdir(path))):
        fpath = path + str(idx) + ".png"
        img = cv.imread(fpath)
        if img is None:
            #data.append(None)
            print("There is no {}".format(path))
            continue
        # Normaliseerime pildi
        #norm_img = np.zeros(shape = (130,210,1))
        norm_img = [[[0] for b in range (210)] for a in range(130)]
        for ri, row in enumerate(img):
            for pi, px in enumerate(row):
                norm_img[ri][pi][0] = px[0]/255
        imgs_norm.append(norm_img)
        imgs.append(img)

    return imgs_norm, imgs


def get_train_data():
    if not os.path.isfile(BUILD_DIR+"coordinates.txt"):
        print("Error: {} does not exist".format("coordinates.txt"))
        raise ValueError
    #marked_imgs = load_from_dir(BUILD_DIR_MARKED, ".png")
    unmarked_norm_imgs, unmarked_imgs = load_from_dir(BUILD_DIR_UNMARKED)
    coordinates = []
    with open(BUILD_COORDINATES_FILE, "r", encoding="utf-8") as coord_file:
        lines = coord_file.readlines()[:DEBUG_NR_PICTURES]
        for line in lines:
            locs = ast.literal_eval(line)

            data = [[[0] for _ in range (210)] for _ in range(130)]
            for loc in locs:
                data[min(129, loc[1])][min(209, loc[0])][0] = 0.999
            coordinates.append(data)

    return np.array(unmarked_norm_imgs), np.array(coordinates), np.array(unmarked_imgs)


def train_model(X_train, y_train):
    input_shape = X_train[0].shape
    x = Input(shape=input_shape)
    a1 = Conv2D(64, (3, 3), strides=1, padding="same")(x)
    a2 = BatchNormalization()(a1)
    a3 = Activation('relu')(a2)
    a4 = Conv2D(64, (3, 3), strides=1, padding="same")(a3)
    a5 = BatchNormalization()(a4)
    a6 = Activation('relu')(a5)
    a7 = Dense(4, activation='softmax')(a6)

    """
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    d2 = Dropout(0.25)(p2)
    f2 = Flatten()(d2)
    #h3 = Dense(input_shape[0] * input_shape[1],activation='softmax')(f2) """
    """ b3 = BatchNormalization()(h3)
    a3 = Activation('relu')(b3)
    d3 = Dropout(0.25)(a3)
    z = Dense(input_shape[0] + input_shape[1],activation='softmax')(d3) """

    model = Model(inputs=x, outputs=a7)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, batch_size=16, epochs=2,verbose = True)
    return model, history


def save_model(model):
    model.save(BUILD_MODEL)


X_imgs, y_locs, X_imgs_unmarked = get_train_data()
divider = int(len(X_imgs) * 0.2)
X_train_imgs, X_test_imgs = X_imgs_unmarked[:-divider], X_imgs_unmarked[-divider:]
X_train, X_test, y_train, y_test = X_imgs[:-divider], X_imgs[-divider:], y_locs[:-divider], y_locs[-divider:]
model, history = train_model(X_train, y_train)

save_model(model)

y_pred = model.predict(X_test)



final_image = X_test_imgs[0].copy()
print(y_pred[0])
print(y_pred[0].shape)


for x, row in enumerate(final_image):
    for y, pixel in enumerate(row):
        if sum(y_pred[0][x][y]) > 0.8:
            pixel[0] = 0
            pixel[1] = 0
            pixel[2] = 255



cv.imshow("test", final_image)
cv.waitKey(0)
# closing all open windows
cv.destroyAllWindows()



#############################
# Predicter
#############################


def model_load():
    model = None
    return model

def model_predict(model):

    return []

# Adds red dots to the locations, and shows the final answer
def show_answer(img, locs):
    return

def predict(img, model):
    locs = model_predict(model, img)
    show_answer(img, locs)
