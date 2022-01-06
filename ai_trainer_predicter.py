import cv2 as cv
import numpy as np
import os

from sklearn.model_selection import train_test_split

from keras.layers import MaxPooling2D, BatchNormalization, Dropout, Reshape
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "processed/marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "processed/unmarked/"
BUILD_COORDINATES_FILE = BUILD_DIR + "processed/coordinates.txt"
BUILD_MODEL = BUILD_DIR + "model.something"
BUILD_DIR_PREDICTED = BUILD_DIR +"predicted/"
DEBUG_NR_PICTURES = 10


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
            splitted_line = line.strip()[2:-2].replace(" ", "").split("),(")
            locs = []
            for x in splitted_line:
                locs.append(tuple(int(y) for y in x.split(",")))

            data = [[[0] for b in range (210)] for a in range(130)]
            for loc in locs:
                for i in range(6):
                    for j in range(6):
                        x = 3-i
                        y = 3-j
                        r = x**2 + y**2
                        val = 0
                        if(r <= 5):
                            val = 0.99
                        elif(r <= 8):
                            val= 0.85
                        elif(r <= 9):
                            val = 0.75
                        else:
                            val = 0.60
                        if not (loc[1]+x < 0 or loc[1]+x >= 130 or loc[0]+y < 0 or loc[0]+y >=210):
                            data[loc[1]+x][loc[0]+y][0] = val
                data[loc[1]][loc[0]][0]= 0.999
            coordinates.append(data)

    return (np.array(unmarked_norm_imgs), np.array(coordinates), np.array(unmarked_imgs))

def train_model(X_train, y_train):
    input_shape = X_train[0].shape
    x = Input(shape=input_shape)
    c1 = Conv2D(64, (3, 3), strides=1, padding="same")(x)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    c2 = Conv2D(64, (3, 3), strides=1, padding="same")(a1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    c3 = Conv2D(4, (3, 3), strides=1, padding="same")(a2)
    p4 = MaxPooling2D(pool_size=(2,2))(c3)
    #h4 = Dense(input_shape[0] * input_shape[1],activation='softmax')(f4)
    r4 = Reshape((130,210,1))(p4)
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

    model = Model(inputs=x, outputs=r4)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, batch_size=16, epochs=1,verbose = True)

    return model, history


def save_model(model):
    model = None


X_imgs, y_locs, X_imgs_unmarked = get_train_data()
divider = int(len(X_imgs) * 0.2)
X_train_imgs, X_test_imgs = X_imgs_unmarked[:-divider], X_imgs_unmarked[-divider:]
X_train, X_test, y_train, y_test = X_imgs[:-divider], X_imgs[-divider:], y_locs[:-divider], y_locs[-divider:]
model, history = train_model(X_train, y_train)

if not os.path.isdir(BUILD_DIR_PREDICTED):
    os.makedirs(BUILD_DIR_PREDICTED)
y_pred = model.predict(X_test)
for idx, y_img in enumerate(y_pred):
    test_img = np.zeros((130,210,3))
    for ri, row in enumerate(y_img):
        for pi, px in enumerate(row):
            val = min(int(255 * px[0]+ 178), 255)
            test_img[ri][pi][0] = val
            test_img[ri][pi][1] = val
            test_img[ri][pi][2] = val

    max_loc = np.unravel_index(y_img.argmax(),y_img.shape)
    print(max_loc)
    print(y_img.shape)
    final_image = X_test_imgs[idx].copy()

    ## add red dot
    for i in range(6):
        for j in range(6):
            x = 3-i
            y = 3-j
            r = x**2 + y**2
            val = 0
            if(r <= 5):
                val = [0,0,255]
            elif(r <= 8):
                val= [104,104,255]
            elif(r <= 9):
                val = [217,217,255]
            else:
                val = [255,255,255]
            if not (max_loc[0]+ x < 0 or max_loc[0]+ x >= 130 or max_loc[1]+y < 0 or max_loc[1]+y >=210):
                final_image[max_loc[0]+x][max_loc[1]+y] = [min(el[0],el[1]) for el in zip(final_image[max_loc[0]+x][max_loc[1]+y], val)]

    cv.imwrite(BUILD_DIR_PREDICTED+str(idx)+'.png', final_image)

    """cv.imshow("test", final_image)
    cv.waitKey(0)
    # closing all open windows
    cv.destroyAllWindows()

    cv.imwrite('color_img.jpg', test_img)
    c = cv.imread('color_img.jpg')
    cv.imshow("test", c)
    cv.waitKey(0)
    # closing all open windows
    cv.destroyAllWindows() """


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
