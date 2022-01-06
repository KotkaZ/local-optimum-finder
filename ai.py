from image_processing import transform_images
from ai_trainer_predicter import train, load_from_dir, predict
import sys
import cv2 as cv
import os
import numpy as np

BUILD_DIR__USER = "./build/user/"
IMG_PATH = sys.argv[1]
DO_TRAIN = sys.argv[2] == "True"
print(DO_TRAIN)

if DO_TRAIN:
    train()

if not os.path.isdir(BUILD_DIR__USER):
    os.makedirs(BUILD_DIR__USER)

path = IMG_PATH

img = cv.imread(path)
if img is None:
    raise ValueError

img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
crop_img = cv.resize(img_grey, dsize=(270, 130), interpolation=cv.INTER_AREA)

crop_img = crop_img[:,:-60].copy()
cv.imwrite(BUILD_DIR__USER+"user.png", crop_img)

imgs_norm = []
imgs = []

img = cv.imread(BUILD_DIR__USER+"user.png")

# Normaliseerime pildi
norm_img = np.zeros(shape = (130,210,1))
for ri, row in enumerate(img):
    for pi, px in enumerate(row):
        norm_img[ri][pi][0] = px[0]/255

imgs_norm.append(norm_img)
imgs.append(img)
imgs_norm = np.array(imgs_norm)
imgs = np.array(imgs)

predict(imgs_norm,imgs,True)



