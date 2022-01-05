import cv2 as cv
import numpy as np
import os

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "marked/"
EXTREMUM_TEMPLATE = "./templates/red_dot.png"
COORDINATES_FILE = BUILD_DIR + "coordinates.txt"

template = cv.imread(EXTREMUM_TEMPLATE)
_, w, h = template.shape[::-1]

with open(COORDINATES_FILE, "w", encoding="utf-8") as dst_f:
    for idx in range(len(os.listdir(BUILD_DIR_MARKED))):
        path = BUILD_DIR_MARKED + str(idx) + ".png"

        plot_img = cv.imread(path)
        if plot_img is None:
            print("test")
            dst_f.write(str([]) + "\n")
            continue

        res = cv.matchTemplate(plot_img, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)

        dst_f.write(str(list(zip(*loc[::-1]))) + "\n")
        img_rgb_new = plot_img.copy()
