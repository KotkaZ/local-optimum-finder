import cv2 as cv
import numpy as np
import os
import random as rn
from PIL import Image

BUILD_DIR = "./build/test_funcs/"
BUILD_DIR_MARKED = BUILD_DIR + "marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "unmarked/"
EXTREMUM_TEMPLATE = "./templates/red_dot.png"
DST_COORDINATES_FILE = BUILD_DIR_MARKED + "coordinates.txt"

template = cv.imread(EXTREMUM_TEMPLATE)
_, w, h = template.shape[::-1]
with open(DST_COORDINATES_FILE, "w", encoding="utf-8") as dst_f:
    for idx in range(len(os.listdir(BUILD_DIR_MARKED))):
        path = BUILD_DIR_MARKED + str(idx) + ".gif"


        plot_img = cv.imread(path[:-3]+"png")
        if plot_img is None:
            continue

        res = cv.matchTemplate(plot_img, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)

        # salvestame koordinaadid
        dst_f.write(str( list(zip(*loc[::-1]))) + "\n")
        img_rgb_new = plot_img.copy()


        # Kaotame punkti
        for pt in zip(*loc[::-1]):
            for i in range(8):
                for j in range(8):
                    px = [255 - rn.randint(0,2) for x in range(3)]
                    if 3 <= i and i <= 5:
                        px = [185,115,130] # joone värv
                        if i != 4:
                            px = [255 - int((255-x) / 10) for x in px]
                        # Lisame müra
                        px = [x + rn.randint(-7,7) for x in px]
                    img_rgb_new[pt[1]-2+i][pt[0]-2+j]= px
            # rectangle(pilt, algusnurk (koordinaadid), lõppnurk, värv (BGR), joone paksus)
            #cv.rectangle(img_rgb_new, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)

        cv.imwrite(BUILD_DIR_UNMARKED + str(idx)+ ".jpg", img_rgb_new)

