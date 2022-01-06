import cv2 as cv
import os
import random as rn
import ast

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "unmarked/"
EXTREMUM_TEMPLATE = "./templates/red_dot.png"
COORDINATES_FILE = BUILD_DIR + "coordinates.txt"


if not os.path.isdir(BUILD_DIR_UNMARKED):
    os.makedirs(BUILD_DIR_UNMARKED)

template = cv.imread(EXTREMUM_TEMPLATE)
_, w, h = template.shape[::-1]

with open(COORDINATES_FILE, encoding="utf-8") as f:
    for idx, coordinates in enumerate(f):
        path = BUILD_DIR_MARKED + str(idx) + ".png"

        plot_img = cv.imread(path)

        # Kaotame punkti
        for pt in ast.literal_eval(coordinates):
            for i in range(8):
                for j in range(8):
                    px = [255 - rn.randint(0,2) for x in range(3)]
                    if 3 <= i and i <= 5:
                        px = [185,115,130] # joone värv
                        if i != 4:
                            px = [255 - int((255-x) / 10) for x in px]
                        # Lisame müra
                        px = [x + rn.randint(-7,7) for x in px]
                    plot_img[pt[1]-2+i][pt[0]-2+j] = px
            # rectangle(pilt, algusnurk (koordinaadid), lõppnurk, värv (BGR), joone paksus)
            #cv.rectangle(img_rgb_new, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)

        cv.imwrite(BUILD_DIR_UNMARKED + str(idx)+ ".png", plot_img)

