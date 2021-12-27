import cv2 as cv
import numpy as np
import os


BUILD_DIR = "./build/"
EXTREMUM_TEMPLATE = "./templates/red_dot.png"


template = cv.imread(EXTREMUM_TEMPLATE)
w, h, _ = template.shape[::-1]


plot_img = cv.imread(BUILD_DIR + "0.png")
cv.imshow("pilt", plot_img)
# Muuda mustvalgeks

cv.waitKey(0)

# closing all open windows
cv.destroyAllWindows()

res = cv.matchTemplate(plot_img, template, cv.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)

img_rgb_new = plot_img.copy()

for pt in zip(*loc[::-1]):
    # rectangle(pilt, algusnurk (koordinaadid), lõppnurk, värv (BGR), joone paksus)
    cv.rectangle(img_rgb_new, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)

cv.imwrite(BUILD_DIR+"0.jpg", img_rgb_new)
