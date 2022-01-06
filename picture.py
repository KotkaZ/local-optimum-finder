import cv2 as cv

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "unmarked/"
BUILD_COORDINATES_FILE = BUILD_DIR_MARKED + "coordinates.txt"

"""
* leidma punkti koordinaadi ja normaliseerida 0-1ni ja punkt eemaldada
* siis muuta resize abil mingisse standard suurusesse (väiksemaks nt 130,270)
* siis teha crop (see valge ala paremal pool 270-60) ja vastavalt siis uus punkti koordinaat leida 0-1ni
"""
for idx in range(700):
    fpath = BUILD_DIR_MARKED + str(idx) + ".png"
    img = cv.imread(fpath)
    # closing all open windows
    crop_img = cv.resize(img, dsize=(270,130), interpolation=cv.INTER_AREA)
    crop_img = crop_img[:,:-60].copy()
    print(idx)
    print("idx {} image.shape == {}".format(str(idx), str(img.shape)))
    print("\timage.shape == {}".format(str(crop_img.shape)))
    cv.imshow("cropped", crop_img)
    cv.waitKey(0)
    # closing all open windows
    cv.destroyAllWindows()
"""
(257, 536, 3)
Didn't find picture for -1x^5+10+2x^(-5)+2x^(-7)

Didn't find picture for +6x^1

Didn't find picture for -4x^4+9x^1+5x^(-5) """