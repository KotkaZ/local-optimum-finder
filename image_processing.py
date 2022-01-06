import cv2 as cv
import os
import ast

BUILD_DIR = "./build/"
BUILD_DIR_MARKED = BUILD_DIR + "marked/"
BUILD_DIR_UNMARKED = BUILD_DIR + "unmarked/"

BUILD_DIR_PROCESSED = BUILD_DIR + "processed/"
BUILD_DIR_PROCESSED_MARKED = BUILD_DIR_PROCESSED + "marked/"
BUILD_DIR_PROCESSED_UNMARKED = BUILD_DIR_PROCESSED + "unmarked/"

BUILD_COORDINATES_FILE = BUILD_DIR + "coordinates.txt"
BUILD_COORDINATES_PROCESSED_FILE = BUILD_DIR_PROCESSED + "coordinates.txt"


def transform_images(src_dir: str, out_dir: str) -> [(float, float)]:
    rescale_factors = []
    for idx in range(len(os.listdir(src_dir))):
        path = src_dir + str(idx) + ".png"

        img = cv.imread(path)
        if img is None:
            continue
        rescale_factors.append((img.shape[0] / 210, img.shape[1] / 130))
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        (thresh, img_black_white) = cv.threshold(img_grey, 127, 255, cv.THRESH_BINARY)
        crop_img = cv.resize(img_black_white, dsize=(210,130), interpolation=cv.INTER_AREA)

        crop_img = crop_img[:,:-60].copy()
        new_path = out_dir + str(idx) + ".png"
        cv.imwrite(new_path, crop_img)
    return rescale_factors


def transform_coordinates(src_dir: str, out_dir: str, rescale_factors: [(float, float)]) -> None:
    with open(out_dir, "w", encoding="utf-8") as dst_f:
        with open(src_dir, encoding="utf-8") as src_f:
            for index, line in enumerate(src_f):
                array = ast.literal_eval(line)
                new_array = []
                rescale_factor = rescale_factors[index]
                for vector in array:
                    new_array.append((round(vector[0] / rescale_factor[0]), round(vector[1] / rescale_factor[1])))
                dst_f.write(str(new_array) + "\n")


if not os.path.isdir(BUILD_DIR_PROCESSED_UNMARKED):
    os.makedirs(BUILD_DIR_PROCESSED_UNMARKED)
if not os.path.isdir(BUILD_DIR_PROCESSED_MARKED):
    os.makedirs(BUILD_DIR_PROCESSED_MARKED)

factors = transform_images(BUILD_DIR_MARKED, BUILD_DIR_PROCESSED_MARKED)
transform_images(BUILD_DIR_UNMARKED, BUILD_DIR_PROCESSED_UNMARKED)
transform_coordinates(BUILD_COORDINATES_FILE, BUILD_COORDINATES_PROCESSED_FILE, factors)