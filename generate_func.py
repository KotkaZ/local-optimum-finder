import urllib.parse
import urllib.request
import time
import os
import random as rn
import re

from PIL import Image
from pytesseract import pytesseract

import selenium.webdriver
from selenium import webdriver


WOLFRAM_URL = "https://www.wolframalpha.com/"
INPUT_TYPE = "input/?i="
INPUT_TYPE_EXTREMA = INPUT_TYPE+"local+extrema+"
INPUT_TYPE_PLOT = INPUT_TYPE+"plot+"

# Download additional drivers if needed.
DRIVER = "./drivers/chrome_96/chromedriver.exe"

PATH_TO_TESSERACT = "../PYTESSERACT/tesseract.exe" # TODO to conf file
OUTPUT_DIR = "./build/test_funcs/"
DST_FUNCTIONS_FILE = OUTPUT_DIR + "test_functions.txt"

NR_OF_FUNS_TO_GEN = 5 # TODO to conf file

def build_url(url: str, is_marked: bool) -> str:
    return WOLFRAM_URL + (INPUT_TYPE_EXTREMA if is_marked else INPUT_TYPE_PLOT) + urllib.parse.quote_plus(url)


def get_output_path(index: int, is_marked: bool) -> str:
    # Create output directory if not exists.
    DIR = OUTPUT_DIR + ("marked/" if is_marked else "unmarked/")
    if not os.path.isdir(DIR):
        os.makedirs(DIR)

    return DIR + str(index) + ".png"


def populate_chrome() -> selenium.webdriver:
    # Start driver in headless mode.
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.headless = True

    return webdriver.Chrome(executable_path=DRIVER, options=chrome_options)


def find_plot_img(driver: selenium.webdriver) -> str:
    try:
        img = driver.find_element_by_xpath('//img[@alt="Plot"]')
    except:
        img = driver.find_element_by_xpath('//img[@alt="Plots"]')
    return img.get_attribute('src')


def download_image(driver: selenium.webdriver, index: int, is_marked: bool) -> None:
    try:
        img_url = find_plot_img(driver)
        urllib.request.urlretrieve(img_url, get_output_path(index,is_marked))
    except:
        print("Didn't find image for", index)

# Generates random polynomial function.
# Higest degree in function could be 10
def generate_function(seed:int) -> str:
    rn.seed(seed)
    nr_of_terms = rn.randint(1,10)
    highest_degree = rn.randint(0,10)
    function = ""
    for _ in range(nr_of_terms):
        term = "+" if rn.random() < 0.5 else "-"
        term += str(rn.randint(1,10))
        if highest_degree != 0:
            term += "x^"
            term += (("" if highest_degree > 0 else "(") + str(highest_degree) + ("" if highest_degree > 0 else ")"))
        function += term
        highest_degree -= rn.randint(0, int(20/nr_of_terms))

    return function

def download_graph(url: str,i: int, is_marked: bool)->str:
    chrome = populate_chrome()
    chrome.get(url)

    time.sleep(5)

    download_image(driver=chrome, index=i, is_marked=is_marked)
    return ""


# Create output directory if not exists.
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

pytesseract.tesseract_cmd = PATH_TO_TESSERACT
with open(DST_FUNCTIONS_FILE, mode="w") as dst_file:
    for i in range(NR_OF_FUNS_TO_GEN):
        function = generate_function(i)
        dst_file.write(function+"\n")

        # Find graph with marked local extremas if exists
        url = build_url(function, True)
        download_graph(url, i, True)
        try:
            img = Image.open(get_output_path(i,True))
            big_img = img.resize((1000, int(1000/img.size[0]*img.size[1]))) # To increase text recognition accuarcy
            text = pytesseract.image_to_string(big_img)
            text = re.search(r"\(x from -?\d+(.\d+)? to -?\d+(.\d+)?\)", text).group() + " " #(x from <float> to <float>)
        except:
            text = ""

        # Find graph without marked local extremas
        url = build_url(text+function, False)
        download_graph(url, i, False)

