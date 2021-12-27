import urllib.parse
import urllib.request
import time
import os

import selenium.webdriver
from selenium import webdriver


WOLFRAM_URL = "https://www.wolframalpha.com/"
INPUT_TYPE = "input/?i=local+extrema+"

# Download additional drivers if needed.
DRIVER = "./drivers/chrome_96/chromedriver.exe"

FUNCTIONS_FILE = "functions.txt"
OUTPUT_DIR = "./downloads/"


def build_url(url: str) -> str:
    return WOLFRAM_URL + INPUT_TYPE + urllib.parse.quote_plus(url)


def get_output_path(index: int) -> str:
    # Create output directory if not exists.
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    return OUTPUT_DIR + str(index) + ".jpg"


def populate_chrome() -> selenium.webdriver:
    # Start driver in headless mode.
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.headless = True

    return webdriver.Chrome(executable_path=DRIVER, options=chrome_options)


def find_plot_img(driver: selenium.webdriver) -> str:
    img = driver.find_element_by_xpath('//img[@alt="Plot"]')
    return img.get_attribute('src')


with open(FUNCTIONS_FILE) as functions:
    for index, function in enumerate(functions):
        url = build_url(function)

        chrome = populate_chrome()
        chrome.get(url)

        time.sleep(5)

        img_url = find_plot_img(chrome)

        # Download image
        urllib.request.urlretrieve(img_url, get_output_path(index))

