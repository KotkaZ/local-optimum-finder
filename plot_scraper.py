import urllib.parse
import requests
import time
import os
import selenium.webdriver

from PIL import Image
from io import BytesIO
from selenium import webdriver


WOLFRAM_URL = "https://www.wolframalpha.com/"
INPUT_TYPE = "input/?i=local+extrema+"

# Download additional drivers if needed.
DRIVER = "./drivers/chrome_96/chromedriver.exe"

BUILD_DIR = "./build/"
FUNCTIONS_FILE = BUILD_DIR + "functions.txt"
FUNCTIONS_MARKED = BUILD_DIR + "marked/"

DOWNLOAD_TIMEOUT = 5


def build_url(url: str) -> str:
    return WOLFRAM_URL + INPUT_TYPE + urllib.parse.quote_plus(url)


def get_output_path(index: int) -> str:
    return FUNCTIONS_MARKED + str(index) + ".png"


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


def download_image(driver: selenium.webdriver, index: int) -> bool:
    try:
        img_url = find_plot_img(driver)
        path = get_output_path(index)
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img.save(path, 'png', optimize=True, quality=80)
        return True
    except:
        return False


# Create output directory if not exists.
if not os.path.isdir(BUILD_DIR):
    os.makedirs(BUILD_DIR)
if not os.path.isdir(FUNCTIONS_MARKED):
    os.makedirs(FUNCTIONS_MARKED)

with open(FUNCTIONS_FILE) as functions:
    chrome = populate_chrome()

    i = 0
    for function in functions:
        url = build_url(function)

        chrome.get(url)

        time_spent = 0
        success = False
        while time_spent < DOWNLOAD_TIMEOUT:
            time.sleep(0.5)
            time_spent += 0.5
            success = download_image(driver=chrome, index=i)
            if success:
                i += 1
                break
        if not success:
            print("Didn't find picture for " + function)
