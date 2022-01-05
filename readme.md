# Plot scraper

Currently, plot scraper takes all functions defined in functions.txt and downloads their plot images with extremum points for WolframAlpha.
Images are downloaded into project's downloads folder and are not committed to repository.

NB! Plot scraper uses Selenium webdriver to scrape the images. Please make sure you have selected the proper driver according to you default web-browser

# Run guide

1. python generate_func.py
2. python plot_scraper.py
3. python coordinates_finder.py
4. python dot_remover.py