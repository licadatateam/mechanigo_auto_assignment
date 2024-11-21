# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:12:58 2024

@author: carlo
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import get_chromedriver
import regex as re
import pandas as pd

class Geocoder:
    """
    A class to geocode addresses using Google Maps via Selenium WebDriver.
    
    Methods:
    --------
    - initialize_driver : Initializes the Selenium Chrome WebDriver in headless mode.
    - close_driver : Closes the WebDriver instance.
    - query_new_address : Queries Google Maps with the provided address.
    - get_lat_long_from_url : Extracts latitude and longitude from the Google Maps URL.
    - geocode : Geocodes an individual address and returns the results.
    - geocode_dataframe : Geocodes a DataFrame of addresses.
    """

    def __init__(self):
        self.driver = None
        
    def initialize_driver(self):
        """
        Initializes a headless Chrome WebDriver with appropriate options.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--incognito")
        #self.driver = webdriver.Chrome(options=chrome_options)
        self.driver = get_chromedriver.create_driver()
        
    def close_driver(self):
        """
        Closes the Chrome WebDriver if it's initialized.
        """
        if self.driver:
            self.driver.quit()
            
    def query_new_address(self, address):
        """
        Queries Google Maps for the given address and waits for the results.
        
        Args:
        -----
        address : str
            The address to search for on Google Maps.
        """
        url = "https://www.google.com/maps"
        self.driver.get(url)
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.clear()
        search_box.send_keys(address)
        search_box.send_keys(Keys.RETURN)
        WebDriverWait(self.driver, 20).until(self.url_matches_pattern)
    
    def url_matches_pattern(self, driver):
        """
        Verifies if the current URL matches the expected Google Maps pattern.
        
        Args:
        -----
        driver : WebDriver
            The Selenium WebDriver instance.

        Returns:
        --------
        bool : True if the URL contains latitude and longitude coordinates, False otherwise.
        """
        url_pattern = r'@(\d+\.\d+),(\d+\.\d+),(\d+z)'
        current_url = driver.current_url
        return re.search(url_pattern, current_url) is not None
    
    def get_lat_long_from_url(self, url):
        """
        Extracts latitude and longitude from the Google Maps URL.
        
        Args:
        -----
        url : str
            The Google Maps URL containing latitude and longitude information.

        Returns:
        --------
        tuple : Latitude and longitude as a tuple of floats.
        """
        url_pattern = r'@(\d+\.\d+),(\d+\.\d+),'
        match = re.search(url_pattern, url)
        if match:
            lat = float(match.group(1))
            long = float(match.group(2))
            return lat, long
        return None, None
    
    def geocode(self, address):
        """
        Geocodes a single address by querying Google Maps and extracting latitude and longitude.

        Args:
        -----
        address : str
            The address to geocode.

        Returns:
        --------
        tuple : Contains address, latitude, and longitude.
        """
        self.query_new_address(address)
        current_url = self.driver.current_url
        lat, long = self.get_lat_long_from_url(current_url)
        return address, lat, long
    
    def geocode_dataframe(self, df, address_col='address'):
        """
        Geocodes a DataFrame of addresses and returns the DataFrame with added latitude and longitude.

        Args:
        -----
        df : pd.DataFrame
            DataFrame containing address information.
        address_col : str, optional
            The name of the column containing the addresses to geocode (default: 'address').

        Returns:
        --------
        pd.DataFrame : The updated DataFrame with 'lat' and 'long' columns.
        """
        lat_list = []
        long_list = []

        for index, row in df.iterrows():
            address = row[address_col]
            _, lat, long = self.geocode(address)
            lat_list.append(lat)
            long_list.append(long)

        df['lat'] = lat_list
        df['long'] = long_list
        return df

"""
Usage Example:
    
geocoder = Geocoder()

# Initialize the driver
geocoder.initialize_driver()

# Sample DataFrame with addresses
df = pd.DataFrame({'address': ['123 Main St, New York, NY', '1600 Pennsylvania Ave NW, Washington, DC']})

# Geocode the addresses in the DataFrame
df = geocoder.geocode_dataframe(df)

# Close the driver
geocoder.close_driver()

# Output the DataFrame with lat/long
print(df)

"""