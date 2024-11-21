# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:44:32 2024

@author: carlo
"""

import requests
import json
import re
import zipfile
import os

if 'chromedriver-win64\\chromedriver.exe' not in os.environ['PATH']:
    os.environ['PATH'] += ';' + r"\chromedriver-win64\chromedriver.exe"
    os.environ['PATH'] += ';' + r"\chromedriver-linux64\chromedriver"
    os.environ['PATH'] += ';' + r"\chromedriver.exe"

# selenium
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


# to run selenium in headless mode (no user interface/does not open browser)
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-gpu")
options.add_argument("--disable-features=NetworkService")
options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-features=VizDisplayCompositor")
options.add_experimental_option("excludeSwitches", ["enable-logging"])

#from win32com.client import Dispatch

def get_json_link(version = None):
    '''
    
    Get json endpoint/url for downloading updated chromedriver
    
    Parameters:
    ----------
        - version : str
            chrome browser version
    
    Returns:
    --------
        - link : str
            json endpoint/url for downloading chromedriver
    
    '''
    
    # url for downloading
    json_url = 'https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json'
    
    # json format
    download_links = requests.get(json_url)
    
    try:
        # need to convert to dictionary
        js_dict = json.loads(download_links.content)
    
    except Exception as e:
        raise e
        
    # if version is None:
    #     # get chrome version
    #     version = get_version_via_com()
    
    
    for ndx, ver in enumerate(js_dict['versions']):
        cond1 = ver['version'].split('.')[:-1] == version.split('.')[:-1]
        cond3 = int(ver['version'].split('.')[3]) > int(version.split('.')[3])
        cond2 = int(ver['version'].split('.')[0]) > int(version.split('.')[0])
        
        if cond1 and cond3:
            break
        elif cond2:
            break
        else:
            continue
    
    # extract link from win64 version
    try:
        link = js_dict['versions'][ndx-1]['downloads']['chromedriver'][-1]['url']
    
    except:
        # linux
        link = js_dict['versions'][ndx-1]['downloads']['chromedriver'][0]['url']
    
    return link
    

def download_zip(url):
    '''
    
    Parameters:
    ----------
        - url : str
            extracted url link/json endpoint from get_json_link
    
    Returns:
    -------
        None
    
    '''
    
    # GET request for download via json endpoint
    r = requests.get(url, allow_redirects = True)
    
    # open file and write chromedriver data
    with open('chromedriver.zip', 'wb') as chromedriver:
        chromedriver.write(r.content)
        
        
def extract_chromedriver():
    
    '''
    Extract chromedriver zip file contents to working directory
    
    Parameters:
    ----------
        None
    
    Returns:
    --------
        None
    '''
    
    path = 'chromedriver.zip'
    
    with zipfile.ZipFile(path) as zf:
        zf.extract(r'chromedriver-win64/chromedriver.exe')

def create_driver():
    '''
    Main function to create chromedriver
    1. Initial attempt to create chromedriver instance
    2. If fail, get chrome version, download appropriate version driver, 
        extract driver
    
    Parameters:
    ----------
        None
    
    Returns:
    -------
        - driver : selenium.webdriver.chrome.webdriver.WebDriver
    
    '''
    try:
        # initial attempt at creating chromedriver
        try:
            service = Service(r".\chromedriver-win64\chromedriver.exe")
        except:
            service = Service(r".\chromedriver.exe")
        
        driver = Chrome(service = service, 
                        options = options)
    
    except Exception as e:
        # extract exception error with chrome version
        error = e
        version = re.search('(?<=Current browser version is ).*(?= with binary path)', error.msg)
        
        try:
            
            # get download link for suitable driver version
            link = get_json_link(version[0])
            
            # download chromedriver zip file from json endpoint/url
            download_zip(link)
            
            # extract chromedriver contents
            extract_chromedriver()
            
            # reattempt to create driver
            driver = Chrome(service = service, 
                            options = options)
        
        except Exception as e:
            driver = None
            raise e
            
    finally:
        return driver
