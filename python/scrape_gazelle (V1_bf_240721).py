import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
import pandas as pd
from datetime import datetime

# Change the current working directory
os.chdir("Replication Package/")  # Change the current working directory

# Initialize the Chrome WebDriver
driver_path = 'D:/chromedriver-win32/chromedriver.exe'  # Path to the Chrome WebDriver
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# Initialize the DataFrame
df = pd.DataFrame(columns=['Device', 'Storage', 'Condition', 'Price'])

Links = pd.read_csv('gazelle_links.csv')

try:
    for url in Links['Link']:
        driver.get(url)

        # Extract the device's name
        device = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.product-title h1"))).text

        # Extract the storage
        storage = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.non-selectable"))).text.split(", ")[1]
        storage = storage.replace(",", "")  # Remove the comma

        # Extract the conditions and prices
        conditions_elements = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "label[data-value]")))
        for condition_element in conditions_elements:
            condition = condition_element.get_attribute("data-value")
            price = condition_element.find_element(By.CSS_SELECTOR, "span.price.discounted-price").text
            price = price.replace("$", "")  # Remove the dollar sign
            df.loc[len(df)] = [device, storage, condition, price]
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    driver.quit()

print(df)
# save the .cvs file
# sort the data by device, storage, condition, and price
df = df.sort_values(by=['Device', 'Storage', 'Condition', 'Price']).reset_index(drop=True)

# Add the date
df['date'] = datetime.now().strftime("%Y-%m-%d")
filename = 'gazelle_' + datetime.now().strftime("%y%m%d") + '.csv'
path = 'Replication Package/Gazelle/'
df.to_csv(path + filename, index=False)