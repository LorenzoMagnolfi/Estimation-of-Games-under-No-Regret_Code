import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
import pandas as pd
from datetime import datetime, timedelta 
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
import re
import os

# Change the current working directory
os.chdir("D:/Dropbox/RA_with_Lorenzo/scrape")  # Change the current working directory

# Initialize the Chrome WebDriver
driver_path = 'D:/chromedriver-win32/chromedriver.exe'  # Path to the Chrome WebDriver
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# Initialize the DataFrame
df = pd.DataFrame(columns=['Device', 'Storage', 'Color', 'Condition', 'Price'])

# Read the links from the CSV file
Links = pd.read_csv('gazelle_device_links.csv')

# Function to re-locate element safely
def find_element(driver, by, value):
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
    return WebDriverWait(driver, 10, ignored_exceptions=ignored_exceptions).until(
        EC.presence_of_element_located((by, value))
    )

try:
    for url in Links['Link']:
        driver.get(url)
        try:
            # Extract the device's title
            title_element = find_element(driver, By.CSS_SELECTOR, "div.product__title h1")
            device = title_element.text.strip()
            print(f"Device Title: {device}")

            # Extract color options
            color_inputs = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input.gz-swatch-input__input"))
            )
            print(f"Found {len(color_inputs)} color options")

            # Extract storage options
            storage_inputs = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "fieldset[name='storage'] input"))
            )
            print(f"Found {len(storage_inputs)} storage options")

            for storage_index in range(len(storage_inputs)):
                attempts_storage = 0
                while attempts_storage < 3:
                    try:
                        # Re-locate the storage input elements after each iteration
                        storage_inputs = driver.find_elements(By.CSS_SELECTOR, "fieldset[name='storage'] input")
                        driver.execute_script("arguments[0].click();", storage_inputs[storage_index])
                        storage = storage_inputs[storage_index].get_attribute("value")
                        print(f"Selected storage: {storage}")

                        for color_index in range(len(color_inputs)):
                            attempts_color = 0
                            while attempts_color < 3:
                                try:
                                    # Re-locate the color input elements after each iteration
                                    color_inputs = driver.find_elements(By.CSS_SELECTOR, "input.gz-swatch-input__input")
                                    driver.execute_script("arguments[0].click();", color_inputs[color_index])
                                    color = color_inputs[color_index].get_attribute("value")
                                    print(f"Selected color: {color}")

                                    # Wait for the conditions to load
                                    condition_elements = find_element(driver, By.CSS_SELECTOR, "fieldset[name='Cosmetic Condition'] label")
                                    condition_elements = driver.find_elements(By.CSS_SELECTOR, "fieldset[name='Cosmetic Condition'] label")

                                    for condition_element in condition_elements:
                                        condition = condition_element.get_attribute("for")
                                        condition_input = find_element(driver, By.CSS_SELECTOR, f"input[id='{condition}']")
                                        condition_value = condition_input.get_attribute("value")
                                        price = condition_element.find_element(By.CSS_SELECTOR, "span").text
                                        price = price.replace("$", "").strip()  # Remove the dollar sign and strip whitespace
                                        print(f"Condition: {condition_value}, Price: {price}")
                                        df.loc[len(df)] = [device, storage, color, condition_value, price]
                                    break
                                except StaleElementReferenceException:
                                    attempts_color += 1
                                    print(f"Stale element reference encountered for color. Retrying... ({attempts_color}/3)")
                            if attempts_color >= 3:
                                print(f"Failed to process color index {color_index} after 3 attempts")
                        break
                    except StaleElementReferenceException:
                        attempts_storage += 1
                        print(f"Stale element reference encountered for storage. Retrying... ({attempts_storage}/3)")
                if attempts_storage >= 3:
                    print(f"Failed to process storage index {storage_index} after 3 attempts")

        except TimeoutException:
            print(f"TimeoutException occurred while processing URL: {url}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    driver.quit()


# save the .cvs file
# sort the data by device, storage, condition, and price
df1 = df.sort_values(by=['Device', 'Storage', 'Condition', 'Color', 'Price']).reset_index(drop=True)

# Add the "Carrier" column with values equal to "Unlocked"
df1['Carrier'] = 'Unlocked'

# Function to remove the suffix '(unlocked)' and storage from the 'Device' column
def clean_device_name(device_name):
    # Remove '(Unlocked)' suffix
    device_name = re.sub(r' \(Unlocked\)', '', device_name)
    # Remove storage information (e.g., '64GB' or '1TB')
    device_name = re.sub(r' \d+(GB|TB)', '', device_name)
    return device_name

# Apply the function to the 'Device' column
df1['Device'] = df1['Device'].apply(clean_device_name)

# Display the cleaned DataFrame
print(df1)

# Reorder the columns as specified: Device, Storage, Condition, Color, Price, Carrier
df1 = df1[['Device', 'Storage', 'Condition', 'Color', 'Carrier', 'Price']]

# Add the date and save the .csv file
df1['date'] = datetime.now().strftime("%Y-%m-%d")
filename = 'gazelle_sell_prices_' + datetime.now().strftime("%y%m%d") + '.csv'
path = 'D:/Dropbox/RA_with_Lorenzo/scrape/Gazelle/sell_prices/'
df1.to_csv(path + filename, index=False)