import pandas as pd
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
from datetime import datetime, timedelta
import os
from datetime import datetime, timedelta

# Change the current working directory
os.chdir("D:/Dropbox/RA_with_Lorenzo/scrape")  # Change the current working directory

# Initialize the Chrome WebDriver
driver_path = 'D:/chromedriver-win32/chromedriver.exe'  # Path to the Chrome WebDriver
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# Initialize the DataFrame
df = pd.DataFrame(columns=['Device', 'Storage', 'Condition', 'Price'])

# Function to find and click the "Yes" buttons safely
def click_yes_buttons(driver):
    for _ in range(3):
        button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-labelledby='Yes'].btn.btn-orange-outline:not(.selected):not([disabled])"))
        )
        driver.execute_script("arguments[0].click();", button)
        print(f"Clicked 'Yes' button")
        time.sleep(1)  # Add a short pause between clicks

# Function to extract the price with retries
def extract_price(driver, retries=3):
    for attempt in range(retries):
        try:
            price = driver.execute_script('return document.querySelector("div#amount h3.dollar-amount span[aria-labelledby=\'amount\']").textContent')
            if price:
                return price
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(2)  # Wait before retrying
    return None

# Read the Excel file containing the list of devices
devices_df = pd.read_excel('device_gazelle.xlsx')

try:
    for device in devices_df['device']:
        url = f"https://www.gazelle.com/iphone/{device}/unlocked"
        driver.get(url)
        print(f"Processing device: {device}")

        try:
            # Find all links with the class "image_link"
            links = WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "image_link"))
            )

            # Extract the href attribute from each link
            hrefs = [link.get_attribute('href') for link in links]
            
            if not hrefs:
                print(f"No storage links found for device: {device}")
                continue

            for storage_url in hrefs:
                driver.get(storage_url)
                print(f"Processing storage URL: {storage_url}")

                try:
                    # Extract the device's title and storage
                    title_element = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "headline"))
                    )
                    title = title_element.text
                    device_name = re.sub(r'\d+GB \(Unlocked\)', '', title).replace('Apple ', '').strip()
                    storage = re.search(r'\d+(GB|TB)', title).group()
                    print(f"Device: {device_name}, Storage: {storage}")

                    # Click "Yes" buttons for the first three questions
                    click_yes_buttons(driver)

                    # Get prices for different conditions
                    conditions = [
                        ("Scratched or scuffed", "Fair"),
                        ("Lightly used", "Good"),
                        ("Flawless or like new", "Excellent")
                    ]
                    for condition_text, condition_value in conditions:
                        button = WebDriverWait(driver, 15).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, f"button[aria-labelledby='{condition_text}'].btn.btn-orange-outline:not(.selected):not([disabled])"))
                        )
                        driver.execute_script("arguments[0].click();", button)
                        print(f"Clicked condition button: {condition_text}")
                        time.sleep(1)  # Add a short pause between clicks

                        # Extract the price with retries
                        price = extract_price(driver)
                        if price:
                            print(f"Extracted price: {price}")

                            # Collect data in the DataFrame
                            df.loc[len(df)] = [device_name, storage, condition_value, price]
                            print(f"Appended data: {device_name}, {storage}, {condition_value}, {price}")
                        else:
                            print(f"Failed to extract price for {device_name}, {storage}, {condition_value}")

                except Exception as e:
                    print(f"An error occurred while processing storage {storage_url} for device {device}: {e}")

        except Exception as e:
            print(f"An error occurred while processing device {device}: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    driver.quit()


# Save the .csv file
# Sort the data by device, storage, condition, and price
df2 = df.sort_values(by=['Device', 'Storage', 'Condition', 'Price']).reset_index(drop=True)

# Add the "Carrier" column with values equal to "Unlocked"
df2['Carrier'] = 'Unlocked'

# Remove " (Unlocked)" from the Device column
df2['Device'] = df2['Device'].str.replace(' \(Unlocked\)', '')  # Why is this not working? Remember to deal with it in the following analysis.

# Reorder the columns as specified: Device, Storage, Condition, Price, Carrier
df2 = df2[['Device', 'Storage', 'Condition', 'Carrier', 'Price']]

# Add the date
df2['date'] = datetime.now().strftime("%Y-%m-%d")

# Display the cleaned DataFrame
print(df2)

# Save the DataFrame to a CSV file
filename = 'gazelle_buy_prices_' + datetime.now().strftime("%y%m%d") + '.csv'
path = 'D:/Dropbox/RA_with_Lorenzo/scrape/Gazelle/buy_prices/'
df2.to_csv(path + filename, index=False)
