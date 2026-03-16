import requests
import pandas
import datetime
import os

os.chdir("C:/Users/DELL/Desktop/No Regret/replication codes")

# Get the current date
date = str(datetime.datetime.now()).split()[0]
d = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
}

url = "https://search-backend.eus.live.channels.em-infra.com/api//browse"

# Fetch data from multiple pages
page = 0
while True:
    print('----------current page is {}----------'.format(page+1))
    # Set the parameters for the API request
    params = {
        "MaxResults": "32",
        "TaxonPermalink": "category/cell-phones/apple",
        "Offset": str(page*32),
        "UseRedirects": "true"
    }
    # Send a GET request to the API endpoint
    response = requests.get(url, headers=headers, params=params)
    if len(response.json()['docs']) == 0:
        break
    # Iterate over the items in the response
    for i in response.json()['docs']:
        item = {}
        item['name'] = i['name']
        item['price'] = i['variants'][0]['price']
        item['date'] = str(date)
        for j in i['product_properties']:
            if j['property_name'] == 'whatsinbox':
                continue
            item[j['property_name']] = j['value']
        print(item)
        d.append(item)
    page += 1
filename_date = 'data_' + datetime.now().strftime("%m%d") + '.xlsx'
pandas.DataFrame(d).to_excel('Data/Decluttr_Daily_Data'+filename_date, index=None)