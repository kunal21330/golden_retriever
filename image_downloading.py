import pandas as pd
import requests
import os


csv_file = 'A2_Data.csv'
image_folder = 'images'

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

df = pd.read_csv(csv_file)

for index, row in df.iterrows():
    image_id = row['id']
    image_urls = eval(row['Image'])  # Convert string representation of list to actual list
    for i, image_url in enumerate(image_urls):
        image_path = os.path.join(image_folder, f'{image_id}_{i}.jpg')  # Include index in filename
        try:
            # Download image from URL
            response = requests.get(image_url)
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Image {image_id}_{i} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading image {image_id}_{i}: {e}")

