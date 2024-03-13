import pandas as pd
import requests
import os


csv_file = 'A2_Data.csv'
text_folder = 'Q2/text_files'

if not os.path.exists(text_folder):
    os.makedirs(text_folder)

df = pd.read_csv(csv_file)

for index, row in df.iterrows():
    text_id = row['id']
    text_content = row['Review Text']  
    text_path = os.path.join(text_folder, f'{text_id}.txt')
    try:
        # Write text content to a .txt file
        with open(text_path, 'w') as f:
            f.write(text_content)
        print(f"Text file {text_id}.txt created successfully.")
    except Exception as e:
        print(f"Error creating text file {text_id}.txt: {e}")
  