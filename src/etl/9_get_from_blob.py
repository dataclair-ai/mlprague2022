import requests
import os
import zipfile

FOLDER = "data/"
os.makedirs(FOLDER, exist_ok=True)

url = "https://staicentreprod001.blob.core.windows.net/share/mlprague2022/4_dataset.zip"

req = requests.get(url)

filename = f"{FOLDER}4_dataset.zip"

with open(filename, "wb") as output_file:
    output_file.write(req.content)

file_zip = zipfile.ZipFile(filename)
file_zip.extractall(FOLDER)
