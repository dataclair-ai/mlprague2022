import requests
import os
import zipfile

FOLDER = "data/0_raw/"
os.makedirs(FOLDER, exist_ok=True)

# Book-crossing dataset
# Set of files decribing the Book Crossing books (title, author, isbn, ...), users (age, location) and their ratings
url = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"

req = requests.get(url)

filename = url.split("/")[-1]
filename = f"{FOLDER}{filename}"

with open(filename, "wb") as output_file:
    output_file.write(req.content)

file_zip = zipfile.ZipFile(filename)
file_zip.extractall(FOLDER)

# Best books ever
# https://zenodo.org/record/4265096
# 52478 records of books on the "GoodReads Best Books Ever" list
url = "https://zenodo.org/record/4265096/files/books_1.Best_Books_Ever.csv?download=1"

req = requests.get(url)

filename = url.split("/")[-1].split("?")[0]
filename = f"{FOLDER}{filename}"

with open(filename, "wb") as output_file:
    output_file.write(req.content)
