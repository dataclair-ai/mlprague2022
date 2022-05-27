import os
import numpy as np
import pandas as pd

from utils import convert_to_13

# for interactive use
# from src.etl.utils import convert_to_13

RAW_FOLDER = "data/0_raw/"
INTR_FOLDER = "data/1_intermediate/"
PRIM_FOLDER = "data/2_primary/"

os.makedirs(INTR_FOLDER, exist_ok=True)
os.makedirs(PRIM_FOLDER, exist_ok=True)

# Best Books: list of books -------------------------------------------------------------------------------------------------------------------------

best_books = pd.read_csv(f"{RAW_FOLDER}books_1.Best_Books_Ever.csv")
best_books = best_books.assign(tags=best_books.genres.apply(eval))
best_books.isbn = best_books.isbn.apply(convert_to_13)

best_books.title = best_books.title.str.lower()

# there are 4720 non-ISBN IDs - likely ASIN
# we keep them in the dataset for now
isbn_nchar = best_books.isbn.apply(len)
isbn_nchar.unique()
best_books.isbn[isbn_nchar != 13]
del isbn_nchar

# Review book price
# -------------------------------------------------------------------------------------------------------------------------
num_dots = best_books.price.str.count("\.")
best_books.price = np.where(
    num_dots > 1, best_books.price.str.replace("\.", "", n=1), best_books.price
).astype("float")

# about 30 % of prices is NA
best_books.price.isna().sum() / best_books.shape[0]

# remove the outliers
best_books.price.describe()
best_books.price.where(best_books.price <= 200, pd.NA, inplace=True)

# Prepare title-tags mapping
# -------------------------------------------------------------------------------------------------------------------------
duplicated_titles = best_books.title[best_books.title.duplicated()]
(
    best_books[best_books.title.isin(duplicated_titles)]
    .sort_values("title")
    .loc[:, ["bookId", "title", "author", "genres"]]
    .iloc[:20]
)

title_tags = best_books.groupby("title", as_index=False).agg(
    {"tags": lambda x: x.explode().unique(), "price": lambda x: x.mean()}
)


# Prepare unique isbn-tags mapping
# -------------------------------------------------------------------------------------------------------------------------
duplicated_isbn = best_books.isbn[best_books.isbn.duplicated()]
duplicated_isbn.unique()
(
    best_books[best_books.isbn.isin(duplicated_isbn)]
    .sort_values("isbn")
    .loc[:, ["isbn", "title", "author", "genres"]]
    .iloc[:20]
)

isbn_tags = (
    best_books.query("isbn != '9999999999999'")
    .groupby("isbn", as_index=False)
    .agg({"tags": lambda x: x.explode().unique(), "price": lambda x: x.mean()})
)

del duplicated_titles, duplicated_isbn


# Book Crossing: list of books -------------------------------------------------------------------------------------------------------------------------

if not os.path.exists(f"{INTR_FOLDER}books.csv"):
    os.system(
        rf"""sed -e 's/\\"//g' {RAW_FOLDER}BX-Books.csv > {INTR_FOLDER}books.csv"""
    )

bx_books = pd.read_csv(f"{INTR_FOLDER}books.csv", sep=";", encoding="ISO-8859-1")
bx_books.columns = bx_books.columns.str.lower().str.replace("-", "_")

bx_books.book_title = bx_books.book_title.str.lower()

# Standardise ISBN to ISBN13
# -------------------------------------------------------------------------------------------------------------------------
isbn_nchar = bx_books.isbn.apply(len)
isbn_nchar.unique()

# one case of trailing '\t'
# bx_books.isbn[isbn_nchar == 11]
bx_books.isbn = bx_books.isbn.str.replace("\t", "")
# bx_books.isbn.apply(len).unique()

bx_books.isbn = bx_books.isbn.apply(convert_to_13)

# there are 114 non-ISBN IDs (starts with "B" and contains characters) - likely ASIN
# we keep them in the dataset for now
isbn_nchar = bx_books.isbn.apply(len)
bx_books.isbn[isbn_nchar != 13]

# Drop duplicated records
# -------------------------------------------------------------------------------------------------------------------------
# we won't be using the images or other additional info
bx_books = bx_books.loc[:, ["isbn", "book_title", "book_author"]].drop_duplicates()
duplicated_isbn = bx_books.isbn[bx_books.isbn.duplicated()]


# Merge Book Crossing and Best Books lists
#     - first using ISBN
#     - second using book titles
# -------------------------------------------------------------------------------------------------------------------------
books = bx_books.merge(isbn_tags, how="inner", on="isbn")

bx_books_t = bx_books[~bx_books.isbn.isin(books.isbn)]
books_t = bx_books_t.merge(
    title_tags, how="inner", left_on="book_title", right_on="title"
).drop("title", axis=1)

books = pd.concat([books, books_t])
books.tags = books.tags.apply(lambda x: ",".join([str(i) for i in x]))
books.tags = np.where(books.tags == "nan", "no_tags", books.tags)

# No NaN values
books[pd.isna(books).any(axis=1)]

books.to_csv(f"{PRIM_FOLDER}books.csv", index=False)


# Book Crossing: ratings -------------------------------------------------------------------------------------------------------------------------------

bx_rating = pd.read_csv(
    f"{RAW_FOLDER}BX-Book-Ratings.csv", sep=";", encoding="ISO-8859-1"
)
bx_rating.columns = bx_rating.columns.str.lower().str.replace("-", "_")
bx_rating.isbn = bx_rating.isbn.apply(convert_to_13)

bx_rating = bx_rating[bx_rating.isbn.isin(books.isbn)]

# No NaN values
bx_rating[pd.isna(bx_rating).any(axis=1)]

# integer values between 0 and 10 (incl)
bx_rating.describe()
bx_rating.book_rating.unique()

bx_rating.to_csv(f"{PRIM_FOLDER}ratings.csv", index=False)


# Book Crossing: users ---------------------------------------------------------------------------------------------------------------------------------
bx_users = pd.read_csv(f"{RAW_FOLDER}BX-Users.csv", sep=";", encoding="ISO-8859-1")
bx_users.columns = bx_users.columns.str.lower().str.replace("-", "_")

# 110/280 of the record contain NaN values
#     - all are in age column
bx_users[pd.isna(bx_users).any(axis=1)]

# we have some very young (0) and very old (244) users
bx_users.describe()

bx_users.to_csv(f"{PRIM_FOLDER}users.csv", index=False)
