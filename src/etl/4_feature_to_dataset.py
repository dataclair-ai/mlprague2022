import os
import numpy as np
import pandas as pd

FEAT_FOLDER = "data/3_feature/"
DS_FOLDER = "data/4_dataset/"

os.makedirs(DS_FOLDER, exist_ok=True)

# fmt: off
user_age =        pd.read_csv(f"{FEAT_FOLDER}user_age.csv")
user_locations =  pd.read_csv(f"{FEAT_FOLDER}user_locations.csv")
user_tags =       pd.read_csv(f"{FEAT_FOLDER}user_tags.csv")

user_favourite =  pd.read_csv(f"{FEAT_FOLDER}user_favourite.csv")
book_freq =       pd.read_csv(f"{FEAT_FOLDER}book_freq.csv")
book_price =      pd.read_csv(f"{FEAT_FOLDER}book_price.csv")


# FEATURES ------------------------------------------------------------------------------------------------------------------------------------------
(
    user_age
    .merge(user_locations, how="inner", on="user_id")
    .merge(user_tags, how="inner", on="user_id")
).to_csv(f"{DS_FOLDER}user_context.csv", index=False)

# fmt: on


# LABELS --------------------------------------------------------------------------------------------------------------------------------------------
user_freq = user_favourite.merge(book_freq, how="inner", on="isbn")


# SIMPLE REWARD
# should we promote the 5 TOP RATED books?
# -------------------------------------------------------------------------------------------------------------------------
(
    user_freq.query("frequent")
    .groupby("user_id", as_index=False)
    .rating.sum()
    .assign(rating=lambda x: (x.rating > 0).astype("int"))
).to_csv(f"{DS_FOLDER}reward_simple.csv", index=False)

# REWARD PER BOOK
# which of the 5 TOP RATED books should we promote?
# -------------------------------------------------------------------------------------------------------------------------
user_freq.isbn.where(user_freq.frequent, "not_frequent", inplace=True)
user_book = user_freq.pivot_table(
    index="user_id",
    columns="isbn",
    values="rating",
    aggfunc=lambda x: int(any(x > 0)),
    fill_value=0,
)
rated = user_book.drop("not_frequent", axis=1).sum(axis=1)
user_book = user_book.assign(not_frequent=np.where(rated > 0, 0, 1))
user_book.reset_index().to_csv(f"{DS_FOLDER}reward_per_book.csv", index=False)
