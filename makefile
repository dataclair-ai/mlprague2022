SHELL := /bin/bash
PREFIX := src/etl/

.PHONY: raw primary book_features user_features dataset

raw:
	# Downloading data
	python ${PREFIX}0_get_raw_data.py
	@echo

primary : raw
	# Preprocessing datafiles
	python ${PREFIX}1_raw_to_primary.py
	@echo

book_features: primary
	# Preparing book features
	jupyter nbconvert --to notebook --inplace --execute ${PREFIX}2_primary_to_book_feature.ipynb
	@echo

user_features: book_features
	# Preparing user features
	jupyter nbconvert --to notebook --inplace --execute ${PREFIX}3_primary_to_user_feature.ipynb
	@echo

dataset: user_features
	# Generating datasets
	python ${PREFIX}4_feature_to_dataset.py

blob: 
	# Downloading data from BlobStorage
	python ${PREFIX}9_get_from_blob.py
	@echo
