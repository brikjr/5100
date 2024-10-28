#!/bin/bash
curl -L -o ../data/data.zip "https://www.kaggle.com/api/v1/datasets/download/arashnic/book-recommendation-dataset"

unzip ../data/data.zip -d ../data