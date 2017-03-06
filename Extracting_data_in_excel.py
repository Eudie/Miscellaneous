#!/usr/bin/python
# Author: Eudie


import json
import pandas as pd
from pymongo import MongoClient
from bson.json_util import dumps
import codecs


def extracting_to_excel():
    client = MongoClient('localhost:27017')
    db = client.DST_news_aylien

    data = json.loads(dumps(db.fitness_dec15_to_apr16.find({}, {"_id": 0, "links.permalink": 1, "body": 1})))

    df = pd.io.json.json_normalize(data)
    df = df[['links.permalink', 'body']]
    df.columns = ['Link', 'Main Article']

    writer = pd.ExcelWriter('Fitness_dec15_to_apr16.xlsx')
    df.to_excel(writer, sheet_name='Fitness_News', index=False)


def extracting_to_json():
    client = MongoClient('localhost:27017')
    db = client.DST_news_aylien

    data = json.loads(dumps(db.fitness_dec15_to_apr16.find({}, {"_id": 0, "links.permalink": 1, "body": 1})))

    with codecs.open('Fitness_dec15_to_apr16.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # extracting_to_excel()
    extracting_to_json()

