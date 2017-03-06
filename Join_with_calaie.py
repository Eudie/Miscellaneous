#!/usr/bin/python
# Author: Eudie


import json
from pymongo import MongoClient
import codecs
from bson.json_util import dumps

client = MongoClient('localhost:27017')
db = client. DST_news_aylien

# List = ['fitness_monthly_july16']

List = ['fitness_monthly_june16',
        'fitness_monthly_aug16',
        'fitness_monthly_may16',
        'fitness_monthly_july16',
        'fitness_dec15_to_apr16']

Big_collection = []


def merge(list_of_db):
    for collection in list_of_db:
        Big_collection.extend(json.loads(dumps(db[collection].find())))

    print(len(Big_collection))
    with codecs.open('temp.json', 'w', encoding='utf-8') as f:
        json.dump(Big_collection, f, ensure_ascii=False, indent=4)


def clean_temp():
    db.temp.drop()


def make_pipeline():
    pipeline = [{'$lookup':
                {'from': 'temp',
                 'localField': 'link',
                 'foreignField': 'links.permalink',
                 'as': 'aylien_data'}}
                ]
    return pipeline


def joining_calai_with_aylien():
    pipe = make_pipeline()
    result = db.calai_fitness_1.aggregate(pipe)

    data = json.loads(dumps(result))
    # print data[1]
    print(len(data))
    with codecs.open('calai_with_aylien_rnn_output.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    n = 0
    for each in data:
        if not each['aylien_data']:
            n += 1
            # print each['aylien_data']

    if n != 0:
        print('Not all documents matched!!!')

if __name__ == "__main__":
    # merge(List)
    joining_calai_with_aylien()
    # clean_temp()

