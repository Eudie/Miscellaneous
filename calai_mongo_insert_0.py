#!/usr/bin/python
# Author: Eudie

import json
import codecs
from pymongo import MongoClient
import os
import sys


client = MongoClient('localhost:27017')
db = client.DST_news_aylien

df = []
flag_name = sys.argv[1]
list_of_json_files = os.listdir(flag_name)


for files in list_of_json_files:
    with codecs.open(flag_name + "/" + files, "r", encoding='utf-8') as data_file:
        data = json.load(data_file)
        print type(data[1])
        print len(data[1])
    df.extend(data)
    # print df

with codecs.open('Fitness_calai_dec_jul.json', 'w', encoding='utf-8') as f:
    json.dump(df, f, ensure_ascii=False, indent=4)


