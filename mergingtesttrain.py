#!/usr/bin/python
# Author: Eudie

import json
import codecs
import pandas as pd
import csv

with codecs.open('calai_with_aylien_rnn_output.json', "r", encoding='utf-8') as data_file:
    data = json.load(data_file, encoding='utf-8')

with codecs.open('calai_with_aylien.json', "r", encoding='utf-8') as data_file:
    data1 = json.load(data_file, encoding='utf-8')


print(len(data))
print(len(data1))

data.extend(data1)


link = []
link1 = []

output = []
for doc in data:
    link.append(doc['link'])


for element in data:
    if element['link'] not in link1:
        output.append(element)
        link1.append(element['link'])

with codecs.open('full_data_for_demo.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

