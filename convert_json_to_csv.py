#!/usr/bin/python
# Author: Eudie

import json
import codecs
import pandas as pd
import csv


with codecs.open('calai_with_aylien.json', "r", encoding='utf-8') as data_file:
    data = json.load(data_file, encoding='utf-8')


id_link = []
type_of_category = []
value = []

for document in data:
    for i in range(0, len(document['calai_response'])):
        keys = document['calai_response'].keys()[i]
        if keys != 'doc':
            cat_type = document['calai_response'].values()[i]['_typeGroup']
            if cat_type in ['entities', 'socialTag']:
                id_link.append(document['link'])
                type_of_category.append(cat_type)
                value.append(document['calai_response'].values()[i]['name'].lower())


output_df = pd.DataFrame({'link': id_link, 'category_type': type_of_category, 'value': value})

output_df.to_csv("calai_categories.csv", encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
