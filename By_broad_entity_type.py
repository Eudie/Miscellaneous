#!/usr/bin/python
# Author: Eudie

import json
import codecs
import pandas as pd
import csv

search = 'sportsgame'
with codecs.open('full_data_for_demo.json', "r", encoding='utf-8') as data_file:
    data = json.load(data_file, encoding='utf-8')

id_link = []
type_of_entity = []
name_of_entity = []

for document in data:
    for i in range(0, len(document['calai_response'])):
        keys = document['calai_response'].keys()[i]
        if keys != 'doc':
            cat_type_for_entity = document['calai_response'].values()[i]['_typeGroup']
            if cat_type_for_entity == 'entities':
                if document['calai_response'].values()[i]['_type'].lower() == search:
                    id_link.append(document['link'])
                    type_of_entity.append(document['calai_response'].values()[i]['_type'].lower())
                    name_of_entity.append(document['calai_response'].values()[i]['name'].lower())


output_df = pd.DataFrame({'entity name': name_of_entity, 'entity type': type_of_entity, 'link': id_link})
print(len(output_df))

# output_df = output_df.drop_duplicates(['entity name'])
# print(output_df)

output_df.to_csv(search + "_calai_entity_type_full_data_for_demo.csv", encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
