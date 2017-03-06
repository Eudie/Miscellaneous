#!/usr/bin/python
# Author: Eudie

import pandas as pd
import json
import codecs
from pandas.io.json import json_normalize
from eventregistry import *

df = pd.DataFrame([])

list_of_json_files = ["category.json"]


for files in list_of_json_files:
    with codecs.open(files, "r", encoding='utf-8') as data_file:
        data = json.load(data_file)

    df = df.append(json_normalize(data))

print df



df = df.replace({'\n': ' ', '\t': ' ', '[|]': ' '}, regex=True)

writer = pd.ExcelWriter('Category.xlsx')
df.to_excel(writer, sheet_name='Sheet1', index=False)