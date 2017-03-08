#!/usr/bin/python
# Author: Eudie


import json
import pandas as pd
from pymongo import MongoClient
from bson.json_util import dumps


def main():
    client = MongoClient('localhost:27017')
    db = client.DST_news_aylien

    data = json.loads(dumps(db.enerdy_by_category.aggregate({}, {"_id": 0, "links.permalink": 1, "body": 1})))


if __name__ == "__main__":
    main()


