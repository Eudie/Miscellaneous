#!/usr/bin/python
# Author: Eudie


import json
from urllib import quote_plus
from pymongo import MongoClient
from bson.json_util import dumps

user = 'meritus'
password = 'Meritus@123'
socket_path = '10.10.0.8'


def main():
    uri = "mongodb://%s:%s@%s/test" % (quote_plus(user), quote_plus(password), quote_plus(socket_path))
    print uri
    client = MongoClient(uri)
    db = client.test
    print json.loads(dumps(db.colltest.find()))


if __name__ == "__main__":
    main()






