#!/usr/bin/python
# Author: Eudie


def add_city(db):
    db.cities.insert({"name": "Chicago"})


def get_city(db):
    return db.cities.find_one()


def get_db():
    # For local use
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    # 'examples' here is the database name. It will be created if it does not exist.
    db = client.examples
    return db


if __name__ == "__main__":
    # For local use
    db = get_db() # uncomment this line if you want to run this locally
    add_city(db)
    print get_city(db)
