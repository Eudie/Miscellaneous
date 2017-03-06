#!/usr/bin/python
# Author: Eudie

import codecs
import json
import pycurl
import urllib
from StringIO import StringIO
from pymongo import MongoClient
from bson.json_util import dumps

client = MongoClient('localhost:27017')
db = client.DST_news_aylien


def fetch_new_stories():
    fetched_stories = []
    stories = None
    next_cursor = "*"
    indicator = 0
    while stories is None or len(stories) > 0:
        try:
            url = "https://api.newsapi.aylien.com/api/v1/stories?" + \
                  "language%5B%5D=en" + \
                  "&published_at.start=2015-10-01T00:00:00.1Z" + \
                  "&published_at.end=2015-11-30T23:59:59.0Z" + \
                  "&categories.taxonomy=iptc-subjectcode" + \
                  "&categories.confident=true" + \
                  "&categories.id%5B%5D=07016000" + \
                  "&cluster=false" + \
                  "&cluster.algorithm=lingo" + \
                  "&sort_by=published_at" + \
                  "&sort_direction=desc" + \
                  "&cursor=" + next_cursor + \
                  "&per_page=100"
            #
            # url = "https://api.newsapi.aylien.com/api/v1/stories?text=Fitness&language%5B%5D=en&published_at.start=NOW-3DAY&published_at.end=NOW&categories.confident=true&cluster=false&cluster.algorithm=lingo&sort_by=published_at&sort_direction=desc&cursor="+ next_cursor +"&per_page=100"

            # url = "https://api.newsapi.aylien.com/api/v1/stories?language%5B%5D=en&published_at.start=NOW-3DAY&published_at.end=NOW&categories.taxonomy=iptc-subjectcode&categories.confident=true&categories.id%5B%5D=06003000&categories.id%5B%5D=04005000&categories.id%5B%5D=04005011&categories.id%5B%5D=04005005&categories.id%5B%5D=04009001&categories.id%5B%5D=06001000&categories.id%5B%5D=06006009&categories.id%5B%5D=04005003&categories.id%5B%5D=04005001&categories.id%5B%5D=04005004&categories.id%5B%5D=11020000&cluster=false&cluster.algorithm=lingo&sort_by=published_at&sort_direction=desc&cursor="+ next_cursor +"&per_page=100"
            # url = "https://api.newsapi.aylien.com/api/v1/stories?text=energy%20oil%20%22natural%20gas%22%20nuclear%20coal%20solar%20renewable%20%22wind%20energy%22%20%22bio%20fuel%22&language%5B%5D=en&published_at.start=NOW-3DAY&published_at.end=NOW&categories.confident=true&cluster=false&cluster.algorithm=lingo&sort_by=published_at&sort_direction=desc&cursor="+ next_cursor +"&per_page=100"

            # url = "https://api.newsapi.aylien.com/api/v1/stories?language%5B%5D=en&published_at.start=NOW-1DAY&published_at.end=NOW&categories.taxonomy=iptc-subjectcode&categories.confident=true&!categories.id%5B%5D=07016000&cluster=false&cluster.algorithm=lingo&sort_by=published_at&sort_direction=desc&cursor="+ next_cursor +"&per_page=100"

            buffer = StringIO()
            c = pycurl.Curl()
            c.setopt(pycurl.URL, url)
            c.setopt(pycurl.HTTPHEADER, ["Accept: application/json", "X-AYLIEN-NewsAPI-Application-ID: 6f23253d",
                                         "X-AYLIEN-NewsAPI-Application-Key: eca227a98bb7866d9d6386f497ac61d2"])
            c.setopt(c.WRITEDATA, buffer)
            c.perform()
            c.close()

            body = json.loads(buffer.getvalue())
        except:
            continue

        stories = body['stories']

        next_cursor = urllib.quote_plus(body['next_page_cursor'])

        fetched_stories += stories
        print("Fetched " + str(len(stories)) +
              " stories. Total story count so far: " + str(len(fetched_stories)))
        if len(stories) > 0:
            db.fitness_dec15_to_apr16.insert_many(stories)

        print indicator
        indicator += 1

    return "Got the data!!!"

if __name__ == "__main__":
    # For local use
    status = fetch_new_stories()

    # data = json.loads(dumps(db.fitness.find({}, {'_id': False})))
    print status

    # with codecs.open('Aylien1.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)






