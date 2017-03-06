#!/usr/bin/python
# Author: Eudie

import aylien_news_api
import json
import codecs
from aylien_news_api.rest import ApiException
#import Making_json_serializable

aylien_news_api.configuration.api_key['X-AYLIEN-NewsAPI-Application-ID'] = ''
# Configure API key authorization: app_key
aylien_news_api.configuration.api_key['X-AYLIEN-NewsAPI-Application-Key'] = ''

# create an instance of the API class
api_instance = aylien_news_api.DefaultApi()

language = ['en']
since = 'NOW-4HOUR'
until = 'NOW'
categories_taxonomy = 'iptc-subjectcode'
categories_id = ['07016000']


class FileItem:
    def __init__(self, fname):
        self.fname = fname

    def __repr__(self):
        return json.dumps(self.__dict__)


def toJSON(someJson):
    return json.dumps(someJson, default=lambda o: o.__dict__,
        sort_keys=True, indent=4)

try:
    # List stories
    api_response = api_instance.list_stories(language=language,
                                             published_at_start=since,
                                             published_at_end=until,
                                             categories_taxonomy=categories_taxonomy,
                                             categories_id=categories_id,
                                             per_page=100)

    #
    #
    # with codecs.open('Aylien.json', 'w', encoding='utf-8') as f:
    #     json.dumps(api_response, f, ensure_ascii=False, indent=4)

    print api_response.stories[1].body
    print len(api_response.stories)
    print toJSON(api_response.stories[0].author)
except ApiException as e:
    print("Exception when calling DefaultApi->list_stories: %s\n" % e)
