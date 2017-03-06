#!/usr/bin/python
# Author: Eudie


import time
import aylien_news_api
from aylien_news_api.rest import ApiException


def fetch_new_stories(params={}):
    fetched_stories = []
    stories = None

    while stories is None or len(stories) > 0:
        try:
            response = api_instance.list_stories(language=params['language'],
                                                 published_at_start=params['published_at_start'],
                                                 published_at_end=params['published_at_end'],
                                                 categories_taxonomy=params['categories_taxonomy'],
                                                 categories_id=params['categories_id'],
                                                 per_page=params['per_page'],
                                                 cursor=params['cursor'])
        except ApiException as e:
            if (e.status == 429):
                print('Usage limit are exceeded. Wating for 30 seconds...')
                time.sleep(30)
                continue

        stories = response.stories
        params['cursor'] = response.next_page_cursor

        fetched_stories += stories
        print("Fetched " + str(len(stories)) +
              " stories. Total story count so far: " + str(len(fetched_stories)))

    return fetched_stories


# Configure API key authorization: app_id
aylien_news_api.configuration.api_key['X-AYLIEN-NewsAPI-Application-ID'] = ''
# Configure API key authorization: app_key
aylien_news_api.configuration.api_key['X-AYLIEN-NewsAPI-Application-Key'] = ''

# create an instance of the API class
api_instance = aylien_news_api.DefaultApi()

params = {
    'language': ['en'],
    'published_at_start': 'NOW-6HOUR',
    'published_at_end': 'NOW',
    'categories_taxonomy': 'iptc-subjectcode',
    'categories_id': ['07016000'],
    'cursor': '*',
    'per_page': 100
}

stories = fetch_new_stories(params)

print stories[0].body
print('************')
print('Fetched ' + str(len(stories)))
