# -*- coding: utf-8 -*-
import tweepy
import csv
import pandas as pd
from tweepy import OAuthHandler


####input your credentials here
consumer_key =''
consumer_secret = ''
access_token = ''
access_token_secret = ''


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

csvFile = open('C:/DataCamp/attempt2/trump.csv', 'a')

csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search, q="#trump", count=100, lang="en", since="01/01/2019").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])