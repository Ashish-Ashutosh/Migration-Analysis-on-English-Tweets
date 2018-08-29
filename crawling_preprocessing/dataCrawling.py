import csv

import tweepy

from crawling_preprocessing import twitter_credentials

# Switching to application authentication
auth = tweepy.AppAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)

# Setting up new api wrapper, using authentication only
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Error handling
if (not api):
    print("Problem Connecting to API")

# Open/Create a file to append data
csvFile = open('MigrationTweets.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

"""def process_or_store(tweet):
    print(json.dumps(tweet))"""

#'immigrant OR refugee OR migration'

hashtag_list ='place:96683cc9126741d1 #proimmigration OR #preimmigrationReform OR' \
                '#proOpenBorders OR #NoWallNoBan OR #ImmigrantsAreWelcome OR' \
                '#AntiIllegalImmigration OR #nationOfImmigrants OR #RefugeesWelcome OR' \
                '#ImmigrantsWelcome OR #fightIgnoranceNotImmigrants OR #noHumansIllegal OR' \
    '            #noBanNoWallNoRaids OR' \
                '#ImmigrationNotWelocme OR #AntiImmigration OR #DayWithoutImmigrants OR' \
                '#BuildTheWall OR #Illegalaliens OR #ImmigrationReform OR #crimmigrants OR' \
                '#immigration OR #immigrants OR #migration OR #ImmigrationBan OR #TravelBan OR' \
                'immigrant ban OR immigration ban OR travel ban OR immigration order,'

for tweet in tweepy.Cursor(api.search,q=hashtag_list,lang="en", since="2018-05-01").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
   # process_or_store(tweet._json)