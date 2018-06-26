import tweepy
import json
import sys

def main():
    
    WRITEFILE="tweetdataFile.json"
    
    consumer_key = "c7AlJgFlPQmabD00ZbJWCa945"
    consumer_secret = "FW6vVdhQ0mJRupysusXaF8wfNzLf3zGJOjApmBsQuoP0qm5Tu0"
    access_token = "935939204898480128-aZo8IRmrsNBy8REatS4Xqdmt2P1gbwQ"
    access_token_secret = "AxfG7AUsUhWjr1JbQjzlPlaLUe8Y2QAEYCQoZPrElvv5S"
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    data=tweepy.Cursor(api.search,q="sri lanka",lang="en",tweet_mode="extended").items(5)
    
    dataFile=open(WRITEFILE, 'w',encoding="utf-8")
    for tweet in data:
        print(tweet._json)
        json.dump(tweet._json,dataFile)
        dataFile.write("\n")
    dataFile.close()

if __name__=='__main__':
    main()