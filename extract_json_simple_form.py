import json
from datetime import datetime
import time

def tweet_simplify_filter(line):
    tweet=json.loads(line)
    if tweet['lang']!='en':
        return("NOT_ENG")
    gmttime=tweet['created_at']
    unixtime=int(time.mktime(time.strptime(gmttime.replace("+0000",''), '%a %b %d %H:%M:%S %Y')))
    hashtags = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    users = [user_mention['screen_name'] for user_mention in tweet['entities']['user_mentions']]
    urls = [url['expanded_url'] for url in tweet['entities']['urls']]
    
    tweetid=tweet['id']
    numfollowers=tweet['user']['followers_count']
    numfriends=tweet['user']['friends_count']
    if 'retweeted_status' in tweet:
        text = tweet['retweeted_status']['full_text']
    else:
        text = tweet['full_text']
    hashtags=[]
    users=[]
    media_urls=[]
    if 'media' in tweet['entities']:
        media_urls = [media['media_url'] for media in tweet['entities']['media']]
    return [unixtime,gmttime, tweetid, text, hashtags, users, urls, media_urls, numfollowers, numfriends]

def main():
    import get_tweets
    READFILE="tweetdataFile.json"
    WRITEFILE="newdataFile.json"
    readfile=open(READFILE,'r',encoding="utf-8")
    writefile=open(WRITEFILE,'w',encoding="utf-8")
    tweet_num=0
    for line in readfile:
        tweet_num+=1
        tweet_simpl=tweet_simplify_filter(line)
        if tweet_simpl=="NOT_ENG":
            pass
        else:
            writefile.write(str(tweet_simpl)+"\n")
            print(tweet_num," SUCCESS")
    readfile.close()
    writefile.close()

if __name__=="__main__":
    main()