# Project: Twitter Classification Algorithms

**Project description:** There's two parts of this project. 

In the first part, I wrote a system that predicts whether or not a tweet will go viral by using a K-Nearest Neighbor classifier. Some questions to answer:
1. What features of a tweet are the most important in determining its virality? 
2. Does the length of the tweet matter? 
3. What about the number of hashtags? 
4. Maybe information about the account that sent the tweet is most important. 

In the second part, I wrote a system that tests the power of Naive Bayes classifiers by predicting whether a tweet was sent from New York City, London, or Paris. Some questions to answer:
1. How is languaged used differently in these three cities?
2. Can the classifier automatically detect the difference between French and English? 
3. Can it learn local phrases or slang? 
4. Can you create tweets that trick the system?


# PART ONE: PREDICTING VIRAL TWEETS

## 1. Getting to know the data
Taking a look inside the data for the first time, we first have to import the essential packages before eventually printing the ```.head()``` of the DataFrame:
```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

all_tweets = pd.read_json("random_tweets.json", lines=True)
print(all_tweets.head(10))
```
Returns:
```
  created_at                   id               id_str  \
0 2018-07-31 13:34:40+00:00  1024287229525598210  1024287229525598208   
1 2018-07-31 13:34:40+00:00  1024287229512953856  1024287229512953856   
2 2018-07-31 13:34:40+00:00  1024287229504569344  1024287229504569344   
3 2018-07-31 13:34:40+00:00  1024287229496029190  1024287229496029184   
4 2018-07-31 13:34:40+00:00  1024287229492031490  1024287229492031488   
5 2018-07-31 13:34:40+00:00  1024287229491994625  1024287229491994624   
6 2018-07-31 13:34:40+00:00  1024287229454233605  1024287229454233600   
7 2018-07-31 13:34:40+00:00  1024287229445734400  1024287229445734400   
8 2018-07-31 13:34:40+00:00  1024287229441658880  1024287229441658880   
9 2018-07-31 13:34:40+00:00  1024287229424947201  1024287229424947200   

                                                text  truncated  \
0  RT @KWWLStormTrack7: We are more than a month ...      False   
1  @hail_ee23 Thanks love its just the feeling of...      False   
2  RT @TransMediaWatch: Pink News has more on the...      False   
3  RT @realDonaldTrump: One of the reasons we nee...      False   
4  RT @First5App: This hearing of His Word doesn’...      False   
5  RT @attackerman: This is torture: “The staff t...      False   
6  Did a demo of our Mobile Prototyping Kit at UX...       True   
7  RT @itstae13: Stop getting rid of your pets be...      False   
8  RT @RealErinCruz: Someone sent me a thought pr...      False   
9  RT @penis_hernandez: when you ask how a white ...      False   

                                            entities  \
0  {'hashtags': [], 'symbols': [], 'user_mentions...   
1  {'hashtags': [], 'symbols': [], 'user_mentions...   
2  {'hashtags': [], 'symbols': [], 'user_mentions...   
3  {'hashtags': [], 'symbols': [], 'user_mentions...   
4  {'hashtags': [], 'symbols': [], 'user_mentions...   
5  {'hashtags': [], 'symbols': [], 'user_mentions...   
6  {'hashtags': [], 'symbols': [], 'user_mentions...   
7  {'hashtags': [], 'symbols': [], 'user_mentions...   
8  {'hashtags': [], 'symbols': [], 'user_mentions...   
9  {'hashtags': [], 'symbols': [], 'user_mentions...   
```
... and a lot more columns. Since we don't have the luxury to manually gain every hidden insight when first touching the data, we can do so by entering:

```python
print("-------------------------")
#The length of the dataset (i.e. the number of tweets)
print(str(len(all_tweets)) + " tweets found in this dataset.")
print("-------------------------")
#The name of the columns
print(all_tweets.columns)
print("-------------------------")
#The first tweet in the dataset
print(all_tweets.loc[0]['text'])
print("-------------------------")
#The user (as a dictionary) from the first tweet in the dataset
print(all_tweets.loc[0]['user'])
print("-------------------------")
#The location from the first tweet in the dataset
print(all_tweets.loc[0]['user']['location'])
```
This prints the following useful information:
```
-------------------------
11099 tweets found in this dataset.
-------------------------
Index(['created_at', 'id', 'id_str', 'text', 'truncated', 'entities',
       'metadata', 'source', 'in_reply_to_status_id',
       'in_reply_to_status_id_str', 'in_reply_to_user_id',
       'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo',
       'coordinates', 'place', 'contributors', 'retweeted_status',
       'is_quote_status', 'retweet_count', 'favorite_count', 'favorited',
       'retweeted', 'lang', 'possibly_sensitive', 'quoted_status_id',
       'quoted_status_id_str', 'extended_entities', 'quoted_status',
       'withheld_in_countries'],
      dtype='object')
-------------------------
RT @KWWLStormTrack7: We are more than a month into summer but the days are getting shorter. The sunrise is about 25 minutes later on July 3…
-------------------------
{'id': 145388018, 'id_str': '145388018', 'name': 'Derek Wolkenhauer', 'screen_name': 'derekw221', 'location': 'Waterloo, Iowa', 'description': '', 'url': None, 'entities': {'description': {'urls': []}}, 'protected': False, 'followers_count': 215, 'friends_count': 335, 'listed_count': 2, 'created_at': 'Tue May 18 21:30:10 +0000 2010', 'favourites_count': 3419, 'utc_offset': None, 'time_zone': None, 'geo_enabled': True, 'verified': False, 'statuses_count': 4475, 'lang': 'en', 'contributors_enabled': False, 'is_translator': False, 'is_translation_enabled': False, 'profile_background_color': '022330', 'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme15/bg.png', 'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme15/bg.png', 'profile_background_tile': False, 'profile_image_url': 'http://pbs.twimg.com/profile_images/995790590276243456/cgxRVviN_normal.jpg', 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/995790590276243456/cgxRVviN_normal.jpg', 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/145388018/1494937921', 'profile_link_color': '0084B4', 'profile_sidebar_border_color': 'A8C7F7', 'profile_sidebar_fill_color': 'C0DFEC', 'profile_text_color': '333333', 'profile_use_background_image': True, 'has_extended_profile': True, 'default_profile': False, 'default_profile_image': False, 'following': False, 'follow_request_sent': False, 'notifications': False, 'translator_type': 'none'}
-------------------------
Waterloo, Iowa
```

## 2. Defining viral Tweets
A K-Nearest Neighbor classifier is a supervised machine learning algorithm, and as a result, we need to have a dataset with tagged labels. For this specific example, we need a dataset where every tweet is marked as viral or not viral. Unfortunately, this isn't a feature of our dataset — we'll need to make it ourselves.

For a tweet to be defined 'Viral', I would want to look at the number of retweets that tweet has. Let's create a new column called `is_viral` which returns 1 if the amount of retweets is high enough, and 0 otherwise. 

But how high is *high enough*? I would want to know the median amount of retweets among the 11099 tweets.

To make this entirely new feature, we enter the following:
```python
threshold_of_viral_RT = (all_tweets.retweet_count.median())
print(threshold_of_viral_RT)
```

Prints `13.0`.

```python
all_tweets['is_viral'] = np.where(all_tweets.retweet_count > threshold_of_viral_RT, 1, 0)
print(all_tweets.is_viral.value_counts())
```

Prints:
```
0    8972
1    2127
Name: is_viral, dtype: int64
```
From this, we can first conclude that there are 2127 'Viral' tweets according to my algorithm, and 8972 'non-Viral' tweets. Further calculation on my end proves that only **19% of the dataset is classified as 'Viral'**

## 3. Making features