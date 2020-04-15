## Project: Twitter Classification Algorithms

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
```python3
import pandas as pd
from matplotlib import pyplot as plt
all_tweets = pd.read_json("random_tweets.json", lines=True)
all_tweets.head(10)
```
Returns:
| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
