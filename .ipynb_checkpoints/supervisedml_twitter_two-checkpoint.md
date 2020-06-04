# Project: Twitter Classification Algorithms – Part Two

**Project description:** There’s two parts of this project.

In this second (finally!) part, I wrote a system that predicts where does a tweet come from; either New York, London, or Paris, based on the language for each tweet. This project uses a Multinomial naïve Bayes classifier, a common option for text classification problems.

There are three datasets to work from:
* `new_york.json`
* `london.json`
* `paris.json`

These three files contain tweets from those locations.


```python
# Importing necessary modules...
from matplotlib import pyplot as plt
import pandas as pd
import sklearn.naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, multilabel_confusion_matrix #New in version 0.21.
from sklearn.naive_bayes import MultinomialNB
```

# Investigate the Data

To begin, let's take a look at the data. Some questions to answer for each dataset:
* The number of tweets.
* The columns, or features, of a tweet.
* The text of the n-th tweet in each dataset.
* What the n-th tweet actually looks like (corresponding to the columns)

#### `MultinomialNB` implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification.
See more at: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

## For New York...
Using the 12th tweet as example...


```python
new_york_tweets = pd.read_json("new_york.json", lines=True)
print(len(new_york_tweets))
print("---------------------------")
print(new_york_tweets.columns)
print("---------------------------")
print(new_york_tweets.loc[12]["text"])
print("---------------------------")
print(new_york_tweets.loc[12])
```

    4723
    ---------------------------
    Index(['created_at', 'id', 'id_str', 'text', 'display_text_range', 'source',
           'truncated', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
           'in_reply_to_user_id', 'in_reply_to_user_id_str',
           'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
           'contributors', 'is_quote_status', 'quote_count', 'reply_count',
           'retweet_count', 'favorite_count', 'entities', 'favorited', 'retweeted',
           'filter_level', 'lang', 'timestamp_ms', 'extended_tweet',
           'possibly_sensitive', 'quoted_status_id', 'quoted_status_id_str',
           'quoted_status', 'quoted_status_permalink', 'extended_entities',
           'withheld_in_countries'],
          dtype='object')
    ---------------------------
    Be best #ThursdayThoughts
    ---------------------------
    created_at                                           2018-07-26 13:32:43+00:00
    id                                                         1022474798973235200
    id_str                                                     1022474798973235200
    text                                                 Be best #ThursdayThoughts
    display_text_range                                                         NaN
    source                       <a href="http://twitter.com/download/iphone" r...
    truncated                                                                False
    in_reply_to_status_id                                                      NaN
    in_reply_to_status_id_str                                                  NaN
    in_reply_to_user_id                                                        NaN
    in_reply_to_user_id_str                                                    NaN
    in_reply_to_screen_name                                                   None
    user                         {'id': 1409002225, 'id_str': '1409002225', 'na...
    geo                                                                       None
    coordinates                                                               None
    place                        {'id': '01a9a39529b27f36', 'url': 'https://api...
    contributors                                                               NaN
    is_quote_status                                                          False
    quote_count                                                                  0
    reply_count                                                                  0
    retweet_count                                                                0
    favorite_count                                                               0
    entities                     {'hashtags': [{'text': 'ThursdayThoughts', 'in...
    favorited                                                                False
    retweeted                                                                False
    filter_level                                                               low
    lang                                                                        en
    timestamp_ms                                        2018-07-26 13:32:43.395000
    extended_tweet                                                             NaN
    possibly_sensitive                                                         NaN
    quoted_status_id                                                           NaN
    quoted_status_id_str                                                       NaN
    quoted_status                                                              NaN
    quoted_status_permalink                                                    NaN
    extended_entities                                                          NaN
    withheld_in_countries                                                      NaN
    Name: 12, dtype: object
    

## For London...
Using the 24th tweet as example...


```python
london_tweets = pd.read_json("london.json", lines=True)
print(len(london_tweets))
print("---------------------------")
print(london_tweets.columns)
print("---------------------------")
print(london_tweets.loc[24]["text"])
print("---------------------------")
print(london_tweets.loc[24])
```

    5341
    ---------------------------
    Index(['created_at', 'id', 'id_str', 'text', 'display_text_range', 'source',
           'truncated', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
           'in_reply_to_user_id', 'in_reply_to_user_id_str',
           'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
           'contributors', 'is_quote_status', 'extended_tweet', 'quote_count',
           'reply_count', 'retweet_count', 'favorite_count', 'entities',
           'favorited', 'retweeted', 'filter_level', 'lang', 'timestamp_ms',
           'possibly_sensitive', 'quoted_status_id', 'quoted_status_id_str',
           'quoted_status', 'quoted_status_permalink', 'extended_entities'],
          dtype='object')
    ---------------------------
    Great idea, so important to keep hydrated during this hot weather and hugely important to look after others who may… https://t.co/zKVkL3uGXY
    ---------------------------
    created_at                                           2018-07-26 13:39:55+00:00
    id                                                         1022476612070268929
    id_str                                                     1022476612070268928
    text                         Great idea, so important to keep hydrated duri...
    display_text_range                                                    [0, 140]
    source                       <a href="http://twitter.com/#!/download/ipad" ...
    truncated                                                                 True
    in_reply_to_status_id                                                      NaN
    in_reply_to_status_id_str                                                  NaN
    in_reply_to_user_id                                                        NaN
    in_reply_to_user_id_str                                                    NaN
    in_reply_to_screen_name                                                   None
    user                         {'id': 537961307, 'id_str': '537961307', 'name...
    geo                                                                       None
    coordinates                                                               None
    place                        {'id': '7093398a4249d151', 'url': 'https://api...
    contributors                                                               NaN
    is_quote_status                                                           True
    extended_tweet               {'full_text': 'Great idea, so important to kee...
    quote_count                                                                  0
    reply_count                                                                  0
    retweet_count                                                                0
    favorite_count                                                               0
    entities                     {'hashtags': [], 'urls': [{'url': 'https://t.c...
    favorited                                                                False
    retweeted                                                                False
    filter_level                                                               low
    lang                                                                        en
    timestamp_ms                                        2018-07-26 13:39:55.671000
    possibly_sensitive                                                           0
    quoted_status_id                                                   1.02242e+18
    quoted_status_id_str                                               1.02242e+18
    quoted_status                {'created_at': 'Thu Jul 26 10:04:20 +0000 2018...
    quoted_status_permalink      {'url': 'https://t.co/75wKQg2TcV', 'expanded':...
    extended_entities                                                          NaN
    Name: 24, dtype: object
    

## For Paris...
Using the 36th tweet as example...


```python
paris_tweets = pd.read_json("paris.json", lines=True)
print(len(paris_tweets))
print("---------------------------")
print(paris_tweets.columns)
print("---------------------------")
print(paris_tweets.loc[36]["text"])
print("---------------------------")
print(paris_tweets.loc[36])
```

    2510
    ---------------------------
    Index(['created_at', 'id', 'id_str', 'text', 'source', 'truncated',
           'in_reply_to_status_id', 'in_reply_to_status_id_str',
           'in_reply_to_user_id', 'in_reply_to_user_id_str',
           'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
           'contributors', 'is_quote_status', 'quote_count', 'reply_count',
           'retweet_count', 'favorite_count', 'entities', 'favorited', 'retweeted',
           'filter_level', 'lang', 'timestamp_ms', 'display_text_range',
           'extended_entities', 'possibly_sensitive', 'quoted_status_id',
           'quoted_status_id_str', 'quoted_status', 'quoted_status_permalink',
           'extended_tweet'],
          dtype='object')
    ---------------------------
    Quiero Volver de Tini et Sebastian ma giga vie je vais chialer ma race
    ---------------------------
    created_at                                           2018-07-27 17:42:39+00:00
    id                                                         1022900084021907458
    id_str                                                     1022900084021907456
    text                         Quiero Volver de Tini et Sebastian ma giga vie...
    source                       <a href="http://twitter.com/download/iphone" r...
    truncated                                                                False
    in_reply_to_status_id                                                      NaN
    in_reply_to_status_id_str                                                  NaN
    in_reply_to_user_id                                                        NaN
    in_reply_to_user_id_str                                                    NaN
    in_reply_to_screen_name                                                   None
    user                         {'id': 4544602403, 'id_str': '4544602403', 'na...
    geo                                                                       None
    coordinates                                                               None
    place                        {'id': '09f6a7707f18e0b1', 'url': 'https://api...
    contributors                                                               NaN
    is_quote_status                                                          False
    quote_count                                                                  0
    reply_count                                                                  0
    retweet_count                                                                0
    favorite_count                                                               0
    entities                     {'hashtags': [], 'urls': [], 'user_mentions': ...
    favorited                                                                False
    retweeted                                                                False
    filter_level                                                               low
    lang                                                                        es
    timestamp_ms                                        2018-07-27 17:42:39.252000
    display_text_range                                                         NaN
    extended_entities                                                          NaN
    possibly_sensitive                                                         NaN
    quoted_status_id                                                           NaN
    quoted_status_id_str                                                       NaN
    quoted_status                                                              NaN
    quoted_status_permalink                                                    NaN
    extended_tweet                                                             NaN
    Name: 36, dtype: object
    

# Classifying using language: Naive Bayes Classifier

We're going to create a Naive Bayes Classifier! Let's begin by looking at the way language is used differently in these three locations. Let's grab the text of all of the tweets and make it one big list. In the code block below, we've created a list of all the New York tweets. Do the same for `london_tweets` and `paris_tweets`.

Then combine all three into a variable named `all_tweets` by using the `+` operator. For example, `all_tweets = new_york_text + london_text + ...`

Let's also make the labels associated with those tweets. `0` represents a New York tweet, `1`  represents a London tweet, and `2` represents a Paris tweet. Finish the definition of `labels`.


```python
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

all_tweets = new_york_text + london_text + paris_text
labels = ([0] * len(new_york_text)) + ([1] * len(london_text)) + ([2] * len(paris_text))
```

A preview of the first ten entries of the  `new_york_text`  list:


```python
num = 1
for x in new_york_text[:10]:
    print(str(num) + ". " + x)
    num += 1
```

    1. @DelgadoforNY19 Calendar marked.
    2. petition to ban more than one spritz of cologne
    3. People really be making up beef with you in they head lol
    4. 30 years old.. wow what a journey... I moved to NYC at 22 young and dumb, without even $100 in my bank account and… https://t.co/awjzsvoGS7
    5. At first glance it looked like asparagus with chicken and gravy smothered over it or potatoes. She gotta be extra w… https://t.co/InBNnsKuWu
    6. texting me bullshit i just swipe and delete it
    7. Nailed it. https://t.co/dYYvyYVnxZ
    8. 🗽Cammy Set for tomboyfeels 
    Cop @ https://t.co/eaNB5dNIdG (custom pieces 2)
    Shot by lexi_vv_photography 
    Creative D… https://t.co/25N9vMi97j
    9. @notepinuch Thank you ka 😂
    10. I'm at Crunch - Bushwick - @crunchgym in Brooklyn, NY https://t.co/WRGDRsEkPD
    

# Making a Training and Test Set

We can now break our data into a training set and a test set. We'll use scikit-learn's `train_test_split` function to do this split. This function takes two required parameters: It takes the data, followed by the labels. Set the optional parameter `test_size` to be `0.2`. Finally, set the optional parameter `random_state` to `42` (can be any value we want, in this case, I used 42 for the hell of it). 

This will make it so your data is split in the same way as the data in our solution code. 

This function returns 4 items in this order:
1. The training data
2. The testing data
3. The training labels
4. The testing labels

Store the results in variables named `xtrain`, `xtest`, `ytrain`, and `ytest`. In which:

1. `xtrain` and `xtest` is representing the actual data
2. `ytrain` and `ytest` is representing the labels we instantiated earlier.

Then we print the length of `xtrain` and the length of `xtest`.


```python
xtrain, xtest, ytrain, ytest = train_test_split(all_tweets, labels, train_size=0.8, test_size=0.2, random_state = 42)
```

Now the first ten entries `xtrain` and the `ytrain` labels look as follows:


```python
for x, y in zip(xtrain[:10], ytrain[:10]):
    print(x + " | corresponds to the label " + str(y))
    print("--------------------------------------------------------------------------------")
```

    💦 Ensure your lashes are always clean, especially in this hot weather 💦
    .
    .
    To book your appointment, click the lin… https://t.co/7JHNrfFaSm | corresponds to the label 1
    --------------------------------------------------------------------------------
    August 3rd #Underwater @CodySimpson @codyandthetide https://t.co/HXXQlrqEDf | corresponds to the label 2
    --------------------------------------------------------------------------------
    @Nyusha_Nyusha Свет не видимый можно сделать как на пульте, только свет пробивает даже алмазы | corresponds to the label 2
    --------------------------------------------------------------------------------
    @GrantJBackwell @WindyCOYS @AintNoSanAn Yeah. The annoying thing deals like Grealish should have been wrapped up ag… https://t.co/U5EMSzP5jQ | corresponds to the label 1
    --------------------------------------------------------------------------------
    Depuis physique ou chimie je la trouvais trop belle | corresponds to the label 2
    --------------------------------------------------------------------------------
    I don’t remember ever feeling so hot and sticky by doing absolutely nothing. Polished one bottle of sparkling water… https://t.co/SxtMYWYCPO | corresponds to the label 1
    --------------------------------------------------------------------------------
    @Baby_Anthoo @ap55k I'll be at ur doorstep in 10 mins keep that energy | corresponds to the label 0
    --------------------------------------------------------------------------------
    Fab Melo lol Rest In Peace in his soul https://t.co/cvyKpcXY10 | corresponds to the label 0
    --------------------------------------------------------------------------------
    Tumbling Dice is on it. Deservedly so https://t.co/o9Det328Sv | corresponds to the label 0
    --------------------------------------------------------------------------------
    What a terrible combination 🤢 | corresponds to the label 0
    --------------------------------------------------------------------------------
    

## Remember that:

* Labels 0 corresponds to New York tweets
* Labels 1 corresponds to London tweets
* Labels 2 corresponds to Paris tweets

# Making the Count Vectors

To use a Naive Bayes Classifier, we need to transform our lists of words into count vectors. This changes the sentence `"I love New York, New York"` into a list that contains:

* Two `1`s because the words `"I"` and `"love"` each appear once.
* Two `2`s because the words `"New"` and `"York"` each appear twice.
* Many `0`s because every other word in the training set didn't appear at all.

To start, we create a `CountVectorizer` named `counter`.

Next, fit the model the `.fit()` method using `xtrain` as a parameter. This teaches the counter our vocabulary.

Finally, let's transform `xtrain` and `ytrain` into Count Vectors. Call `counter`'s `.transform()` method using `xtrain` as a parameter and store the result in `train_counts`. Then, we do the same for `xtest` and store the result in `test_counts`.

We then print `xtrain[3]` and `train_counts[3]` to see what a tweet looks like as a Count Vector.


```python
counter = CountVectorizer() #Instantiating the counter object
counter.fit(xtrain) 
train_counts = counter.transform(xtrain) #Actually counting the words
test_counts = counter.transform(xtest)

print(xtrain[3])
print(train_counts[3])
```

    @GrantJBackwell @WindyCOYS @AintNoSanAn Yeah. The annoying thing deals like Grealish should have been wrapped up ag… https://t.co/U5EMSzP5jQ
      (0, 1959)	1
      (0, 2079)	1
      (0, 2617)	1
      (0, 3840)	1
      (0, 6214)	1
      (0, 7578)	1
      (0, 11818)	1
      (0, 11849)	1
      (0, 12392)	1
      (0, 13036)	1
      (0, 16141)	1
      (0, 24477)	1
      (0, 26597)	1
      (0, 26795)	1
      (0, 27787)	1
      (0, 28087)	1
      (0, 29442)	1
      (0, 29650)	1
      (0, 30058)	1
    

# Train and Test the Classifier

We now have the inputs to our classifier. We can use the CountVectors to train and test the classifier.

First, we make a `MultinomialNB` object named `classifier`.

Next, we call `classifier`'s `.fit()` method. This method takes two parameters &mdash; the training data and the training labels. `train_counts` contains the training data and `ytrain` containts the labels for that data. Calling `.fit()` calculates all of the probabilities used in Bayes Theorem. The model is now ready to quickly predict the location of a new tweet. 

We can now test our model by calling `classifier`'s `.predict()` method using `test_counts` as a parameter. We're storing the results in a variable named `predictions`.


```python
classifier = MultinomialNB() #Instantiating the classifier object we imported earlier
classifier.fit(train_counts, ytrain)
predictions = classifier.predict(test_counts)
for x, y in zip(xtest[:10], predictions[:10]):
    print(x + " | corresponds to the label " + str(y))
    print("--------------------------------------------------------------------------------")
```

    @Muflette1 Point de vocabulaire: un individu qui agresse des policiers n'est pas un "manifestant", mais un "délinquant". | corresponds to the label 2
    --------------------------------------------------------------------------------
    Also - I was right. @bchettt saw "Do You Wanna Come Over?" live and HAD FUN. | corresponds to the label 1
    --------------------------------------------------------------------------------
    @lunglock @KirstenPrice1 @helloitsnicola It’s sensible.  Last year I had mc fs on Friday and Sunday, and had a fuck… https://t.co/gQ1OVXxTHA | corresponds to the label 1
    --------------------------------------------------------------------------------
    This morning, I visited a Psychic for the first time. But instead of predicting my future, she predicted yours. You… https://t.co/hP7gO3j1GL | corresponds to the label 0
    --------------------------------------------------------------------------------
    Tonight ain't the end of the story,
    just keep turning the page.
    Don't give into the heartache, 
    don't give into the… https://t.co/zYj4ZQ6y6q | corresponds to the label 0
    --------------------------------------------------------------------------------
    @KeyomeRosetta HAPPY BIRTHDAY BITCHHHH 😘😘 | corresponds to the label 0
    --------------------------------------------------------------------------------
    @ditterhansen @RubiesB4Swine @Femi_Sorry Eton’s primary purpose is/was to teach the upper echelons of society. That… https://t.co/3dN3G1gAtj | corresponds to the label 0
    --------------------------------------------------------------------------------
    👀 https://t.co/CQlF1ICEuj | corresponds to the label 1
    --------------------------------------------------------------------------------
    5quid to get from Hastings to London. Will cost me more to get across London that to get there! | corresponds to the label 1
    --------------------------------------------------------------------------------
    Go go go "Sans un Bruit" ! | corresponds to the label 2
    --------------------------------------------------------------------------------
    

# Evaluating Your Model

Now that the classifier has made its predictions, let's see how well it did. Let's look at two different ways to do this. First, we call scikit-learn's `accuracy_score`, `recall_score`, `precision_score`, `f1_score` functions. All of these functions should take two parameters &mdash;  the `ytest` and your `predictions`. 

Some quick definitions as a refresher:
* Accuracy is the total number of correctly classified points and dividing by the total number of points
* Recall is the ratio of correct positive predictions to the total positives examples.
* Precision is the ratio of correct positive predictions to the total predicted positives.
* F1 score is the harmonic mean of precision and recall. 


```python
print("The accuracy of the model is: " + str(accuracy_score(ytest, predictions)))
print("The recall of the model is: " + str(recall_score(ytest, predictions, average=None)))
print("The precision of the model is: " + str(precision_score(ytest, predictions, average=None)))
print("The F1 Score of the model is: " + str(f1_score(ytest, predictions, average=None)))
```

    The accuracy of the model is: 0.7145129224652087
    The recall of the model is: [0.62512873 0.78256795 0.7442348 ]
    The precision of the model is: [0.72175981 0.65285379 0.89873418]
    The F1 Score of the model is: [0.66997792 0.71184996 0.81422018]
    

* The parameter `pos_label` is ignored, because we're dealing with a multi-label classification (three labels in total)

* The list containing three elements in `recall`, `precision`, and `F1 Score` represents each class of the label. In other words, the first element corresponds to the label `0`, which is a New York tweet.

* We use `average=None` in the last three metrics because in this instance, I actually want to see how the model fares to classify all three kinds of tweets. Read more about the parameters here: https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification

# Confusion matrix

The other way we can evaluate your model is by looking at the **confusion matrix**. A confusion matrix is a table that describes how your classifier made its predictions. For example, if there were two labels, A and B, a confusion matrix might look like this:

```
9 1
3 5
```

In this example, the first row shows how the classifier classified the true A's. It guessed that 9 of them were A's and 1 of them was a B. The second row shows how the classifier did on the true B's. It guessed that 3 of them were A's and 5 of them were B's.

For our project using tweets, there were three classes &mdash; `0` for New York, `1` for London, and `2` for Paris. We can see the confustion matrix by printing the result of the `multilabel_confusion_matrix` function using `ytest` and `predictions` as parameters.

Read more about the `multilabel_confusion_matrix` on: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html


```python
print(multilabel_confusion_matrix(ytest, predictions))
```

    [[[1310  234]
      [ 364  607]]
    
     [[1004  444]
      [ 232  835]]
    
     [[1998   40]
      [ 122  355]]]
    

#### The three series of confusion matrices are based on the one-vs-all transformation.

For instance, in the last series (belonging to Paris) of confusion matrix, the classifier performs:
* 1998 true negatives
* 355 true positives

This is reasonable because France is not an English-speaking country. Whereas in the first series (belonging to New York) of confusion matrix, the classifier performs:
* 1310 true negatives
* 607 true positives

The errors are a little big higher than the one in Paris, because both London residents and New Yorkers speak English on a daily basis. Tweets coming from two English speaking countries are harder to distinguish than tweets in different languages.

# Testing New Tweets

Considering I'm on Twitter literally every hour, we can put our custom-made tweets and see whether the classifier got it right. 
* Here, I'm storing my new tweet in the `tweet` variable.
* We then call `counter`'s `.transform()` method using `[tweet]` as a parameter (with the brackets to make an array of the regarded string). Then, we save the result as `tweet_counts`.
* Finally, we pass `tweet_counts` as parameter to `classifier`'s `.predict()` method and save it as the variable `secondPrediction`. 


```python
tweet = "Hey, can you predict where this tweet comes from?"
tweet_counts = counter.transform([tweet])
secondPrediction = classifier.predict(tweet_counts)
print(secondPrediction)
```

    [1]
    

Would you look at that, our classifier predicts that my tweet `Hey, can you predict where this tweet comes from?` comes from London!
