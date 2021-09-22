import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import gensim
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import tweepy
import re

tweets = []

def newsTweets(tweets):


    longtext = []

    consumer_key = "urWpojmuasQPi89qSo0LGfTJd"
    consumer_secret = "VtVVnKiJFXr0Q7tPyJlFUlq8VdJ8VPmJCgpcd20yvRnuYhLnyy"
    access_key = "225925050-rg5SVLJMFDdeEAMQEWUEhTVzlsLGYKvy4BR5thgh"
    access_secret = "FqWSfmsp5EXLnfE9j7AkFBF8w3xrmG2Ob6EIcoEkpy8p5"

    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        auth.get_authorization_url()
        api = tweepy.API(auth)
    except tweepy.TweepError:
        print('Hata')

    public_tweets = api.user_timeline(screen_name="nytimes", count=30)
    public_tweets2 = api.user_timeline(screen_name="CNN", count=30)

    for tweet in public_tweets:
        tmp = re.sub(r"http\S+", "", tweet.text)
        longtext.append(tmp)

    for tweet in public_tweets2:
        tmp = re.sub(r"http\S+", "", tweet.text)
        longtext.append(tmp)
        tweets = longtext.copy()
    return tweets
tweets = newsTweets(tweets)

for i in range(len(tweets)):
    print(i," ",tweets[i])



news_articles = pd.read_json("News_Category_Dataset_v2.json", lines = True)
news_articles.info()
news_articles.head()

print("The total number category:>",news_articles['category'].nunique())
category=news_articles['category'].value_counts()
print(category)



news_articles = news_articles[news_articles['date'] >= pd.Timestamp(2015,1,1)]
news_articles.shape
news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split())>5)]
print("Total number of articles after removal of headlines with short title:", news_articles.shape[0])
news_articles.sort_values('headline',inplace=True, ascending=False)
duplicated_articles_series = news_articles.duplicated('headline', keep = False)
news_articles = news_articles[~duplicated_articles_series]
print("Total number of articles after removing duplicates:", news_articles.shape[0])
news_articles.isna().sum()
print("Total number of articles : ", news_articles.shape[0])
print("Total number of authors : ", news_articles["authors"].nunique())
print("Total number of unqiue categories : ", news_articles["category"].nunique())


print("The total number category:>",news_articles['category'].nunique())
category=news_articles['category'].value_counts()
print(category)

news_articles.index = range(news_articles.shape[0])
news_articles["day and month"] = news_articles["date"].dt.strftime("%a") + "_" + news_articles["date"].dt.strftime("%b")
news_articles_temp = news_articles.copy()
print(news_articles_temp.columns )



stop_words = set(stopwords.words('english'))

for i in range(len(news_articles_temp["headline"])):
    string = ""
    for word in news_articles_temp["headline"][i].split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if not word in stop_words:
          string += word + " "
    news_articles_temp.at[i,"headline"] = string.strip()


lemmatizer = WordNetLemmatizer()

for i in range(len(news_articles_temp["headline"])):
    string = ""
    for w in word_tokenize(news_articles_temp["headline"][i]):
        string += lemmatizer.lemmatize(w,pos = "v") + " "
    news_articles_temp.at[i, "headline"] = string.strip()


pickle.format_version
loaded_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


vocabulary = loaded_model.vocab
w2v_headline = []
for i in news_articles_temp['headline']:
    w2Vec_word = np.zeros(300, dtype="float32")
    for word in i.split():
        if word in vocabulary:
            w2Vec_word = np.add(w2Vec_word, loaded_model[word])
    w2Vec_word = np.divide(w2Vec_word, len(i.split()))
    w2v_headline.append(w2Vec_word)
w2v_headline = np.array(w2v_headline)



def twitter_file():
    tweetsTemp = tweets.copy()
    stop_words = set(stopwords.words('english'))

    for i in range(len(tweets)):
        string = ""
        for word in tweets[i].split():
            word = ("".join(e for e in word if e.isalnum()))
            word = word.lower()
            if not word in stop_words:
                string += word + " "
        tweetsTemp[i] = string.strip()

    lemmatizer = WordNetLemmatizer()

    for i in range(len(tweets)):
        string = ""
        for w in word_tokenize(tweets[i]):
            string += lemmatizer.lemmatize(w, pos="v") + " "
        tweetsTemp[i] = string.strip()

    tweets_vectorizer = CountVectorizer()
    tweets_features = tweets_vectorizer.fit_transform(tweetsTemp)

    tweets_features.get_shape()

    w2v_tweets = []
    for i in tweetsTemp:
        w2Vec_word = np.zeros(300, dtype="float32")
        for word in i.split():
            if word in vocabulary:
                w2Vec_word = np.add(w2Vec_word, loaded_model[word])
        w2Vec_word = np.divide(w2Vec_word, len(i.split()))
        w2v_tweets.append(w2Vec_word)
    w2v_tweets = np.array(w2v_tweets
                          )
    return w2v_tweets

w2v_tweets_tmp = twitter_file()
def avg_w2v_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(w2v_headline, w2v_tweets_tmp[row_index].reshape(1,-1))
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
                'Category': news_articles['category'][indices].values,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})


    return df.iloc[1:,[1,2]]


def categoryCalculator(x):
    dizi = []
    for i in x['Category']:
        dizi.append(i)
    max= 0
    result = ""
    for i in dizi:
        maxTmp = dizi.count(i)
        if maxTmp>max:
            max = maxTmp
            result = i
    return result

Categories = []
print ("      News Tweets                                                                                 Recomended Category")
for i in range(len(w2v_tweets_tmp)):
    Categories.append(categoryCalculator(avg_w2v_based_model(i, 20)))
    print(i," ",tweets[i], "  ", Categories[i])



def avg_w2v_based_model_test(row_index, num_similar_items):
    couple_dist = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
                'Category': news_articles['category'][indices].values,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})

    return df.iloc[1:,[1,2]]
test = []

testCount = 0
for i in range(0,100):
    index = random.randint(0, 97000)
    testCategory = categoryCalculator(avg_w2v_based_model_test(index, 25))

    if testCategory.strip(" ").lower() == news_articles['category'][index].strip(" ").lower():
        testCount+=1

print("Total news number : 97000" )
print("Total number of news selected for category calculating : 25")
print("Total random selected news ( for calculating success rate ) : 100")
print("Success Rate is : %",testCount)
