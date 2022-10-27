#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[3]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/flipkart_reviews.csv')


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


import nltk
import re
nltk.download('stopwords')
stemmer=nltk.SnowballStemmer('english')
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


# In[7]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[8]:


data['Review']=data['Review'].apply(clean)


# # Sentiment Analysis of Flipkart reviews

# In[9]:


ratings=data['Rating'].value_counts()


# In[10]:


numbers= ratings.index
quantity=ratings.values


# In[11]:


import plotly.express as px
figure=px.pie(data,values=quantity,names=numbers,hole=0.5)
figure.show()


# ## 60% of the reviews are 5/5 ratings to the product they buy at flipkart

# In[12]:


text=" ".join(i for i in data.Review)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color='white').generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ## Analyzing the sentiment by adding three columns

# In[16]:


nltk.download('vader_lexicon')
sentiments=SentimentIntensityAnalyzer()
data['Positive']=[sentiments.polarity_scores(i)['pos'] for i in data['Review']]
data['Negative']=[sentiments.polarity_scores(i)['neg'] for i in data['Review']]
data['Neutral']=[sentiments.polarity_scores(i)['neu'] for i in data['Review']]
data=data[["Review","Positive","Negative","Neutral"]]
print(data.head())


# In[18]:


#Now lets see how most of the reviewers think about the product and services of flipkart
x=sum(data['Positive'])
y=sum(data['Negative'])
z=sum(data['Neutral'])


# In[19]:


def sentiment_score(a,b,c):
    if (a>b) and (a>c):
        print('Positive😊')
    if (b>a) and (b>c):
        print('Negative😠')
    else:
        print('Neutral🙂')


# In[20]:


sentiment_score(x,y,z)


# In[21]:


print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)

