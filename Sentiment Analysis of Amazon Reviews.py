#!/usr/bin/env python
# coding: utf-8

# In[48]:


get_ipython().system('pip install textblob')
# TextBlob is a Python library that provides a simple API for diving into 
# common natural language processing (NLP) tasks such as part-of-speech tagging, 
# noun phrase extraction, sentiment analysis, classification, translation, and more


# In[50]:


get_ipython().system('pip install wordcloud')
# WordCloud object is used to generate a word cloud from the input text, and the word cloud is displayed using Matplotlib


# In[51]:


get_ipython().system('pip install cufflinks')
# Cufflinks is a Python library that connects the Pandas data frame with Plotly, 
# allowing you to create interactive visualizations directly from data


# In[52]:


import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import init_notebook_mode, iplot 
init_notebook_mode(connected=True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.max_columns', None)


# In[54]:


# Provide the file path as a string, using forward slashes or escaping backslashes
file_path = r'C:\Users\mouni\Desktop\amazon.csv'  # Use 'r' to create a raw string

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)


# In[55]:


df.head()


# In[56]:


df = df.sort_values ("wilson_lower_bound", ascending = False) 
df.drop("Unnamed: 0", inplace=True, axis=1)
df.head()


# In[57]:


import numpy as np
import pandas as pd

def missing_values_analysis(df):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=True)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df


def check_dataframe(df, head=5, tail=5):
    print("SHAPE".center(82, '~'))
    print('Rows: {}'.format(df.shape[0]))
    print('Columns: {}'.format(df.shape[1]))
    print("TYPES".center(82, '~'))
    print(df.dtypes)
    print("".center(82, '~'))
    print("DUPLICATED VALUES".center(83, '~'))
    print(missing_values_analysis(df))
    print(df.duplicated().sum())
    print("QUANTILES".center(82, '~'))
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Assuming you have a DataFrame named 'df'
check_dataframe(df)


# In[58]:


# create a summary of the number of unique classes in each column of the DataFrame
import pandas as pd

def check_class(dataframe):
    nunique_df = pd.DataFrame({
        'Variable': dataframe.columns,
        'Classes': [dataframe[i].nunique() for i in dataframe.columns]
    })
    nunique_df = nunique_df.sort_values('Classes', ascending=False)
    nunique_df = nunique_df.reset_index(drop=True)
    return nunique_df

# Assuming you have a DataFrame named 'df'
result = check_class(df)
print(result)


# In[59]:


constraints = ['red', 'blue', 'green', 'yellow', 'purple']
def categorical_variable_summary (df, column_name):
    fig= make_subplots (rows = 1, cols =2,
                        subplot_titles=('Countplot', 'Percentage'), 
                        specs=[[{"type": "xy"}, {"type": "domain"}]])
    
    
    fig.add_trace(go. Bar( y=df[column_name].value_counts().values.tolist(),
                          x = [str(i) for i in df [column_name].value_counts().index], text=df[column_name].value_counts().values.tolist(),
                          textfont = dict(size=14),
                          name = column_name,
                          textposition = 'auto',
                          showlegend=False,
                          marker=dict(color=constraints,
                                      line= dict(color="#DBE6EC",
                                                 width =1))),
                  row = 1, col = 1)
    fig.add_trace(go. Pie (labels = df [column_name].value_counts().keys(),
                           values= df[column_name].value_counts().values,
                           textfont = dict(size= 20),
                           textposition='auto',
                           showlegend= False,
                           name =column_name,
                           marker = dict(colors = constraints)),
                  row = 1, col =  2)
    fig.update_layout (title={'text': column_name,
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                              template = 'plotly_white')
    iplot(fig)


# In[60]:


categorical_variable_summary(df, 'overall')


# In[61]:


df.reviewText.head()


# In[62]:


review_example=df.reviewText[317]
review_example


# In[63]:


# split it into words.
import re

# Convert to lowercase and split into words while removing non-alphabet characters
words = re.findall(r'\b\w+\b', review_example.lower())

# Now 'words' contains a list of lowercase words without punctuation and spaces
print(words)


# In[64]:


review_example=re.sub("[^a-zA-Z]", '',review_example)
review_example=review_example.split()
review_example                     


# In[65]:


df.reviewText[2031].split()


# In[71]:


rt = lambda x: re.sub("[^a-zA-Z]",'', str(x))
# Apply the lambda function to the `reviewText` column
df["reviewText"] = df["reviewText"].map(rt)
# Convert the `reviewText` column to lowercase
df["reviewText"] = df["reviewText"].str.lower()


# Print the first few rows of the DataFrame
df.head()


# **Sentiment analysis**

# In[67]:


get_ipython().system('pip install vaderSentiment')


# In[92]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from textblob import TextBlob
# import pandas as pd

# Assuming you have a DataFrame 'df' with a column 'reviewText'

# Apply sentiment analysis and create new columns for 'polarity' and 'subjectivity'
df[['polarity','subjectivity']] = df['reviewText'].apply(lambda Text:pd.Series(TextBlob(Text).sentiment))
# df['subjectivity'] = "Not applicable"  # You can set this to any default value

for index, row in df['reviewText'].iteritems():
    
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    
    
    neg=score['neg']
    neu=score['neu']
    pos = score['pos']
    if neg>pos:
        df.loc[index, 'sentiment']="Negative"
    elif pos > neg:
        df.loc[index,'sentiment']="Positive"
    else:
        df.loc[index,'sentiment'] ="Neutral"


# In[95]:


df[df['sentiment']=='Positive'].sort_values("wilson_lower_bound",ascending= False).head(5)


# In[96]:


categorical_variable_summary(df,'sentiment')


# In[ ]:





# In[ ]:





# In[ ]:




