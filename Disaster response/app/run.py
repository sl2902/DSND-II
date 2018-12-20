import json
import plotly
import pandas as pd
import re
import numpy as np
import random

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from collections import Counter


app = Flask(__name__)

# http://michelleful.github.io/code-blog/2015/06/20/pipelines/
class AvgResponseLength(BaseEstimator, TransformerMixin):
    """ Compute the average length of the resposne"""
    def __init__(self):
        pass
    
    def avg_word_length(self, text):
        return np.mean([len(word.strip()) if len(word.strip()) != 0 else 0 for word in text.split()])
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        return pd.DataFrame(pd.Series(x).apply(self.avg_word_length)).fillna(0)
    
def word_count(X, top_words=50):
    """
    Compute the word frequency in the messages
    
    Args: X An input array of messages
          top_words - An integer specifying the number
          of words
    Return - A list of tuples of word counts
    """
    newX = []
    for text in X:
        newX += tokenize(text)
    setwords = [word for word in (' '.join(newX)).split()]
    return Counter(setwords).most_common(top_words)

def tokenize(text):
    """
    Normalize the input text. Steps performed
    1) Remove punctuations
    1) Tokenize text
    2) Lemmatize text
    3) Strip whitespaces
    4) Convert to lower
    5) Remove stopwords
    
    Args: text - String
    Return - A list of cleaned tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok.lower() not in stop_words]

    return clean_tokens
   
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

# load data
engine = create_engine('sqlite:///../data/responses.db')
df = pd.read_sql_table('emergency_messages', engine)
top_word_freq = dict(word_count(df['message'].values))

# load model
model = joblib.load("../models/rf_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    top_words = list(top_word_freq.keys())
    frequencies = np.array(list(top_word_freq.values())) / 100
    weights = frequencies
    colors =  [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for _ in frequencies]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # https://community.plot.ly/t/wordcloud-in-dash/11407/4
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=random.choices(range(len(frequencies)), k=len(frequencies)),
                    y=random.choices(range(len(frequencies)), k=len(frequencies)),
                    mode='text',
                    text=top_words,
                    marker= {
                       'opacity': 0.3
                    },
                    hovertext=['{0}-{1}'.format(word, freq) for word, freq in zip(top_words, list(top_word_freq.values()))],
                    textfont= {
                        'size': weights,
                        'color': colors
                    }
                )
            ],
            
            'layout': {
                'title': 'Top 50 words',
                'xaxis': {
                        'showgrid': False, 
                        'showticklabels': False, 
                        'zeroline': False
                 },
                'yaxis': {
                        'showgrid': False, 
                        'showticklabels': False, 
                        'zeroline': False
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()