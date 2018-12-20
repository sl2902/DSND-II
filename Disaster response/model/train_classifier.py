import sys
# import libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVC

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sqlite3
from sqlalchemy import create_engine
from warnings import filterwarnings
import pickle
filterwarnings('ignore')
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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


# not working as expected. the length of the series is not in 
# sync with the input X
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.c = 0
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            self.c += 1
            if len(pos_tags) == 0:
                return False
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)  
    

def load_data(db, table_name):
    """
    Load the database and convert it to a
    dataframe
    Args: db - A sqlite database
          table_name - A sqlite table
    Returns - An input array of messages and output 
              array of label, including label names
    """
    engine = create_engine('sqlite:///{}'.format(db))
    table_name = table_name
    df = pd.read_sql_table(table_name, engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    
    return X, Y, df.columns[4:]


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
    stop_words = stopwords.words("english")
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok.lower() not in stop_words]

    return clean_tokens


def create_pipeline(clf, tokenize):
    """
    Create a pipeline of a sequence of actions
    to perform on the input
    1) CountVectorizer()
    2) TfIdfTransformer()
    3) MultiOutputClassifier() or OneVsRestClasssifier()
    
    Args: clf - Classifier object
          tokenize - A function to normalize the input
                      text
    Returns - A pipeline object
    """
    pipeline = Pipeline([
                       ('union', FeatureUnion([
                           ('text_pipeline', Pipeline([
                                ('vect', CountVectorizer(
                                                ngram_range=(1, 2),
                                                max_df=0.93,
                                                min_df=7,
                                                tokenizer=tokenize
                                )),
                                ('tfidf', TfidfTransformer()),
                               
                           ])),
#                             ('avg_length', AvgResponseLength()),
#                             ('starting_verb', StartingVerbExtractor())
                       ])),
                        
                       ('clf', OneVsRestClassifier(clf))
                       ])
    
    return pipeline


def split_data(X, y, test_size=0.2):
    """
    Split the data into train and validation sets
    Args: X - An input array of messages
          y - A array of labels
    Returns X_train - An array of training set of messages
            X_valid - An array of validation set of message
            y_train - An array of training labels
            y_valid - An array of validation labels
    """
    return train_test_split(X, y, test_size=test_size, random_state=2018)
    

def display_report(y_test, preds, labels=None):
    """
    Display a classification report showing
    f1_score, precision and recall
    Args: y_test- An array of test labels
          preds - An array of predicted labels
          labels - A list of label names
    Returns None      
    """
    print(classification_report(y_test, preds, target_names=labels))
    
    
def build_model(model):
    """
    Build model by creating a pipeline
    of transformations and classifier
    with Grid Search
    Args: model - A classifier object
    Returns gs - A Grid Search object
    """
#    X_train, X_valid, y_train, y_valid = split_data(X, y, test_size=test_size)
#     pipeline = create_pipeline(model, tokenize)
#     pipeline.fit(X_train, y_train)
    
    # create pipeline
    gs_pipeline = create_pipeline(model, tokenize)
    # update pipeline with user-defined feature engineering
    gs_pipeline.steps[0][1].get_params().update({
                       'transformer_list': gs_pipeline.steps[0][1].get_params()['transformer_list'].extend([('avg_length', AvgResponseLength())]),
                       'avg_length': AvgResponseLength()})
    parameters = {
                'union__text_pipeline__vect__max_features': [9000, 12000],
#                'clf__estimator__n_estimators':[10, 100, 200],
#                'clf__estimator__min_samples_split': [2, 5]
}

    # parameters = {
    #                 'union__text_pipeline__vect__max_features': [9000, 12000],
    #                 'clf__estimator__loss': ['log', 'modified_huber'],
    #                 'clf__estimator__penalty': ['l2', 'elasticnet']
    # }
    gs = GridSearchCV(gs_pipeline, param_grid=parameters, n_jobs=-1, 
                  scoring=make_scorer(f1_score, average='weighted'),
                 cv=3, verbose=4)
    return gs


def evaluate_model(model, X_test, y_test, labels=None):
    """
    Make predictions on the test data
    and then display the f1-score, precision,
    and recall for every class
    Args: model - A classifier object
          X_test - An array of messages
          y_test - An array of labels
          labels - A list of classes
    Returns None
    """
    preds = model.predict(X_test)
    display_report(y_test, preds, labels=labels)


def save_model(model, file):
    """
    Pickle the model
    Args: model - Model object
          file - name of pickled object
    Returns None      
    """
    pickle.dump(model, open(file, 'wb'))


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, table_name = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, labels = load_data(database_filepath, table_name)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.1)
        
        print('Building model...')
        rf = RandomForestClassifier(min_samples_split=5,
                            n_estimators=20,
                            random_state=2018)
        # gb = GradientBoostingClassifier(random_state=2018)
        # svc = SVC(random_state=2018)
        # lg = LogisticRegression(random_state=2018)
        # sg = SGDClassifier(random_state=2018)
        # mb = MultinomialNB()
        model = build_model(rf)
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, labels)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()