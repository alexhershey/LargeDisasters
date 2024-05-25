## Train Classifier.py

import sys
# import libraries
import pandas as pd
import numpy as np
import sqlite3
import nltk
import seaborn as sns

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report 
from sklearn import ensemble


import pickle

database_filepath = ' data/DisasterResponse.db' 
# 'sqlite:///DisasterResponse.db'

def load_data(database_filepath):
    # load data from database
    
#     engine = create_engine('sqlite:///DisasterResponse.db')
    engine = create_engine(database_filepath)
#     engine = create_engine(database_filepath)
    #InsertTableName
    df = pd.read_sql_table('InsertTableName', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    #child_alone has no values and related is multi-class
    Y.drop(['child_alone','related'], axis = 1, inplace = True)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    Purpose  :Tokenize and transform text
    Arguments:
                -text: Accepts a pandas series of dataframe  
    Outputs  : Returns tokenized text
    
    '''
    text         = text.str.replace('[^a-zA-Z]'," ")
    tv           = TfidfVectorizer(max_features= 100, stop_words = 'english')
    tv_transform = tv.fit_transform(text)
    tv_df        = pd.DataFrame( tv_transform.toarray()
                                , columns = tv.get_feature_names()
                                )
    return tv_df

# X,Y,category_names
def build_model():
    
    X, Y, category_names = load_data(database_filepath)
    #preprocess the variables
    X = tokenize(X)
    X = StandardScaler.fit_transform(X)
    X_train, X_test, y_train,y_test = train_test_split(X,Y, test_size = 0.3, random_state= 42)
    
    lr_cv = MultiOutputClassifier(estimator = LogisticRegression(C = 1.0, class_weight=None
                                             , dual=False
                                             , fit_intercept=True
                                             , max_iter=100
                                             , multi_class='ovr'
                                             , n_jobs=1
                                             , penalty='l2'
                                             ,  solver ='liblinear'
                                            )
                             )
    lr_cv.fit(X_train,y_train)


    
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pipe.predict(X_test)
    report = classification_report( y_test
                               , ypred
                               , target_names = category_names
                               )
#     print(report)


def save_model(model, model_filepath):
    with open (model_filepath, 'wb') as f:
        pickle.dump(model,f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()