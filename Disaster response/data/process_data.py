import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages into a dataframe and
    categories into another dataframe and 
    merge them
    Args: message_filepath - A file containing the messages
          categories_filepath - A file containing the categories
    Returns - A dataframe with both messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Split the categories into separate columns,
    remove any duplicates and fix any non binary
    values against the class labels
    Args: df - A dataframe containing the messages
          and categories
    Returns - A cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = [category[:-2] for category in categories.loc[0].tolist()]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # extract the number from the class and convert to int
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1).astype(int)
    # fix the invalid numbers against the class; there should be only
    # 1 or 0 against the class label
    cat_fix = []    
    for category in category_colnames:
        mask = (categories[category]) > 1 | (categories[category] < 0)
        if len(categories[mask][category]) > 0:
            cat_fix.append(category)
    for category in cat_fix:
        categories[category] = categories[category].apply(lambda x: 1 if(x != 1 or x != 0) else x)
    
    # drop the original categories column and concatenate
    # the new categories to the df dataframe
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # drop the duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename, table_name):
    """
    Save the dataframe into a Sqlite
    database
    Args: df - A dataframe containing the messages
          and categories
          database_filename - A string specifying the db filename
          table_name - A string specifying the table name
    Returns None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath, table_name = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()