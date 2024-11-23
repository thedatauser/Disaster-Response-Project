import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """
    Load data from CSV files and merge them into a single DataFrame.

    Args:
    messages_filepath (str): Filepath for the messages CSV file.
    categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    pd.DataFrame: Merged DataFrame containing messages and categories.

    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id', how="inner")
    return df


def clean_data(df):
    """
    Clean the merged DataFrame by splitting categories into separate columns,
    converting category values to binary, and removing duplicates.

    Args:
    df (pd.DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    pd.DataFrame: Cleaned DataFrame with separate category columns and no duplicates.
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    row_items = row.values.tolist()

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda x: x[:-2], row_items))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Iterate through the category columns to keep only the last character and convert to numeric

    for column in categories:
        categories[column] = categories[column].apply(lambda x:int(x[-1]))

    # drop the original categories column from `df`
    df= df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df_new = pd.concat([df,categories], axis=1 )

    # drop duplicates
    df_no_duplicates = df_new.drop_duplicates(subset='id')

    # Fill missing values with 0
    df_filled = df_no_duplicates.fillna(0)

    return df_filled


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    Args:
    df (pd.DataFrame): Cleaned DataFrame containing messages and categories.
    database_filename (str): Filename for the SQLite database.

    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesCategory', engine, index=False, if_exists ='replace')




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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