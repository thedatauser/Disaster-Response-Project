import sys
import pandas as pd
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load data from the SQLite database and split it into features and target variables.

    Returns:
    X (pd.Series): Series containing the messages.
    y (pd.DataFrame): DataFrame containing the categories for each message.
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesCategory', engine)
    X= df['message']
    y = df.iloc[:,4:]
    #y= df.drop(columns=['id','message', 'original', 'genre'], axis=1)
    return X,y, y.columns


def tokenize(text):
    """
    Function to preprocess text

    INPUT: 
    text(str): text to be preprocessed

    OUTPUT:
    clean tokens (str): preprocessed text
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with a specified classifier.

    Args:
    None.

    Returns:
    best_pipeline (Pipeline): A scikit-learn Pipeline object that has been trained 
                              with the best found parameters.
    """
    pipeline = Pipeline([        
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    parameters = {
            'clf__estimator__n_estimators': [50,100,200],        
            'clf__estimator__min_samples_split': [2, 5, 10]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring='accuracy', n_jobs=-1)

    return cv
    
    
    
def evaluate_model(model, X_test, y_test, category_names):

    """
    Evaluate a trained machine learning model on a test dataset.

    This function retrieves the best parameters from a trained model, makes predictions 
    on the test set, and prints the classification report (including f1 score, precision, 
    and recall) for each output category.

    Args:
    model (GridSearchCV): The trained model with GridSearchCV.
    X_test (DataFrame or array-like): The test features.
    y_test (DataFrame or array-like): The true labels for the test set.
    category_names (list of str): The names of the output categories.

    Returns:
    None.
    """
        
    # Get the best parameters and best score
    best_params = model.best_params_    
    print("Best Parameters:", best_params)
    
    #Make predictions with the best estimator
    best_pipeline = model.best_estimator_

    # Predict on the test set using the estimator
    y_pred = best_pipeline.predict(X_test)

    #Report f1 score, precision and recall for each output category
    for i, column in enumerate(category_names):
        print(f"Classification Report for {column}:")
        print(classification_report(y_test[column], y_pred[:, i]))


def save_model(model, model_filepath):
    # Save the model to a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

print("Model saved as final_model.pkl")


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