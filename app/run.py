import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Load data
database_filepath = 'data/DisasterResponse.db'
engine = create_engine(f'sqlite:///{database_filepath}')
df = pd.read_sql_table('MessagesCategory', con=engine)

# load model
model = joblib.load("models/rfmodel.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Count the number of 1s in each column
    category_columns = df.iloc[:, -36:]
    counts = category_columns.apply(lambda x: (x == 1).sum())

      # Generate a word cloud from the disaster messages
    disaster_text = ' '.join(df['message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(disaster_text)

    # Save the word cloud to a buffer
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # create visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x = counts.index,
                    y = counts.values
                )
            ],

            'layout': {
                'title': 'Count of Messages in Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Categories"
                }
                }
        },
       
        {          
                'data': [
                { 
                'type': 'image',
                'source': 'data:image/png;base64,{}'.format(image_base64),
                'xref': 'x',
                'yref': 'y',
                'x': 0,
                'y': 0,
                'sizex': 1,
                'sizey': 1,
                'sizing': 'stretch',
                'opacity': 1,
                'layer': 'below'
                }
                ],
            
                'layout': {
                    'title': 'Word Cloud of Disaster Messages'

                }
        }        
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, image_base64=image_base64)


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