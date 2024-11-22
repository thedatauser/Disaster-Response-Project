# Project Description
This project analyzes disaster data from Appen(opens in a new tab) (formerly Figure 8) and uses that to build a model that classifies disaster messages into 36 categories

# File Structure
- myapp
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
