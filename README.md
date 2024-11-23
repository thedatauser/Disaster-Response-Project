# Project Description
This project analyzes disaster data from Appen (formerly Figure 8) and uses that to build a model that classifies messages into 36 different categories. The model is deployed as a flask web app which accepts input (disaster messages) from a user and returns the appropriate categories for that message.

# File Structure
There are 3 python files to be executed. 
-The first file pre-process the data and saves to a sqlite database. 
-The second file loads the data, build a classification model, evaluates the model and saves into a pickle file
-The third file allows you to load the web app which contains the classification model and shows some analysis of the data.

The file structure and the commands to run are shown below

- app
  - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_messages.csv  # data to process. contains disaster messages
  - disaster_categories.csv  # data to process. contains categories of disaster messages
  - process_data.py # file that performs etl and saves clean data to a database
  - DisasterResponse.db  # database to save clean data to

- models
  - train_classifier.py #the file that create the ML pipeline for classification of disaster messages
  - final_model.pkl  # saved model 

- README.md 


# Execution
To run files in this project,
- create a DisasterResponse Folder then clone the repo files into it.
- From the command line, execute the following commands (ensure you are in the DisasterResponse folder dir):
  - python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 
  - python models/train_classifier.py data/DisasterResponse.db models/final_model.pkl
  - python app/run.py 

# Author and Acknowledgements
- Author: Kikelomo Obayemi
- Acknowledgements: Appen, Udacity
