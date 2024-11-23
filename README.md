## Udacity Disaster Response Pipeline Project
This project analyzes disaster data from Appen (formerly Figure 8) and uses that to build a model that classifies messages into 36 different categories. The model is deployed as a flask web app which accepts input (disaster messages) from a user and returns the appropriate categories for that message.


## Table of Contents

1. [File Structure](#filestructure)
2. [Execution](#execution)
3. [Results](#results)
4. [Acknowledgements](#acknowledgements)


## File Structure<a name="filestructure"></a>
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


## Execution<a name="execution"></a>
To run files in this project,
- create a DisasterResponse Folder then clone the repo files into it.
- From the command line, execute the following commands (ensure you are in the DisasterResponse folder dir):
  - python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 
  - python models/train_classifier.py data/DisasterResponse.db models/final_model.pkl
  - python app/run.py 

## Results<a name="results"></a>
![webapp](https://github.com/user-attachments/assets/4616d951-19de-42f6-8f99-bfc82daf871f)
![Barchart](https://github.com/user-attachments/assets/7f1b92b0-752f-4e24-ad9a-4ec1d9add9ca)
![Wordcloud](https://github.com/user-attachments/assets/e82ba19a-9fa6-45b9-a5ba-53de59ac3052)
![modeloutput](https://github.com/user-attachments/assets/e2f065bf-0426-4ca5-9446-d7ad04a5c95e)


## Acknowledgements<a name="acknowledgements"></a>
Author: Kike Obayemi <br>
Udacity: for the training and guidance <br>
[Appen](https://www.appen.com/): for the datasets
