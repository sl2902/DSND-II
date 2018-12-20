# Disaster Response Pipeline Project
In this project we build an application that classifies disaster response messages into 1 or more categories from the 36 categories
available; it is a multiclass multilabel classifier project, which involves building an ETL pipeline to process the raw data, and then building
and ML pipeline to train the classifier and fine tune it

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db table_name`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl `

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
4. If you are running the app from your local machine, then change the host value in the `run.py` script, like so `host=127.0.0.1`. 
   Go to http://localhost:3001/
   
### File description:
`app/run.py` - Deploys the app either locally or in the workspace<br/>
`data/process_data.py` - ETL pipeline to parse the disaster messages and categories data, clean the dataframe, <br/>
                          and write the dataframe to a sqlite database<br/>
`model/train_classifier.py` - ML pipeline to train, tune, evalute, and save the pickled classifier<br/>
`data/disaster_messages.csv` - CSV file containing the disaster messages<br/>
`data/disaster_categories.csv` - CSV file containing the 36 disaster classes<br/>
`app/template/*.html` - Supporting files for the app, including visualization
`
