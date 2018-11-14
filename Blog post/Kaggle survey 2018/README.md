<h1>Overview</h1></br>

[Kaggle](https://www.kaggle.com/kaggle/kaggle-survey-2018) recently concluded its second annual Machine Learning and Data Science Survey 2018. The survey posed 395 questions, some of which had sub-parts; the questions ranged from the usual demographic related queries to activities performed, Cloud computing tools, Big data tools, types of datasets, Machine learning frameworks and, other challenging questions.
As an aspiring data scientist, I was curious to find out the daily grind the average respondent undergoes at work or at school, and see how I compared with the 23859 respondents who participated from 147 countries and territories.
</br>
<h1>Libraries used</h1></br>

```
Python 3.5.3
Pandas 0.23.4
Numpy 1.15.2
Matplotlib 2.2.2
Seaborn 0.9.0

D3.js 3.2.7
```

<h1>Directories</h1></br>

```
*d3_input* - Input directory that contains the input csv files for creating Sankey charts
*data* - Contains the raw Kaggle survey data; there are three files: 1) multipleChoiceResponses.csv 2) freeFormResponses.csv 3) SurveySchema.csv
*html* - Contains the d3.js file, html file for creating the Sankey charts, including png assets used in the Sankey chart
*img* - Contains the output png of various charts used in the Blog to answer the various queries that I had about the survey
*sankey_kaggle_demographic_insight.py* - Python script used to create the JSON file for the Sankey chart
*Kaggle survey EDA 2018.ipynb* - Jupyter notebook containing the code used to analyze and generate the various charts used to answer the
queries that I had
```

<h1>Summary</h1></br>

+ 23589 respondents participated from 147 questions and answered about 395 questions (not necessarily all)</br>

+ The most common respondent is male, is about 25–29 years of age, is either from the USA or from one of the BRIC countries, is working   as either a student or a Data scientist, and who earns roughly under $30K per annum</br>

+ The respondents holds a Bachelor’s degree or higher in one of the STEM disciplines, has 3 years or less of work experience, with 2       years or less of coding experience, and spends anywhere from 1%-25% to 50%-74% of his/her time in coding</br>

+ Of the 4137 professionally employed Data scientsts , 77% of the respondents who were asked whether they see themselves as Data           scientists or not were confident of being seen as Data scientists; this community uses Python as its weapon of choice. Some of the       datasets commonly worked on are: Geospatial data, Audio data, Tabular data, etc.</br>

+ With respect to ML frameworks, Scikit-Learn is popular with this community; the community spends a little less than 25% of its time on   various data munging and modelling tasks such as gathering data, cleaning data, visualising data, and model building and, less than     12% of its time on deploying models to production and/or finding and communicating insights to stakeholders</br>

+ [Are you a Data Scientist?](https://medium.com/@laxmsun/are-you-a-data-scientist-4ca03d00a316)
<h1>Acknowledgement</h1></br>

[Kaggle](https://www.kaggle.com/kaggle/kaggle-survey-2018) - For providing the data

[Sankey](http://bl.ocks.org/d3noob/c9b90689c1438f57d649) - For providing boiler plate code

[Chord](https://www.delimited.io/blog/2013/12/8/chord-diagrams-in-d3) - For providing boiler plate code





