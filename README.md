# Udacity DSND Capstone Data Analysis with Spark
Machine learning on a large dataset using PySpark

## Table of Contents
* [Definition](#definition)
* [Input Data](#input-data)
* [Implementation](#implementation)
* [Repository Structure](#repository-structure)
* [Conclusions](#conclusions)
* [External Libraries](#external-libraries)

## Definition

### Project Overview
Predicting churn rates is a challenging and common problem that data scientists and analysts regularly encounter in any customer-facing business. It is crucial for businesses to identify customers who are about to churn and take action to retain them before it happens.
The goal of this project was to help Sparkify music service retain their customers. In this project, I analyzed Sparkify data, built a machine learning model to predict churn and developed a web application to demonstrate the results.

### Problem Statement
As the goal of the project is to help to retain the customers, the main task of the project is to make a prediction, whether the customer is about to churn. Such a prediction can be made for each customer by a binary classifier model. The following tasks should be completed to create the model:
•	Analyze and preprocess the data to extract features for each customer;
•	Train classifier to predict customer churn;
•	Evaluate the classifier concerning the chosen metric;
•	Build a web application to demonstrate the results.

### Metrics
The initial dataset analysis shows us that the dataset is imbalanced (see section Input Data): there are more than 3 times fewer users, who churned, than other users. That is why I can’t use accuracy (which is the number of correct predictions divided by the total number of predictions) as a metric to evaluate the resulting model. 
In our case, we should care about both types of errors: false negatives and false positives because in case of false negative we can miss the customer who is going to churn and lose the customer and in case of false positive we can have unnecessary costs on retaining the customer who was not going to churn. That is why as a metric to evaluate the model I chose F1 score because it equally considers both the precision and the recall.

## Input Data
As input data I have several datasets, which contain the log of Sparkify music service events:
* medium_sparkify_event_data.json – medium sized dataset.
* mini_sparkify_event_data.json – a tiny subset of the full dataset, which is useful for preliminary data analysis.
Both files contain the following data:

|#| Column | Type | Description |
| --- | --- | --- | --- |
| 1 | userId | string | Unique identifier of the user, the event is related to |
| 2 | artist | string | Name of the artist related to the song related to the event |
| 3 | auth | string | “Logged in” or “Cancelled” |
| 4 | firstName | string | First name of the user |
| 5 | gender | string | Gender of the user, “F” or “M” |
| 6 | itemInSession | bigint | Item in session |
| 7 | lastName | string | Last name of the user |
| 8 | length | double | Length of the song related to the event |
| 9 | level | string | Level of the user’s subscription, “free” or “paid”. User can change the level, so events for the same user can have different levels |
| 10 | location | string | Location of the user at the time of the event |
| 11 | method | string | “GET” or “PUT” |
| 12 | page | string | Type of action: “NextSong”, “Login”, “Thumbs Up” etc. |
| 13 | registration | bigint | Registration |
| 14 | sessionId | bigint | Session id|
| 15 | song | string | Name of the song related to the event |
| 16 | status | bigint | Response status: 200, 404, 307 |
| 17 | ts | bigint | Timestamp of the event |
| 18 | userAgent | string | Agent, which user used for the event, for example, “Mozilla/5.0” |

__[Sample data file](https://github.com/Lexie88rus/Udacity-DSND-Capstone-Data-Analysis-with-Spark/blob/master/data/sample_sparkify_data.json)__.

## Implementation
The input datasets contain massive amounts of data, which can’t be processed on a single machine. That is why I will use Spark clusters to analyze data and predict customer churn. I use PySpark and SparkML libraries to implement the solution.
The implementation of the project consists of two parts:
* Application of machine learning methods to predict churn. This part involves creation of machine learning pipelines, evaluation and tuning of the approach.
* Development of a web application to demonstrate the resulting model.

### Machine Learning Pipelines
Machine learning pipeline for our task consists of the following steps:
1.	Split dataset into train, test, and validation.
2.	Create dummy columns out of categorical columns ‘gender’, ‘last_level’, and ‘last_state’. When using pyspark machine learning library sparkml, this step actually consists of two parts: indexing categorical column and encoding it.
3.	Create a feature vector.
4.	Train the classifier.
The Random Forest Classifier was chosen from the set of other models (Logistic Regression, Gradient-boosted Tree, Naive Bayes) because it demonstrated the best performance in terms of the F1 score (81%).

### Web Application
The web application is implemented with:
* Flask running the back-end,
* Bootstrap controls of front-end.
The web application consists of the following parts:
* Python script [create_model.py](https://github.com/Lexie88rus/Udacity-DSND-Capstone-Data-Analysis-with-Spark/blob/master/model/create_model.py) which builds the machine learning model. This script accepts the path to the dataset and the path where the resulting model should be saved as parameters.
* The machine learning model, which is created by script create_model.py. The application loads the model and uses it to make predictions.
* Python script [run.py](https://github.com/Lexie88rus/Udacity-DSND-Capstone-Data-Analysis-with-Spark/blob/master/app/run.py), which runs the logic of the application and renders web pages. The script loads the model on start and applies it to make predictions out of the data provided by the user on the web page.
* Web page templates [master.html](https://github.com/Lexie88rus/Udacity-DSND-Capstone-Data-Analysis-with-Spark/blob/master/app/templates/master.html) and [go.html](https://github.com/Lexie88rus/Udacity-DSND-Capstone-Data-Analysis-with-Spark/blob/master/app/templates/go.html) of application web pages. Pages use bootstrap controls.
The web application allows the user to enter the information about the customer and then tells whether the customer is about to churn based on this information.
### Demo
Demo of the web application:
![demo](https://github.com/Lexie88rus/Udacity-DSND-Capstone-Data-Analysis-with-Spark/blob/master/demo/demo2.gif)

### Setup Instructions
To run the web application follow steps:
1. Download the repository and install required libraries (see [External Libraries](#external-libraries) section below).
2. From `model` folder run command to build the model:
```
$ python create_model.py ../mini_sparkify_event_data.json ../classifier
```
3. Navigate to `app` folder:
```
$ cd ../app
```
4. Run app:
```
$ python run.py
```
5. Open http://0.0.0.0:3001/ in browser.

## Repository Structure
The repository has the following structure:
```
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
| - static
| |- githublogo.png  # github logo used in the main page
| |- jumbotron.jpg # jumbotron image
|- run.py  # Flask script that runs app

- data
|- sample_sparkify_data.json # sample data file

- demo
|- main.png # the screenshot of the main page
|- result.png # the screenshot of page with prediction result
|- demo1.gif # animation with successful prediction
|- demo2.gif # animation with prediction of churn

- model
|- create_model.py # script, which builds the classifier to predict customer churn

- DSND Capstone Report.pdf # detailed report on the project
- README.md
- Sparkify Medium.ipynb # Python 3 notebook, which contains analysis of medium dataset
- Sparkify Refinement.ipynb # Python 3 notebook, which contains model refinement and conclusion
- Sparkify.ipynb # Python 3 notebook, which contains EDA of small dataset
```

## Conclusions

### Results Summary
The goal of the project is to help the Sparkify service to retain the customers. The solution which I proposed to reach this goal is as follows:
* A large part of the solution is the preprocessing of the initial data. The initial data was in terms of music service events. I transformed it into records in terms of each Sparkify customer. Feature engineering and preprocessing were required to obtain the dataset which is ready for machine learning.
* The second large part of the solution is the machine learning pipeline, which predicts customer churn. I tried several classifiers and compared their F1 scores to choose the best performing solution. I also tuned the hyperparameters with the help of a grid search and cross-validation for the chosen classifier. The final F1 score of the solution is 81%.
* The last part of the solution is the web application which demonstrates the churn prediction. The web application allows the user to enter the information about the customer and then identifies whether this customer is about to churn. 
* All parts of the solution are built in Python using Spark.
The most challenging parts of this journey were the feature engineering and the refinement of the model. In feature engineering it is challenging to propose features, which on one hand will help to predict churn and will not overfit the model on the other. Trying to refine the model I tried out several techniques, but not all of them worked (for example, bucketing of continuous numerical features).

### Improvement
It was quite a journey, but still, a lot can be done to improve the proposed solution:
* Introduce the different approach for the customer churn prediction. We can try to predict the churn event depending on the sequence of preceding events.
* Use larger dataset for machine learning. Having more data could raise the robustness of the model.
* Try model stacking to raise the accuracy of the prediction. The ensemble of different classifiers often makes a more accurate resulting model.

## External Libraries
1. [PySpark](https://spark.apache.org/docs/2.2.1/api/python/index.html#), SparkML
2. [Flask](http://flask.pocoo.org/docs/1.0/installation/)
3. [findspark](https://github.com/minrk/findspark)
