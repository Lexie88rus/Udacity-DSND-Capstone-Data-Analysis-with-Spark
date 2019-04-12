#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  10 2019

@author: aleksandraastakhova

Script which runs flask web application

"""
# imports

from flask import Flask
from flask import render_template, request

from datetime import datetime

import findspark
findspark.init() # find spark

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import PipelineModel

app = Flask(__name__)

# get spark session
spark = SparkSession.builder \
    .master("local") \
    .appName("Sparkify") \
    .getOrCreate()

# load model
model = PipelineModel.load('../model')

# index webpage receives user input for the model
@app.route('/')
@app.route('/index')
def index():
    
    # render web page
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    
    # get parameters from the form
    gender = request.args.get('gender', '') 
    avgEvents = request.args.get('avgEvents', 0) 
    avgSongs = request.args.get('avgSongs', 0)
    thumbsup = request.args.get('thumbsup', 0)
    thumbsdown = request.args.get('thumbsdown', 0)
    add_friend = request.args.get('add_friend', 0)
    reg_date = request.args.get('reg_date', '') # 2018-08-19
    level = request.args.get('level', '')
    location = request.args.get('location', '')
    
    # calculate number of days since the 1st event for the user
    days_active = (datetime.now() - datetime.strptime(reg_date, '%Y-%m-%d')).days
    
    # encode gender values
    if gender == 'male':
        gender = 'M'
    else:
        gender = 'F'
   
    # get spark context
    sc = SparkContext.getOrCreate()
    
    # create spark dataframe to predict customer churn using the model
    df = sc.parallelize([[gender, level, days_active, location, avgSongs, avgEvents, thumbsup, thumbsdown, add_friend]]).\
    toDF(["gender", "last_level", "days_active", "last_state", "avg_songs", "avg_events" , "thumbs_up", "thumbs_down", "addfriend"])
    
    # set correct data types
    df = df.withColumn("days_active", df["days_active"].cast(IntegerType()))
    df = df.withColumn("avg_songs", df["avg_songs"].cast(DoubleType()))
    df = df.withColumn("avg_events", df["avg_events"].cast(DoubleType()))
    df = df.withColumn("thumbs_up", df["thumbs_up"].cast(IntegerType()))
    df = df.withColumn("thumbs_down", df["thumbs_down"].cast(IntegerType()))
    df = df.withColumn("addfriend", df["addfriend"].cast(IntegerType()))
 
    # predict using the model
    pred = model.transform(df)
    
    if pred.count() == 0:
        # if model failed to predict churn then return -1
        prediction = -1
    else:
        # get prediction (1 = churn, 0 = stay)
        prediction = pred.select(pred.prediction).collect()[0][0]
    
    # print out prediction to the app console
    print("Prediction for the customer is {prediction}.".format(prediction = prediction))
    
    # render the go.html passing prediction resuls
    return render_template(
        'go.html',
        result = prediction
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()