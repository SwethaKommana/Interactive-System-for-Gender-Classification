from flask import Flask, render_template, request, session
import os
import json
import pandas as pd
import numpy as np
import csv
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras import backend
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils
from sklearn import preprocessing
from numpy import array
import matplotlib.pyplot as plt
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import load_model
from csv import writer

app = Flask(__name__,template_folder='templateFiles',static_folder='staticFiles')
app.secret_key = 'This is your secret key to utilize session in flask'
@app.route('/')
def index():
    return render_template('home.html')
@app.route('/Record',methods=("POST","GET"))
def ClickRecord():
    if request.method=='POST':
        return render_template('home.html')
@app.route('/getValue',methods=("POST","GET"))
def getValues():
    if request.method=='POST':
        gen = request.form.get('gender')
        print(gen)
        if(gen==" "):
            gen = "No Input Given"
        session['gender'] = gen
        return render_template('home.html')
@app.route('/upload',methods=("POST","GET"))
def UploadSample(): 
    if request.method=='POST':
        filename = "destination.wav"
        r = robjects.r
        r['source']('C:/Users/tejas/Documents/Project_4/Rcode1.R')
        TestSample = robjects.globalenv['sample']
        df_result_r = TestSample(filename)
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(df_result_r)
        print(r_from_pd_df)

        with open('C:/Users/tejas/Downloads/voiceDetails.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                r = row
        #file =("C://Users/Sirisha Kodukula/Downloads/voiceDetails.csv")
        #newData = pd.read_csv(file)
        #r = newData.values[:1]
        print(r)
        print(type(r))
        r = [float(i) for i in r]
        r1 = [r]
        data = pd.read_csv("C:/Users/tejas/voice.csv")
        y = data.label.values
        x_data = data.drop(["label"],axis=1)
        x = (x_data - np.min(x_data)) / (np.max(x_data)).values
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
        g = session.get('gender')
        #knn
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(x_train,y_train)
        print("Score for Number of Neighbors = 2: {}".format(knn.score(x_test,y_test)))
        y_pred_knn = knn.predict(r1)
        #svm
        svm = SVC(random_state=42)
        svm.fit(x_train,y_train)
        print("SVM Classification Score is: {}".format(svm.score(x_test,y_test)))
        y_pred_svm = svm.predict(r1)
        #deep learning model
        np.random.seed(1)
        data = pd.read_csv('voice.csv')
        X = data.iloc[:,0:20]
        Y = data.iloc[:,20]
        encoder1 = preprocessing.LabelEncoder()
        encoder1.fit(Y)
        classes1 = encoder1.classes_
        l = len(classes1)
        encoder_Y = encoder1.transform(Y)
        Y1 = np_utils.to_categorical(encoder_Y)
        X_train,X_test,Y1_train,Y1_test = train_test_split(X,Y1,test_size = 0.2)
        model = load_model("C:/Users/tejas/Documents/Project_4/model.h5")
        # summarize model.
        model.summary()
        score = model.evaluate(X_test, Y1_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        prediction = model.predict(r1)
        for i in range(0,len(r1)):
            if prediction[i][0] < prediction[i][1]:
                print("male")
                df = "male"
            else:
                print("Female")
                df = "Female"
        print("%i,X=%s"%(i,prediction[i]))
        r.append(g)
        r.append(y_pred_knn)
        r.append(y_pred_svm)
        r.append(df)
        with open('C:/Users/tejas/Documents/Project_4/voiceData.csv', 'a') as f_object:
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)
            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(r)
            # Close the file object
            f_object.close()
        return render_template('page2.html',g=g,knn=y_pred_knn,svm=y_pred_svm,df=df)
if __name__=='__main__':
    app.run(debug=True)
