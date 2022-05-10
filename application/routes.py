from application import app
import numpy as np
from joblib import load
import pandas as pd
from flask import render_template, redirect, url_for, flash,get_flashed_messages, request
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import style
import matplotlib.cm as cm
import matplotlib.colors as colors

import seaborn as sns

from io import BytesIO
import base64
#import forms
# Categorical pipeline
categorical_preprocessing = Pipeline(
[
    ('Imputation', SimpleImputer(strategy='constant', fill_value='?')),
    ('One Hot Encoding', OneHotEncoder(handle_unknown='ignore')), #OneHotEncoder(handle_unknown='ignore')
]
)

# Numeric pipeline
numeric_preprocessing = Pipeline(
[
     ('Imputation', SimpleImputer(strategy='mean')),
     ('Scaling', StandardScaler()) #MinMax giver lidt mere accuracy - RobustScaler er god til at detektere overvægt (min. F1score på 0.8905)
]
)


# Creating preprocessing pipeline
preprocessing = make_column_transformer(
     (numeric_preprocessing, ['Age','Height','FCVC','NCP','CH2O','FAF','TUE']),
     (categorical_preprocessing, ['Gender','family_history_with_overweight','FAVC','CAEC','SCC','CALC','MTRANS']),
)

pipeline = Pipeline(
[('Preprocessing', preprocessing)]
)

@app.route("/")
@app.route("/index") #decorators - decorator for en funktion
@app.route("/home")
def index():
    return render_template("index.html", index=True)

@app.route("/project", methods =['GET', 'POST'])
def project():
    if request.method == 'POST':
        data = pd.read_csv("ObesityData.csv")
        X = data.drop('NObeyesdad',axis=1)
        X = X.drop('SMOKE',axis=1)#prøvede at fjerne features for at gøre den mere præcist
        X = X.drop('Weight',axis=1)  
        t_list=(request.form.getlist('mycheckbox'))#returns a list
        t_list=convert(t_list) #convert all numbers in the list to float
        #if (len(t_list)!=14):
         #   return render_template("project.html", index=True)
        #print(t_list)
        t_list_1=t_list[1:15]
        print(len(t_list_1))
        test_data=[]
        test_data.append(t_list_1)#appends the data - needs a 2d input
        t=pd.DataFrame(test_data, columns=['Gender', 'Age', 'Height', 'family_history_with_overweight','FAVC','FCVC','NCP','CAEC','CH2O', 'SCC','FAF','TUE','CALC','MTRANS'])
        print(t)
        if(len(X)>2111):
            X=X.drop(X.index[-1])
        X=X.append(t, ignore_index=True)
        X_fit2=pipeline.fit_transform(X)#transforming the data, so it's scaled according to the rest of the data
        F=[]
        F.append(X_fit2[-1])#appends the last list again - model needs 2d array
        model = load('modelrf.joblib')
        y_pred_0 = model.predict(F)#predicts the result based on the latest input
        results=[]
        results.append(y_pred_0[0])#sends the results
        #plot
        bmi_results=[]
        print(t_list[0])
        print(t_list[3])
        bmi_results.append(BMI(t_list[0], t_list[3]))
        img=make_figure(t_list[3],t_list[0],y_pred_0[0])
        return render_template("project.html", index=True, results=results, bmi_results=bmi_results, img=img)
    else:
        return render_template("project.html", index=True)


def convert(float_str): 
  def is_float(s): 
    try: 
      float(s)
      return True
    except: 
      return False
  new_float=[]  
  for x in float_str:
      if is_float(x)==1:
          f=float(x)
          new_float.append(f)
      else:
          new_float.append(x)   
  return new_float

def BMI_calc(weight, height):
    bmi=((weight)/(height*height))
    return bmi
def BMI(weight, height):
    bmi_result=""
    bmi=BMI_calc(weight, height)
    print(bmi)
    if bmi<18.5:
        bmi_result ="Insufficient_Weight"
    elif bmi >= 18.5 and bmi <=24.9:
        bmi_result ="Normal_Weight"
    elif bmi >= 25 and bmi <=29.9:
        bmi_result ="Overweight"
    elif bmi >= 30.0 and bmi <=34.9:
        bmi_result ="Obesity_Type_I"
    elif bmi >= 35.0 and bmi <=39.9:
        bmi_result ="Obesity_Type_II"
    elif bmi >= 40:
        bmi_result ="Obesity_Type_III"
    return bmi_result

def make_figure(height, weight, bmi):
    style.use("ggplot")
    f= Figure((5,5))
    p = f.add_subplot()
    p.set_ylim(0,175)
    p.set_xlim(1.4,2.1)
    p.set_ylabel("Weight[kg]")
    p.set_xlabel("Height[meters]")
    def calc_weight (bmi, height):
        weight=bmi*(height*height)
        return weight
    #BYG FUNKTION - https://stackoverflow.com/questions/10046262/how-to-shade-region-under-the-curve-in-matplotlib
    #https://www.geeksforgeeks.org/matplotlib-axes-axes-twinx-in-python/
    h = np.arange(1.4,2.1,0.01)
    p.plot(h, calc_weight(39.91,h), color ='red') #OBESITY III 
    p.plot(h,calc_weight(35,h)) #OBESITY II
    p.plot(h,calc_weight(30,h)) #OBESITY I
    p.plot(h,calc_weight(25,h)) #OVERWEIGHT
    p.plot(h,calc_weight(18.5,h), color='green')#NORMAL
    #p.plot(np.linspace(1.2,2.5),(np.linspace(26.63,115.656)))
    d= pd.read_csv("ObesityData.csv")
    cond=[
        (d['NObeyesdad']=='Obesity_Type_III'),
        (d['NObeyesdad']=='Obesity_Type_II'),
        (d['NObeyesdad']=='Obesity_Type_I'),
        (d['NObeyesdad']=='Overweight_Level_II'),
        (d['NObeyesdad']=='Overweight_Level_I'),
        (d['NObeyesdad']=='Normal_Weight'),
        (d['NObeyesdad']=='Insufficient_Weight'),
    ]
    colorlist = ['red','orange','yellow','purple','purple','green','blue']
    d['c']=np.select(cond, colorlist)
    p.scatter(d['Height'], d['Weight'], c=d['c'].values)
 
    p.legend()

    def weight_calc_bmi(bmi, height):
        bmi_val=0
        if bmi=='Insufficient_Weight':
            bmi_val=18.3
        elif bmi=='Normal_Weight':
            bmi_val=21.7 #median
        elif bmi=='Overweight_Level_I':
            bmi_val=26.25 #median
        elif bmi=='Overweight_Level_II':
            bmi_val=28.075 #median
        elif bmi=='Obesity_Type_I':
            bmi_val=32.45 #median
        elif bmi=='Obesity_Type_II':
            bmi_val=37.45 #median
        elif bmi=='Obesity_Type_III':
            bmi_val=42 #median

        w=bmi_val*(height*height)
        return w

    x=np.array([])
    x=np.append(x, height)
    x=np.append(x, height)
    y=np.array([])
    y=np.append(y,weight)
    y=np.append(y,weight_calc_bmi(bmi,height))
    c=np.array(["black", "grey"])
    #cdict={0: 'red',1: 'red',2: 'green',3: 'red',4: 'red',5: 'red',6: 'red'}
    p.scatter(x,y, c=c)
    p.arrow(x[0],y[0],dx=0, dy=y[1]-y[0], width=0.01, color='black')
   # for g in np.unique(x):
    #    ix = np.where(x== g)
    #colors = {"Underweight": 'red', "Normal": 'green', "Overweight":'red', "Obesity I":'red', "Obesity II":'red', "Obesity III":'red'}
    #p.axes.set_xticklabels(["","Underweight", "Normal", "Overweight", "Obesity I", "Obesity II", "Obesity III", ""], rotation=20, ha='right', rotation_mode='anchor', fontdict=None, minor=False)
    #weight.axes.set_xticklabels(["","Underweight", "Normal", "Overweight", "Obesity I", "Obesity II", "Obesity III", ""], rotation=20, ha='right', rotation_mode='anchor', fontdict=None, minor=False)
    #https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.arrow.html
  
    # Save it to a temporary buffer. https://matplotlib.org/3.5.0/gallery/user_interfaces/web_application_server_sgskip.html
    buf = BytesIO()
    f.savefig(buf, format="jpeg")
    #b64encoded_str = base64.b64encode(f)
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii") #https://docs.python.org/3/library/base64.html
    return data 




