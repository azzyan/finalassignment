import streamlit as st 
import numpy as np 
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor

#from sklearn.metrics import classification_report

st.title('Machine Learning - REGRESSION')

st.sidebar.write("""
This is a web app is my product for the final assignment 
""")

choice = st.sidebar.radio(
    "Choose a dataset",   
    ('Default', 'User-defined '),
    index = 0
    
)

st.write(f"## You Have Selected <font color='Aquamarine'>{choice}</font> Dataset", unsafe_allow_html=True)



def get_default_dataset(name):
    file = pd.read_csv('https://raw.githubusercontent.com/azzyan/finalassignment/main/winequality-red.csv')
   if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'volatile acidity':
        file = datasets.volatile_acidity()
    elif name == 'citric acid':
        file = datasets.citric_acid()
    elif name == 'residual sugar':
        file = datasets.residual_sugar()
    elif name == 'chlorides':
        file = datasets.chlorides()
    elif name == 'free sulfur dioxide':
        file = datasets.free_sulfur_dioxide()
    elif name == 'total sulfur dioxide':
        file = datasets.total_sulfur_dioxide()    
    elif name == 'density':
        file = datasets.density() 
    elif name == 'pH':
        file = datasets.pH() 
    elif name == 'sulphates':
        file = datasets.sulphates() 
    else:
        file = datasets.alcohol()
    X = file.drop('quality', axis = 1)
    y = file.quality
    return X, y

def add_dataset_ui(choice_name):
    X=[]
    y=[]
    X_names = []
    X1 = []
    if choice_name == 'Default':
       dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol' )
        )
       X, y = get_default_dataset (dataset_name)
       X_names = X
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV",
            type='csv'    )
        

        if uploaded_file!=None:
           
           st.write(uploaded_file)
           data = pd.read_csv(uploaded_file)
  
        
           y_name = st.sidebar.selectbox(
                    'Select Label @ y variable',
                    sorted(data)
                    )

           X_names = st.sidebar.multiselect(
                     'Select Predictors @ X variables.',
                     sorted(data),
                     default = sorted(data)[1],
                     help = "You may select more than one predictor"
                     )

           y = data.loc[:,y_name]
           X = data.loc[:,X_names]
           X1 = X.select_dtypes(include=['object'])
        
           X2 = X.select_dtypes(exclude=['object'])

           if sorted(X1) != []:
              X1 = X1.apply(LabelEncoder().fit_transform)
              X = pd.concat([X2,X1],axis=1)

           y = LabelEncoder().fit_transform(y)
        else:
           st.write(f"## <font color='Aquamarine'>Note: Please upload a CSV file to analyze the data.</font>", unsafe_allow_html=True)

    return X,y, X_names, X1

X, y , X_names, cat_var= add_dataset_ui (choice)

