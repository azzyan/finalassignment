import streamlit as st 
import numpy as np 
import pandas as pd

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



st.title('Machine Learning - CLASSIFICATION')

st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

st.sidebar.write ("For more info, please contact:")

st.sidebar.write("<a href='https://www.linkedin.com/in/yong-poh-yu/'>Dr. Yong Poh Yu </a>", unsafe_allow_html=True)


choice = st.sidebar.radio(
    "Choose a dataset",   
    ('Default', 'User-defined '),
    index = 0
    
)

st.write(f"## You Have Selected <font color='Aquamarine'>{choice}</font> Dataset", unsafe_allow_html=True)

def get_default_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

def add_dataset_ui(choice_name):
    X=[]
    y=[]
    X_names = []
    X1 = []
    if choice_name == 'Default':
       dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Iris', 'Breast Cancer', 'Wine')
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

