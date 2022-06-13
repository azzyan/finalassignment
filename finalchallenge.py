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
    data = None
    if name == 'Fixed Acidity':
        data = datasets.load_fixed_acidity()
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
            ('Fixed Acidity', 'Breast Cancer', 'Wine')
        )
       X, y = get_default_dataset (dataset_name)
       X_names = X
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV",
            type='csv'    )
