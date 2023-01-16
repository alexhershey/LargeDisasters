# Disaster Projection--Multiple Incident Types
This repository is to investigate ways to predict disasters at the county level. 

# Motivation: 
The motivation for this project is to determine if there are ways to predict disaster outcomes for different groups of people. The major drivers assisnting in the prediction is the rate of registrations per household. By subdividing the data into smaller units, the variance is more easily accounted for and better predictions are thought to be made. 

# Questions:
This analysis was to investigate three questions. Specifically:
1.Are female heads of households, households with people with disabilities, households with elderly people, working individuals or SNAP recipients good indicators of the number of registrations received?
2. Is there a meaningful way to sub-divide the data to reduce the variance?
How can we use the number of registrations per household to help make predictions about the disaster?
3. Can we find a model to classify the counties into the sub-divided groups?
    a. Is there a relationship between the geographic characteristics of a county and the number of registrations received?


# Imported Pre-Processing and Visualization libraries include:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from datetime import datetime
import requests
from io import BytesIO, StringIO
import seaborn as sns
import warnings
warnings.simplefilter(action = 'ignore',category = FutureWarning)
warnings.simplefilter(action = 'ignore',category = RuntimeWarning)

Census API Key found here:
https://www.census.gov/data/developers/guidance/api-user-guide.html

# Imported Machine Learning libraries include:
#Import Machine Learning Libraries: skLearn and StatsModel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import  KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import  mean_squared_error, r2_score, confusion_matrix, accuracy_score, f1_score, recall_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestRegressor as RFR, RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.cluster import KMeans
import statsmodels.api as sm

# Files:
ReadMe: This file contains a list of libraries, motivation for the project, a list of files and resources used.

JupyterNotebook: This file contains all of the code related ot findings and results. Within it are connections to the FEMA and Census API and evaluates a series of different paremeters for forecasting the number of registrations

BlogPost: The blog post describes the findings within the Jupyter notebook.

# Findings:
1. The Lasso Coefficients identify S2201_C01_007E as the element most useful to the regression. 
2. The data clusters more closely around the line of best fit when sub-divided into 5 groups (Community Impact). 
3. The area square miles of water did improve the R² value, but was not statistically significant.
4. The classification of the five groups for future use is correct about 60% of the time. Finding additional data to remove the imputation step might help in future analyses. 

# Special Thanks:
The idea for the p-value extraction dictionary came from the following location: https://www.statology.org/statsmodels-linear-
