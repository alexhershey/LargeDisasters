# LargeDisasters
This repository is to investigate ways to predict disasters at the county level. 

# Motivation: 
The motivation for this project is to determine if there are ways to predict disaster outcomes for different groups of people. The Major drivers assisnting in the prediction is the rate of registrations per household. By subdividing the data into smaller units, the variance is more easily accounted for and better predictions are thought to be made. 


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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor as RFR
import statsmodels.api as sm

# Files:
ReadMe: This file contains a list of libraries, motivation for the project, a list of files and resources used.

JupyterNotebook: This file contains all of the code related ot findings and results. Within it are connections to the FEMA and Census API and evaluates a series of different paremeters for forecasting the number of registrations

BlogPost: The blog post describes the findings within the Jupyter notebook. 

# Special Thanks:
The idea for the p-value extraction dictionary came from the following location: https://www.statology.org/statsmodels-linear-