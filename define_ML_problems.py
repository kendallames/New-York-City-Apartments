import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, plot_confusion_matrix, plot_roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('NYC.csv')

def wrangle(filepath):

  df = pd.read_csv('NYC.csv')

  # Import with DateTime Index
  df = pd.read_csv(filepath, parse_dates=['Date'], index_col = 'Date')
  
  # drop rows with no overall rating
  df.dropna(subset=['overall'], inplace = True)

  # Create great column as target
  df['great'] = (df['overall'] >= 4).astype(int)  

  # Drop overall column to prevent data leakage
  df.drop(columns='overall', inplace = True)

  # Clean binary encoded columns
  categorical_cols = df.select_dtypes('object').columns

  # use categorical columns which are basically binary encoded
  binary_cols = [col for col in categorical_cols if df[col].nunique() < 4]
  for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if isinstance(x, str) else 0)

  # Drop high-cardinality categorical variables
  threshold = 20

  high_card_cols =  [col for col in categorical_cols
                     if df[col].nunique() > threshold ]
  df.drop(high_card_cols, axis=1, inplace=True)

  # Dropping columns with high number of NaN values
  df.dropna(axis=1, thresh=300, inplace = True)

  return df

df.isnull().sum()

df = wrangle('NYC.csv')

TARGET = 'SALE PRICE'
y = df[TARGET]
X = df.drop(TARGET, axis=1)

# histogram
X['GROSS SQUARE FEET'].hist()