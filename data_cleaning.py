import pandas as pd
import numpy as np 

# url 
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

column_headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                   'income']

adult = pd.read_csv(data_url, names = column_headers)

# print out first five rows of the DataFrame
adult.head()

# assign first ten and last ten rows to adult_head and adult_tail
adult_head = adult.head(10)
adult_tail = adult.tail(10)

# count number of int64 variable types and assign to number_int64
# count number of object variable types and assign to number_object
number_int64 = adult.dtypes.value_counts()[1]
number_object = adult.dtypes.value_counts()[0]

# find dimensions of dataframe and assign result to adult_dimension
adult_dimension = adult.shape

# check for missing values in the dataset 
adult_missing = adult.isnull().sum().sum()

# find the mean age and assign to mean_age
# find the standard deviation for hours/week and assign to std_hpw
adult.describe()
mean_age = 38.58

std_hpw = round(adult.describe().loc['std', 'hours-per-week'], 2)

# find number of unique education values and assign to unique_edu
# find the number of times the most frequent observation for income occurs and assign to freq_income
adult.describe(exclude = 'number')
unique_edu = 16
freq_income = 24720

# finding value counts
adult['relationship'].unique()
adult['relationship'].value_counts()

adult_other_rel = adult['relationship'].value_counts()[5]

# create a series from occupation and name it adult_occup
adult_occup = adult['occupation']

