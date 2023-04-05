import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import seaborn as sns 
from category_encoders import OneHotEncoder

data_url = 'https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Applied-Modeling/master/data/'

# wrangle data
def wrangle(filepath):
    # Import csv file
    cols = ['BOROUGH', 'NEIGHBORHOOD',
            'BUILDING CLASS CATEGORY', 'GROSS SQUARE FEET',  
            'YEAR BUILT', 'SALE PRICE', 'SALE DATE']
    df = pd.read_csv(filepath, 
                     usecols=cols, 
                     parse_dates=['SALE DATE'], 
                     index_col='SALE DATE', 
                     dtype={'BOROUGH': 'object'})
    # clean column headers 
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # transform sale_price into int
    df['sale_price'] = (
        df['sale_price']
        .str.replace('$', '')
        .str.replace('-', '')
        .str.replace(',', '')
        .astype(int))
    
    # subseet df to one-fam dwellings, 100k-2mil
    df = df[(df['building_class_category'] == '01 ONE FAMILY DWELLINGS')
            & (df['sale_price'] > 100_000)
            & (df['sale_price'] < 2_000_000)]

    return df.drop(columns='building_class_category')

filepath = data_url+'condos/NYC_Citywide_Rolling_Calendar_Sales.csv'

df = wrangle(filepath)

# split the data in order to predict sale_price
target = 'sale_price'
X = df.drop(columns=target)
y = df[target]

# split x and y into a training set and test set
cutoff = '2019-04-01'
mask = X.index < cutoff
X_train, y_train = X.loc[mask], y.loc[mask]
X_test, y_test = X.loc[~mask], y.loc[~mask]

# calculate baseline mean
y_pred = [y_train.mean()] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred)

# use a onehotencoder to transform X_train and X_test
ohe = OneHotEncoder(use_cat_names=True)
ohe.fit(X_train)

XT_train = ohe.transform(X_train)
XT_test = ohe.transform(X_test)

# linear regression model named model_lr
model_lr = LinearRegression()
model_lr.fit(XT_train, y_train)

# ridge model named model_r
model_r = Ridge()
model_r.fit(XT_train, y_train)

# metrics for model_lr
training_mae_lr = mean_absolute_error(y_train, model_lr.predict(XT_train))
test_mae_lr = mean_absolute_error(y_test, model_lr.predict(XT_test))

# metrics and R^2 for model_r
training_mae_r = mean_absolute_error(y_train, model_r.predict(XT_train))
test_mae_r = mean_absolute_error(y_test, model_r.predict(XT_test))

training_r2 = model_r.score(XT_train, y_train)
test_r2 = model_r.score(XT_test, y_test)

# barchart
coefficients = model_r.coef_
features = ohe.get_feature_names()
feat_imp = pd.Series(coefficients, index=features).sort_values(key=abs)

feat_imp.tail(10).plot(kind='barh')
