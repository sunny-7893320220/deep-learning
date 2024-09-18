import pandas as pd

import lightgbm as lgb

import datetime

# Reading in the CSV file as a DataFrame

df = pd.read_csv(r'C:\Users\muham\Downloads\train (1).csv', low_memory=False)

# Looking at the first five rows

df

# Printing the shape

df.shape

df.head()

df.shape

df.isnull().sum()

df['LotFrontage'].mode()

df['LotFrontage'].fillna(60,inplace=True)

df['LotFrontage'].isnull().sum()

df['Alley'].fillna(df['Alley'].mode,inplace=True)

df['Alley'].isnull().sum()

df['Street'].isnull().sum()

df['LotShape'].isnull().sum()

df['LandContour'].isnull().sum()

df['Utilities'].isnull().sum()

df['PoolArea'].isnull().sum()

df['PoolQC'].isnull().sum()

df['PoolQC'].fillna(df['PoolQC'].mean,inplace=True)

df['PoolQC'].isnull().sum()

df['Fence'].isnull().sum()

df['Fence'].fillna(df['Fence'].mean, inplace=True)

df['Fence'].isnull().sum()

df['MiscFeature'].isnull().sum()

df['MiscFeature'].fillna(df['MiscFeature'].mean, inplace=True)

df['MiscFeature'].isnull().sum()

df['MiscVal'].isnull().sum()

df['MoSold'].isnull().sum()

df['YrSold'].isnull().sum()

df['SaleType'].isnull().sum()

df['SaleCondition'].isnull().sum()

df['SalePrice'].isnull().sum()

# lets check all data types

df.dtypes

df.drop(['MSZoning'], axis=1, inplace=True)

# Street	Alley	LotShape	LandContour	Utilities

df.drop(['Street'], axis=1, inplace=True)

df.drop(['Alley'], axis=1, inplace=True)

df.drop(['LotShape'], axis=1, inplace=True)

df.drop(['LandContour'], axis=1, inplace=True)

df.drop(['Utilities'], axis=1, inplace=True)

# 	LotConfig	LandSlope	Neighborhood	Condition1	Condition2	BldgType, PoolQC	Fence	MiscFeature

# SaleType	SaleCondition

df.drop(['LotConfig'], axis=1, inplace=True)

df.drop(['LandSlope'], axis=1, inplace=True)

df.drop(['Neighborhood'], axis=1, inplace=True)

df.drop(['Condition1'], axis=1, inplace=True)

df.drop(['Condition2'], axis=1, inplace=True)

df.drop(['BldgType'], axis=1, inplace=True)

df.drop(['PoolQC'], axis=1, inplace=True)

df.drop(['Fence'], axis=1, inplace=True)

df.drop(['MiscFeature'], axis=1, inplace=True)

df.drop(['SaleType'], axis=1, inplace=True)

df.drop(['SaleCondition'], axis=1, inplace=True)

df.drop(['HouseStyle'], axis=1, inplace=True)

df.drop(['RoofStyle'], axis=1, inplace=True)

df.drop(['RoofMatl'], axis=1, inplace=True)

df.drop(['Exterior1st'], axis=1, inplace=True)

df.drop(['Exterior2nd'], axis=1, inplace=True)

df.drop(['MasVnrType'], axis=1, inplace=True)

df.drop(['ExterQual'], axis=1, inplace=True)

df.drop(['ExterCond'], axis=1, inplace=True)

df.drop(['Foundation'], axis=1, inplace=True)

df.drop(['BsmtQual'], axis=1, inplace=True)

df.drop(['BsmtCond'], axis=1, inplace=True)

#BsmtExposure

df.drop(['BsmtExposure'], axis=1, inplace=True)

df.drop(['BsmtFinType1'], axis=1, inplace=True)

df.drop(['BsmtFinSF1'], axis=1, inplace=True)

df.drop(['BsmtFinType2'], axis=1, inplace=True)

df.drop(['Heating'], axis=1, inplace=True)

df.drop(['HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu',

'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'MoSold'], axis=1, inplace=True)

df.drop(['YrSold'], axis=1, inplace=True)

df.drop(['LotFrontage', 'MasVnrArea', 'Functional', 'GarageYrBlt'], axis=1, inplace=True)

df

df.dtypes

df

# Getting the target (y) from the splitted DataFrames

train_y = df["SalePrice"].astype(float).values

eval_y = df["SalePrice"].astype(float).values



# Getting the features (X) from the splitted DataFrames

train_X = df.drop(['SalePrice', 'GarageCars'], axis=1)

eval_X = df.drop(['SalePrice', 'GarageCars'], axis=1)

from sklearn.model_selection import train_test_split

def train_lightgbm(train_X, train_y, eval_X, eval_y):

    

    # Initializing the training dataset

    lgtrain = lgb.Dataset(train_X, label=train_y)

    

    # Initializing the evaluation dataset

    lgeval = lgb.Dataset(eval_X, label= eval_y)

    

    # Hyper-parameters for the LightGBM model

    params = {

        "objective" : "regression",

        "metric" : "rmse", 

        "num_leaves" : 30,

        "min_child_samples" : 100,

        "learning_rate" : 0.1,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.5,

        "bagging_seed" : 2018,

        "verbosity" : -1

    }

    

    # Training the LightGBM model

    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgeval], early_stopping_rounds=100, verbose_eval=100)

    

    # Returning the model

    return model



# Training the model 

model = train_lightgbm(train_X, train_y, eval_X, eval_y)

# Index to test row 1458

index_val = 1400



# Selecting the index value from the evaluation DataFrame

actual_X_value = eval_X.reset_index(drop=True).iloc[index_val]



# Selecting the Sale Price from the target variable array

actual_y_value = eval_y[index_val]

# Printing the feature values

actual_X_value

# Printing the SalePrice

actual_y_value

# Predicting the value

predict_price = model.predict(actual_X_value.astype(float), predict_disable_shape_check=True)

predict_price

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier()

model_RF.fit(train_X, train_y)

predict_RF = model_RF.predict(eval_X)

predict_RF

from sklearn.metrics import mean_absolute_error

mean_absolute_error(predict_RF, eval_y)

from sklearn.metrics import r2_score

r2_score(predict_RF, eval_y)

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

cv1 = KFold(n_splits=10, random_state=12,shuffle= True)

# evaluate the model with cross validation

scores = cross_val_score(model_RF, train_X, train_y, scoring='accuracy', cv=cv1, n_jobs=-1)

scores

from statistics import mean, stdev

# report perofmance

print('Accuracy: %.3f(%.3f)'% (mean(scores), stdev(scores)))

accuracy_score(predict_RF, eval_y)

# lets use Hyper parametres like Random Search to improve our RFC model

# Random Search

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

random_search = {'criterion': ['entropy', 'gini'],

 'max_depth': list(np.linspace(5, 1200, 10, dtype = int)) + [None],

 'max_features': ['auto', 'sqrt','log2', None],

 'min_samples_leaf': [4, 6, 8, 12],

 'min_samples_split': [3, 7, 10, 14],

 'n_estimators': list(np.linspace(5, 1200, 3, dtype = int))}

clf = RandomForestClassifier()

model_R = RandomizedSearchCV(estimator = clf, param_distributions = random_search, 

 cv = 4, verbose= 5, random_state= 101, n_jobs = -1)

model_R.fit(train_X,train_y)

model_R.best_params_

predict_R = model_R.predict(eval_X)

predict_R

r2_score(predict_R, eval_y)

accuracy_score(predict_R, eval_y)

from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()

model_LR.fit(train_X,train_y)

predict_LR = model_LR.predict(eval_X)

predict_LR

mean_absolute_error(predict_LR, eval_y)

r2_score(predict_LR, eval_y)

from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)

model_skfold = LinearRegression()

results_skfold = cross_val_score(model_skfold, train_X, train_y, cv=skfold)

print("Accuracy: %.2f%%" %(results_skfold.mean()*100.0))

if __name__ == "__main__":
	
	app.run(host='0.0.0.0', port=9000, debug=True)
