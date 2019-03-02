import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

class EnsembleTree:

    def __init__(self,df):
        self.df = df

        self.X = self.df.drop(['price'],axis = 1).values
        self.y = self.df["price"].values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)

    def skEnsemble(self):

         # Train the model
        regr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, loss='ls')
        regr.fit(self.X_train, self.y_train)

        # Make predictions based on independent variable testing data
        y_pred = regr.predict(self.X_test)

        # Compare independent variables predictions to independent variable test data
        from sklearn import metrics

        mae = round(metrics.mean_absolute_error(self.y_test, y_pred), 2)
        mse = round(metrics.mean_squared_error(self.y_test, y_pred), 2)
        rmse = round(np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)), 2)
        r2 = round(metrics.r2_score(self.y_test, y_pred), 2)

        print('Error Measures: GradientBoostingRegressor')
        print('')
        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:', mse)
        print('Out of sample R-square value: ', r2)
        print('')
        print('Important!')
        print('Root Mean Squared Error:', rmse)
        print("MAX: " , np.max(self.y_train))
        print("MIN: " , np.min(self.y_train))
        print("In Sample R2 Score: {0}" .format(regr.score(self.X_train,self.y_train)))
