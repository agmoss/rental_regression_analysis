import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn import metrics
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

class Linear:

    def __init__(self,df):
        self.df = df

        self.X = self.df.drop(['price'],axis = 1).values
        self.y = self.df["price"].values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)

    def smlinear_w_constant(self):

        self.X_train = sm.add_constant(self.X_train) # intercept (beta_0) t

        # sm.OLS(output, input)
        model = sm.OLS(self.y_train, self.X_train).fit() 
        #predictions = model.predict(X)

        # Print out the statistics
        print(model.summary())

    def sklinear(self):

        lm = linear_model.LinearRegression()
        lm.fit(self.X_train,self.y_train)

        # Make predictions based on independent variable testing data
        y_pred = lm.predict(self.X_test)

        mae = round(metrics.mean_absolute_error(self.y_test, y_pred), 2)
        mse = round(metrics.mean_squared_error(self.y_test, y_pred), 2)
        rmse = round(np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)), 2)
        r2 = round(metrics.r2_score(self.y_test, y_pred), 2)

        print('Error Measures: Multiple Linear ')
        print('')
        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:', mse)
        print('Out of sample R-square value: ', r2)
        print('')
        print('Important!')
        print('Root Mean Squared Error:', rmse)
        print("MAX: " , np.max(self.y_train))
        print("MIN: " , np.min(self.y_train))


        print("In Sample R2 Score: {0}" .format(lm.score(self.X_train,self.y_train)))
        #print("Coefficients: {0}".format(lm.coef_))
        #print("Intercepts: {0} " .format(lm.intercept_))
