import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

class Linear:

    def __init__(self,df):
        self.df = df

    def smlinear_no_constant(self):

        df = self.df

        X = df["sq_feet"]
        y = df["price"]

        model = sm.OLS(y, X).fit()
        predictions = model.predict(X) # make the predictions by the model

        # Print out the statistics
        print(model.summary())


    def smlinear_w_constant(self):

        df = self.df
        headers = list(df)

        headers.remove('price')

        X = df[headers].astype(float) # Independent variable

        y = df["price"] # dependent variable

        X = sm.add_constant(X) # intercept (beta_0) t

        # sm.OLS(output, input)
        model = sm.OLS(y, X).fit() 
        predictions = model.predict(X)

        # Print out the statistics
        print(model.summary())

    def sklinear(self):

        df = self.df

        X = df.drop(['price'],axis = 1)
        y = df["price"]

        lm = linear_model.LinearRegression()
        model = lm.fit(X,y)

        print("R2 Score: {0}" .format(lm.score(X,y)))
        print("Coefficients: {0}".format(lm.coef_))
        print("Intercepts: {0} " .format(lm.intercept_))
