# Standard imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Evaluation
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Models
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


class Regressor:

    def __init__(self,df):
        self.df = df

        self.X = self.df.drop(['price'],axis = 1).values
        self.y = self.df["price"].values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)

    def getData(self):

        return self.X_train,self.X_test,self.y_train,self.y_test


    @staticmethod
    def evaluate(name,model,y_test,y_pred,X_train,y_train):
        """name,model,y_test,y_pred,X_train,y_train"""

        mae = round(metrics.mean_absolute_error(y_test, y_pred), 2)
        mse = round(metrics.mean_squared_error(y_test, y_pred), 2)
        rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)
        r2 = round(metrics.r2_score(y_test, y_pred), 2)
        inSampleR2 = round(model.score(X_train,y_train), 2)

        print('')
        print('')
        print('Error Measures: ', name)
        print('')
        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:', mse)
        print('Out of sample R-square value: ', r2)
        print("In Sample R2 Score: {0}" .format( inSampleR2 ))
        print('')
        print('Important!')
        print('Root Mean Squared Error:', rmse)
        print("MAX: " , np.max(y_train))
        print("MIN: " , np.min(y_train))
        print('')


class Linear(Regressor):

    def __init__(self,df):

        # Instantiate superclass
        super().__init__(df)

        # Get data
        self.X_train,self.X_test,self.y_train,self.y_test = super().getData()

    def smlinear_w_constant(self):

        self.X_train = sm.add_constant(self.X_train) # intercept (beta_0) t

        # sm.OLS(output, input)
        model = sm.OLS(self.y_train, self.X_train).fit() 
        #predictions = model.predict(X)

        # Print out the statistics
        print(model.summary())

        return 1

    def sklinear(self):

        lm = linear_model.LinearRegression()
        lm.fit(self.X_train,self.y_train)

        # Make predictions based on independent variable testing data
        y_pred = lm.predict(self.X_test)

        super().evaluate("Multiple Linear",lm,self.y_test,y_pred,self.X_train,self.y_train)

        return 1 

class EnsembleTree(Regressor):

    def __init__(self,df):
        # Instantiate superclass
        super().__init__(df)

        # Get data
        self.X_train,self.X_test,self.y_train,self.y_test = super().getData()

    def skEnsemble(self):

         # Train the model
        regr = GradientBoostingRegressor()
        regr.fit(self.X_train, self.y_train)

        # Make predictions based on independent variable testing data
        y_pred = regr.predict(self.X_test)

        super().evaluate("Ensemble Tree",regr,self.y_test,y_pred,self.X_train,self.y_train)

        return 1 


class Knn(Regressor):

    def __init__(self,df):
        # Instantiate superclass
        super().__init__(df)

        # Get data
        self.X_train,self.X_test,self.y_train,self.y_test = super().getData()

    def skKnn(self):

        neigh = KNeighborsRegressor(n_neighbors=5)
        neigh.fit(self.X_train, self.y_train)

        # Make predictions based on independent variable testing data
        y_pred = neigh.predict(self.X_test)

        # Compare independent variables predictions to independent variable test data
        super().evaluate("KNN",neigh,self.y_test,y_pred,self.X_train,self.y_train)

        return 1 

class Mlp(Regressor):

    def __init__(self,df):
        # Instantiate superclass
        super().__init__(df)

        # Get data
        self.X_train,self.X_test,self.y_train,self.y_test = super().getData()

    def skMlp(self):

        mlp = MLPRegressor(hidden_layer_sizes=(500,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)

        mlp.fit(self.X_train, self.y_train)

        y_pred = mlp.predict(self.X_test)

        # Compare independent variables predictions to independent variable test data
        super().evaluate("MPL",mlp,self.y_test,y_pred,self.X_train,self.y_train)

        return 1

