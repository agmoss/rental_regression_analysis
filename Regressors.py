# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Evaluation
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Scale
from sklearn.preprocessing import StandardScaler

# Models
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV


class Regressor:
    def __init__(self, df):
        self.df = df

        self.dependent_vars = list(self.df.drop(["price"], axis=1))

        # Simulated test case for prediction (to be populated with user input)
        self.x_case = {
            "sq_feet": 5000,
            "bedrooms": 3,
            "baths": 1,
            "dogs": 0,
            "cable": 0,
            "Quadrant_SW-Central": 0,
            "type_Basement": 0,
            "type_Condo": 0,
            "type_House": 1,
            "type_Shared": 0,
            "community_Beltline": 0,
            "community_Downtown": 0,
            "community_Eau Claire": 0,
            "community_Victoria Park": 0,
            "den 1": 0,
        }

        self.X = self.df.drop(["price"], axis=1).values
        self.y = self.df["price"].values
        self.y = self.y.reshape(-1, 1)

        # Scale objects
        self.scalerX = StandardScaler().fit(self.X)
        self.scalery = StandardScaler().fit(self.y)

        # Scale the data (mean 0 , SD 1)
        self.X = self.scalerX.transform(self.X)
        self.y = self.scalery.transform(self.y)

        # Scale the test case
        vals = np.fromiter(self.x_case.values(), dtype=float).reshape(1, -1)
        self.x_case = self.scalerX.transform(vals)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=1
        )

    def evaluate(self, name, model, y_pred):
        """Testing error measures"""

        rescaled_y_test = self.scalery.inverse_transform(self.y_test)
        rescaled_y_pred = self.scalery.inverse_transform(y_pred)
        rescaled_y_train = self.scalery.inverse_transform(self.y_train)

        mae = round(metrics.mean_absolute_error(rescaled_y_test, rescaled_y_pred), 2)
        mse = round(metrics.mean_squared_error(rescaled_y_test, rescaled_y_pred), 2)
        rmse = round(
            np.sqrt(metrics.mean_squared_error(rescaled_y_test, rescaled_y_pred)), 2
        )
        r2 = round(metrics.r2_score(self.y_test, y_pred), 2)
        inSampleR2 = round(model.score(self.X_train, self.y_train), 2)

        print("")
        print("")
        print("Error Measures: ", name)
        print("")
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("Out of sample R-square value: ", r2)
        print("In Sample R2 Score: {0}".format(inSampleR2))
        print("")
        print("Important!")
        print("Root Mean Squared Error:", rmse)
        print("MAX: ", np.max(rescaled_y_train))
        print("MIN: ", np.min(rescaled_y_train))
        print("")

        return 1

    def sample_backwards(self, y_pred):

        rescaled_y_test = self.scalery.inverse_transform(self.y_test)
        rescaled_y_pred = self.scalery.inverse_transform(y_pred)

        rescaled_y_test = rescaled_y_test.ravel()
        rescaled_y_pred = rescaled_y_pred.ravel()

        print("")
        __df = pd.DataFrame({"Actual": rescaled_y_test, "Predicted": rescaled_y_pred})
        print(__df)

        return 1

    def predict(self, model):

        model.fit(self.X, self.y)

        # Make prediction
        y_pred = model.predict(self.x_case)

        rescaled_y_pred = self.scalery.inverse_transform(y_pred)

        print(rescaled_y_pred)

        return 1


class Linear(Regressor):
    def __init__(self, df):

        # Instantiate superclass
        super().__init__(df)

    def smlinear_w_constant(self):

        self.X_train = sm.add_constant(self.X_train)  # intercept (beta_0) t

        # sm.OLS(output, input)
        model = sm.OLS(self.y_train, self.X_train).fit()
        # predictions = model.predict(X)

        # Print out the statistics
        print(model.summary())

        return 1

    def sklinear(self, predict=False):

        lm = linear_model.LinearRegression()

        if not predict:
            lm.fit(self.X_train, self.y_train)

            # Make predictions based on independent variable testing data
            y_pred = lm.predict(self.X_test)

            super().evaluate("Multiple Linear", lm, y_pred)
            super().sample_backwards(y_pred)

        else:

            super().predict(lm)

        return 1


class EnsembleTree(Regressor):
    def __init__(self, df):
        # Instantiate superclass
        super().__init__(df)

    def skEnsemble(self, predict=False):

        regr = GradientBoostingRegressor()

        if not predict:
            # Train the model

            regr.fit(self.X_train, self.y_train)

            # Make predictions based on independent variable testing data
            y_pred = regr.predict(self.X_test)

            super().evaluate("Ensemble Tree", regr, y_pred)
            super().sample_backwards(y_pred)

        else:

            super().predict(regr)

        return 1


class Knn(Regressor):
    def __init__(self, df):
        # Instantiate superclass
        super().__init__(df)

    def skKnn(self, predict=False):

        neigh = KNeighborsRegressor(n_neighbors=5)

        if not predict:

            neigh.fit(self.X_train, self.y_train)

            # Make predictions based on independent variable testing data
            y_pred = neigh.predict(self.X_test)

            # Compare independent variables predictions to independent variable test data
            super().evaluate("KNN", neigh, y_pred)
            super().sample_backwards(y_pred)

        else:

            super().predict(neigh)

        return 1


class Mlp(Regressor):
    def __init__(self, df):
        # Instantiate superclass
        super().__init__(df)

    def skMlp(self, predict=False):

        # Multi-layer Perceptron regressor
        mlp = MLPRegressor(
            hidden_layer_sizes=(30, 10),
            activation="tanh",
            solver="adam",
            learning_rate="adaptive",
            max_iter=1000,
            learning_rate_init=0.001,
            warm_start=True,
            alpha=0.01,
        )

        if not predict:

            # Fit the best algorithm to the data.
            mlp.fit(self.X_train, self.y_train)

            # mlp.fit(self.X_train, self.y_train)

            y_pred = mlp.predict(self.X_test)

            super().evaluate("MPL", mlp, y_pred)  # Error measures
            super().sample_backwards(
                y_pred
            )  # Compare independent variables predictions to independent variable test data

        else:

            super().predict(
                mlp
            )  # Predict the price using the sample user specified input data

        return 1
