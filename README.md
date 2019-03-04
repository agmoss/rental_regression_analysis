# Multiple Regression Analysis of Rental Listing Data 
>Predicting rental prices in the city of Calgary 

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) 

## Table of Contents 

1. [About the Project](#about-the-project) 
1. [Project Status](#project-status) 
1. [Variables](#Variables) 
1. [Exploratory Data Analysis](#exploratory-data-analysis) 
1. [Regression](#regression) 
    * [Principles of Regression Analysis](#principles-of-regression-analysis) 
    * [Linear Model](#linear-model) 
    * [Results](#results) 
    * [Supervised Machine Learning Regression](#Supervised-Machine-Learning-Regression) 
        * [Multilayer Perceptron (MLP)](#Multilayer-Perceptron-(MLP)) 
        * [K Nearest Neighbors (KNN)](#k-nearest-neighbors-(KNN)) 
        * [Gradient Boosting Regressor](#Gradient-Boosting-Regressor) 
    * [Assessing Model Fit (RMSE)](#assessing-model-fit-rmse) 
    * [Evaluation & Results](#evaluation--results)  
1. [Usage](#usage)  
1. [Prerequisites](#Prerequisites) 
1. [Built With](#Built-With) 
1. [Author](#author) 
1. [License](#license) 


## About the Project  

This analysis has been developed as part of the Calgary Project, a free and open source tool for rental property analysis. The goal of the Calgary Project is to democratize an understanding of the Calgary rental market for all. 

## Project Status  

Beta 

## Variables 

* ***Discrete categorical attributes***: community, den, location, quadrant, type, utilities_included 
* ***Discrete numeric attributes***: baths, bedroons, cats, dogs 
* ***Continuous attributes***: price, sq_feet 

## Exploratory Data Analysis 

An exploratory data analysis dashboard for this data is housed at:  

https://calgaryproject.net/dashboard/ 

## Regression 

### Principles of Regression Analysis 

Regression analysis is a predictive modeling technique used to determine the relationships between variables. A regression model provides the user an opportunity to predict the outcome of a relationship with a dependent variable and one or more independent variables. The dependent variable is the factor you wish to understand or predict, and independent variable(s) cause the changes in the dependent variables. 

In this modeling scenario, the dependent variable is price per month. The independent variables are descriptive features such as community, square feet, dogs & cats (yes/no), and utilities included (electricity, heat, cable, internet etc.). It is presumed that the value of price per month is dependent on the value of these descriptive features. 

### Linear Model  

Correlation analysis and forward feature selection confirm that the main variables affecting price are square feet and number of bedrooms. This confirms natural intuition, however the R2 values in a single linear regression are only in the ~0.1 range. Multiple independent variables are needed to accurately model this scenario. 

After one hot encoding, there are 257 independent variables in the model. Unsurprisingly, this yields a strong degree of multicollinearity. To reconcile for this, forward feature selection and principal component analysis are employed for dimensionality reduction.  

The ffs algorithm selects sq_feet, bedrooms, baths, dogs, cable, Quadrant_SW-Central, type_Basement, type_Condo, type_House, type_Shared, community_Beltline, community_Downtown, community_Eau Claire, community_Victoria Park, and den_1 as the top most variables for regression 

PCA yields inconclusive results as the number of components increases in a completely linear fashion with the cumulative explained variance. A re-evaluation of PCA and other dimensionality reduction techniques should take place.  

### Results 

Measure | Value 
------------- | ------------- 
Out of sample R2 | 0.58 
In Sample R2 | 0.62 
RMSE | $288.33 

**[Back to top](#table-of-contents)** 

## Supervised Machine Learning Regression 

To reduce RMSE and increase R2, more complicated machine learning models are used to predict price. The problem proposal remains a multivariate regression with price being the output.  

Currently, three different machine learning models are being considered:  

* Multilayer perceptron  
* K-nearest neighbors 
* Gradient Boosting Regressor 

### Multilayer Perceptron (MLP) 

A multilayer perceptron is a specific implementation of an artificial neural network (ANN). The ANN is a computational model inspired by the way biological neural networks process information.  

An ANN is comprised of layers of nodes. An MLP contains an input layer, two or more middle layers, and an output layer. The layers are connected via weighted edges. The learning process aims to correctly calibrate the edge weights. Given an input vector, these weights determine what the output vector is. 

The process by which an MLP learns is called backpropagation. For every input in the training dataset, the neural network is activated, and its output is observed. This output is compared with the known dependent variable value. If the output is incorrect, the error is propagated back to the previous layer of the network. This error is noted, and the connection weights are adjusted accordingly. This process is repeated until the output error is below a predetermined threshold. 

Once trained, the neural network connections contain the appropriate weights for the given training data. New data can be fed to the network for predictions.  

### K Nearest Neighbors (KNN) 

The K Nearest Neighbors algorithm operates on the premise of feature similarity. Data points are assigned a value based on how closely it resembles a point in the training set. The algorithm assumes that similar things exist near one another 

The KNN algorithm relies on a distance metric to evaluate the similarity between data points.  

### Gradient Boosting Regressor 

A tree-based method of regression involves stratifying and segmenting the predictor space into several smaller regions. A computer-generated rule is used to separate the data into defined homogenous groups. Since the set of splitting rules used to segment the predictor space can be visualized as a tree, this approach is known as a regression tree. 

The regression tree approach supports the ensemble method of boosting. Boosting is an iterative technique that generates n number of trees. As these trees are generated, a strong weighting is placed on observations that are predicted incorrectly. As the algorithm iterates, the weighting of incorrect predictions accumulates until the model self corrects to minimize this weighting. The resulting model is the best of the previous n models created.  

The advantage of tree-based learning is an ease of implementation. Data of differing scales and denominations can be input to the algorithm without the need for normalization. Furthermore, the implementation of ensemble-based learning provides far superior predicative accuracy when compared to a single model implementation.  

## Assessing Model Fit (RMSE) 

The RMSE is the square root of the variance of the residuals. This value indicates the absolute fit of the model to the data. It determines how close the observed data point is to the models predicted values. RMSE is conveniently denominated in the same units as the dependent variable (price per month). Lower RMSE values indicate better fit. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction. 

As RMSE is the standard deviation of unexplained variance, it should be used to evaluate the potential for incorrect prediction. RMSE claims that historically, the model’s predictions have exhibited a variance equal to the magnitude of the RMSE value. Therefore, if we assume that our historical training data is indicative of the future, the model’s prediction can potentially be off by plus or minus one RMSE value.  

***Interpreting the RMSE***

* Low RMSE value = more accurate prediction  
* Prediction can potentially be off by ±RMSE 

 
**[Back to top](#table-of-contents)** 

## Evaluation & Results 
Model parameters are currently being tuned. 

### Multilayer Perceptron (MLP)
Measure  | Value
------------- | -------------
Out of sample R2 | 0.58
In Sample R2 | 0.6
RMSE | $290.19

### K Nearest Neighbors (KNN)
Measure  | Value
------------- | -------------
Out of sample R2 | 0.34
In Sample R2 | 0.57
RMSE | $361.96

### Gradient Boosting Regressor
Measure  | Value
------------- | -------------
Out of sample R2 | 0.64
In Sample R2 | 0.71
RMSE | $266.03


## Usage

The development of the project has yielded a reusable codebase for machine learning with scikit-learn. The Regressors.py file contains a class called Regressor. Instantiating this class requires a dataframe object of rental listing data. The class exposes formatted NumPy arrays of training/testing data, prediction data, and a sample prediction case. When implemented on the web, prediction dependent variables will be provided to the user and used in the instantiating of this class.  


Four algorithm class subclass the regressor and gain access to this data. The algorithm classes implement thier respective machine learning methods via the scikit-learn package. Methods within the algorithm classed and can either train/test the algorithm or offer a price prediction for the user specified dependent variables. 

```python
if __name__ == "__main__":
    """Main Method"""

    con = Connection.connect() # Database connection object

    df = Query(con).data_for_analysis() # Initial dataframe

    data = Wrangle(df) # Wrangler class for data cleaning/formatting

    data.format() # Rearange and clean

    data.ffs() # Forward feature selection

    # Algorithm objects
    linear = Linear(data.df)
    ensemble = EnsembleTree(data.df)
    neighbors = Knn(data.df)
    nn = Mlp(data.df)
    
    # Training/testing (outputs error measures)
    linear.sklinear(predict = False)
    ensemble.skEnsemble(predict = False)
    neighbors.skKnn(predict = False)
    nn.skMlp(predict=False)

    # Predictions
    linear.sklinear(predict = True)
    ensemble.skEnsemble(predict = True)
    neighbors.skKnn(predict = True)
    nn.skMlp(predict=True)
```

The following is an example of an algorithm class:

```python
class Mlp(Regressor):

    def __init__(self,df):
        # Instantiate superclass
        super().__init__(df)

    def skMlp(self,predict = False):

        # Multi-layer Perceptron regressor
        mlp = MLPRegressor(hidden_layer_sizes=(500,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)

        if not predict:

            mlp.fit(self.X_train, self.y_train)

            y_pred = mlp.predict(self.X_test)

            super().evaluate("MPL",mlp,y_pred) # Error measures
            super().sample_backwards(y_pred) # Compare independent variables predictions to independent variable test data

        else:

            super().predict(mlp) # Predict the price using the sample user specified input data

```

## Contributing

This project is currently not open for contributions

## Prerequisites

Dependencies can be installed via:

```
pip install requirements.txt
```

## Built With 

* [scikit-learn](https://scikit-learn.org/stable/) - Machine Learning in Python  
* [StatsModels](https://www.statsmodels.org/stable/index.html) - Statistics in Python  
* [Pandas](https://pandas.pydata.org/) - Python data analysis library 
* [NumPy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 

## Author 
* **Andrew Moss** - *Creator* - [agmoss](https://github.com/agmoss) 

## License 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details 

**[Back to top](#table-of-contents)** 
