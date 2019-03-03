from Connection import Connection
from Query import Query
from Wrangle import Wrangle

from Regressors import Regressor, Linear, EnsembleTree, Knn, Mlp

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    con = Connection.connect() # Database objects

    df = Query(con).data_for_analysis() # Initial dataframe

    data = Wrangle(df) # Wrangler class for data cleaning/formatting

    data.format() # Rearange and clean

    data.ffs() # Forward feature selection

    # algorithm objects
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
