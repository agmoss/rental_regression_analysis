from Connection import Connection
from Query import Query
from Wrangle import Wrangle

from Regressors import Regressor, Linear, EnsembleTree, Knn, Mlp


if __name__ == "__main__":

    con = Connection.connect()

    df = Query(con).data_for_analysis()

    data = Wrangle(df)

    data.format() # Rearange and clean

    data.ffs() # Forward feature selection

    # Train and Test
    linear = Linear(data.df).sklinear()

    ensemble = EnsembleTree(data.df).skEnsemble()

    neighbors = Knn(data.df).skKnn()

    nn = Mlp(data.df).skMlp()
