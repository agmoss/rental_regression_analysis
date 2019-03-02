from Connection import Connection
from Query import Query
from Wrangle import Wrangle
from Linear import Linear
from EnsembleTree import EnsembleTree


if __name__ == "__main__":

    con = Connection.connect()

    df = Query(con).data_for_analysis()

    data = Wrangle(df)

    data.format()

    data.ffs()

    ffs = Linear(data.df).smlinear_w_constant()
    ffs1 = Linear(data.df).sklinear()
    ensemble = EnsembleTree(data.df).skEnsemble()
