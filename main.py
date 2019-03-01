from Connection import Connection
from Query import Query
from Wrangle import Wrangle
from Linear import Linear


if __name__ == "__main__":

    con = Connection.connect()

    df = Query(con).data_for_analysis()

    data = Wrangle(df)

    data.format()

    data.pca()
    
    lm = Linear(data.df).smlinear_w_constant()