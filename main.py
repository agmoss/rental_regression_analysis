from Connection import Connection
from Query import Query
from Wrangle import Wrangle
from Linear import Linear


if __name__ == "__main__":

    con = Connection.connect()

    df = Query(con).data_for_analysis()

    baseDf = Wrangle(df).format()

    lm = Linear(baseDf).smlinear_w_constant()