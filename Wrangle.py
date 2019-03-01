#Load dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA 
import seaborn

class Wrangle:

    def __init__(self,df):
        self.df = df

    def format(self):

        df = self.df

        df.drop(df[df.sq_feet == 0].index, inplace=True)

        df.dropna(inplace=True)

        # Remove outliers
        df = df[df.sq_feet < df.sq_feet.quantile(.80)]

        # Manual one hot encodng of utilities included column
        df = df.assign(heat = 0,electricity = 0, water = 0, internet = 0, cable = 0)
        for index, row in df.iterrows():
            if 'Heat' in row['utilities_included']:
                df.at[index,'heat'] = 1
            if 'Electricity' in row['utilities_included']:
                df.at[index,'electricity'] = 1
            if 'Water' in row['utilities_included']:
                df.at[index,'water'] = 1
            if 'Internet' in row['utilities_included']:
                df.at[index,'internet'] = 1
            if 'Cable' in row['utilities_included']:
                df.at[index,'cable'] = 1


        # Conditionally replace quadrant names
        df.loc[df['quadrant'] == None, 'quadrant'] = "Unspecified"
        df.loc[(df['quadrant'] == 'Inner-City||SW') | (df['quadrant'] == 'SW||Inner-City') , 'quadrant'] = "SW-Central"
        df.loc[(df['quadrant'] == 'Inner-City||NW') | (df['quadrant'] == 'NW||Inner-City') , 'quadrant'] = "NW-Central"
        df.loc[(df['quadrant'] == 'Inner-City||SE') | (df['quadrant'] == 'SE||Inner-City') , 'quadrant'] = "SE-Central"
        df.loc[(df['quadrant'] == 'Inner-City||NE') | (df['quadrant'] == 'NE||Inner-City') , 'quadrant'] = "NE-Central"

        # One hot encoding of quadrants
        df['quadrant'] = pd.Categorical(df['quadrant'])
        dfDummies = pd.get_dummies(df['quadrant'], prefix = 'Quadrant')
        df = pd.concat([df, dfDummies], axis=1)

        # One hot encoding of type
        df['type'] = pd.Categorical(df['type'])
        dfDummies = pd.get_dummies(df['type'], prefix = 'type')
        df = pd.concat([df, dfDummies], axis=1)

        # One hot encoding of community
        df['community'] = pd.Categorical(df['community'])
        dfDummies = pd.get_dummies(df['community'], prefix = 'community')
        df = pd.concat([df, dfDummies], axis=1)

        # Clean the den column
        df.loc[df['den'] == 'Yes', 'den'] = 1
        df.loc[(df['den'] == 'No') | (df['den'] == None) , 'den'] = 0

        # One hot encoding for den
        df['den'] = pd.Categorical(df['den'])
        dfDummies = pd.get_dummies(df['den'], prefix = 'den')
        df = pd.concat([df, dfDummies], axis=1)

        # Remove unencoded cols
        df.drop(['type','community','den','quadrant','utilities_included'],axis=1,inplace = True)

        # Remove any blank entries (necessary for matrix)
        df.replace('', np.nan, inplace=True)
        df.dropna(inplace=True)

        self.df = df


    def ffs(self):
        """Forward Feature Selection"""

        df = self.df

        from sklearn.feature_selection import f_regression

        X = df.drop(['price'],axis = 1)
        y = df["price"]

        ffs = f_regression(X,y)

        variables = []
        for i in range(0,len(X.columns)-1):
            if ffs[0][i] >=50:
                variables.append(X.columns[i])

        variables.insert(0,'price')

        self.df = df[variables]

    def pca(self):
        """Principal component analysis"""

        df = self.df

        depend = df.drop(['price'],axis = 1)
        y = df["price"]

        scaled = StandardScaler().fit_transform(df)

        X = scaled[:,1:]
        y = scaled[:,0]

        pca = PCA(n_components=6)

        # Conduct PCA
        X_pca = pca.fit_transform(X)

        # results
        # print('Original number of features:', X.shape[1])
        # print('Reduced number of features:', X_pca.shape[1])

        dataset = pd.DataFrame(
            {'price':y,
            'pc1':X_pca[:,0],
            'pc2':X_pca[:,1],
            'pc3':X_pca[:,2],
            'pc4':X_pca[:,3],
            'pc5':X_pca[:,4],
            'pc6':X_pca[:,5]}
            )

        self.df = dataset