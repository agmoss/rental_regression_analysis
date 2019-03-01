import pandas as pd
import numpy as np

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

        return df