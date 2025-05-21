import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def read_data(filepath:str):
    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)


    elif filepath.lower().endswith(".xlsx"):
        df = pd.read_excel(filepath)

    elif filepath.lower().endswith(".sql"):
        df = pd.read_sql(filepath)

    elif filepath.lower().endswith(".json"):
        df = pd.read_json(filepath)

    elif filepath.lower().endswith(".parquet"):
        df = pd.read_parquet(filepath)

    return df





def data_metricsandvisualise(filepath:str , s=None): ##for visualisation of every numeric data column
    df = read_data(filepath)
    mat = np.array(df)
    mat = mat.T
    if s=='all_numeric_data':
        for i in range(mat.shape[0]):
            plt.boxplot(mat[i])
            plt.show()
        for j in range(mat.shape[0]):
            plt.hist(mat[j])
            plt.show()

    return df.describe() , df.head() , df.info() , df.corr()  , df.median() , df.mode()




def preview(filepath:str):
    df = read_data(filepath)
    return df.head() , df.tail()


def clean_data(filepath:str , s=None,c=None , n=None , r=None ,t=None , h=None , g=None): ##r is the  column with non-alpha-numeric-characters we r cleaning and t is a column with street address
    df = read_data(filepath)
    df.drop_duplicates(inplace=True)
    if s=='drop_column':
        df.drop(columns=c , inplace=True)


    elif n=='drop_null':
        df.dropna(inplace=True)

    elif n=='fill_null_with_0':
        df.fillna(0 , inplace=True)

    elif r=='column_with_non-alphanumeric_characters': ##the goal is to remove the non-alphanumeric characters here
        df[r] = df[r].apply(lambda x: re.sub(r'[^a-zA-Z0-9]' , '' , str(x)) if pd.notnull(x) else x)

    elif t=="column_to_standardise":
        df[t] = df[t].apply(lambda x: str(x).replace(h, g) if pd.notnull(x) else x) ##h is string we are standardising g is the standardised string
    
    return df

def transform_data(filepath:str,n:int,filepath1=None  , s=False , h=None , t=None , g=False,c=None , a=None,b=None , d=None , e=None ):
    df = read_data(filepath)
    le = LabelEncoder()
    oe = OneHotEncoder(sparse=False)
    scler = StandardScaler()
    if s==True and t !=None: ##label encoding a categorical variable h in a train-set and the same feature h in test-set (which is t) is being tranformed
        df[h] = le.fit_transform(df[h])
        df[t] = le.transform(df[t])
    elif s==True and t==None:
        df[h] = le.fit_transform(df[h]) ##no respective test feature so only fit-transform


    if g==True and c=='one_hot_with_pandas':
        df[a] = pd.get_dummies(df[a])

    elif g==True and c=='one_hot_with_scikit-learn' and b != None:
        df[a] = oe.fit_transform(df[a])
        df[b] = oe.transform(df[b]) ##respective test-feature to a

    elif g==True and c=='one_hot_with_scikit-learn' and b==None:
        df[a] = oe.fit_transform(df[a])

    if filepath1==None:
        scled_data = scler.fit_transform(df)
        df = pd.DataFrame(scled_data , df.columns)

    elif filepath1 != None:
        df1 = read_data(filepath1)
        scaled_data = scler.fit_transform(df)
        df = pd.DataFrame(scaled_data, columns=df.columns)
        scled_test_data = scler.transform(df1)
        df1 = pd.DataFrame(scled_test_data , df.columns) 

    if d !=None and e !=None:
        df = pd.concat([df[d] , df[e]] , axis=n , ignore_index=True)


    return (df , df1) if df1  is not None else df


    

    

    


        
    








    


    

    

  


















