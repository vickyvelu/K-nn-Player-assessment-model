import os
import codecs
from bs4 import BeautifulSoup
import pandas as pd
from sklearn import neighbors
import numpy as np

import matplotlib.pyplot as plt
from pylab import title,subplot

#%%
def readDf ( data):
    path = "/Users/Vicky/Documents/Final project/Knn/"
    filelist = os.listdir(path+data)
    df = pd.DataFrame(columns = ['Date', 'Pos', 'Opposition', 'Sav', 'Pas', 'Cmp', 'ChC', 'Mst', 'Mcg', 'Svh', 'Svp', 'Svt', 'Sho', 'Itc', 'Fou', 'Fld', 'Condition', 'Rat', 'Ast', 'Gls'])
    for x in filelist:
        f=codecs.open(data+"/"+x, 'r', 'utf-8')
        document= f.read()
        soup = BeautifulSoup(document, 'lxml') # Parse the HTML as a string
        table = soup.find_all('table')[0] # Grab the first table
        rows = table.find_all('tr') # all rows except the header
        new_table = pd.DataFrame(columns=[head.get_text() for head in rows[0].find_all('th')], index = range(len(rows)-1))
        
        row_marker = 0
        for row in rows[1:]:
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                if(column.get_text()):
                    new_table.iat[row_marker,column_marker] = column.get_text()
                else:
                    new_table.iat[row_marker,column_marker] = 0
                column_marker += 1
            row_marker += 1
        
        new_table = new_table.drop(columns = ['#', 'C.', 'Inf.', 'Key', 'Role'])
        
        new_table = new_table[new_table['Rat'] != 0]
        df = df.append(new_table)
    
    
    df = df.fillna(0)
    
    df = df.drop(columns = ['Date', 'Opposition'])
    df = df.reset_index().drop("index", axis=1)
    df.Pos = [x[:2] for x in df.Pos]
    
    return df

def pred( pos, df, dft):
    dfy = pd.DataFrame(df.Rat)
    dfx = df.drop(columns = 'Rat')
    dfx = dfx.astype('int')
    dfy = dfy.astype('float')
    
    
    
    knn = neighbors.KNeighborsRegressor(5, weights = 'distance')
    kfit = knn.fit(dfx, dfy)   #Unsplit
    
    
    dftx = pd.DataFrame(columns = dfx.columns)
    dftx = dftx.append(dft.drop(columns = 'Rat'))  #Unsplit
    dftx = dftx.fillna(0).astype('int')   #Unsplit
    
    
    y = kfit.predict(dftx)
    
    dfty = pd.DataFrame(dft.Rat)   #Unsplit
    dfty = dfty.astype('float').reset_index().drop("index", axis=1)  #Unsplit
    
    
    f = plt.figure(figsize = (20,10)) 
    title(pos+" Ratings")
    plt.plot(dfty, color='b', lw=2.0, label='Truth')
    plt.plot(y, color='g', lw=2.0, label='Prediction')
    plt.legend(loc=3)
    return y

#%%
data = "html data"
df = readDf(data)
dfs = {k: v.drop(columns = 'Pos') for k,v in df.groupby("Pos")}

data = "test data"   #Unsplit
dfts = readDf(data)   #Unsplit

forExcel = []
for pos,dft in dfts.groupby("Pos"):
    t = pred( pos, dfs[pos], dft.drop(columns='Pos'))
    for i,j in zip(t,dft.Rat):
        forExcel.append([pos, i[0],float(j)])

forExcel = pd.DataFrame(forExcel)
forExcel.columns = ['Position', 'Prediction', 'Original']
forExcel.to_excel('predictions.xlsx', index = False)
