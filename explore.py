# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 16:23:06 2014

@author: dlaraujo
"""

""" This is code using csv package
import csv as csv
import numpy as np

csv_file_object = csv.reader(open('train.csv', 'rb')) 
header = csv_file_object.next() 
# 'PassengerId' - 0
# 'Survived' - 1
# 'Pclass' - 2
# 'Name' - 3
# 'Sex' - 4
# 'Age' - 5
# 'SibSp' - 6 
# 'Parch' - 7 
# 'Ticket' - 8 
# 'Fare' - 9
# 'Cabin' - 10 
# 'Embarked' -11

data=[] 

for row in csv_file_object:
    data.append(row)
data = np.array(data) 

#print set(data[0::,10])

"""

# Lets use Pandas package and the dataframe data structure
import pandas as pd
import numpy as np
import pylab as P

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('train.csv', header=0)

#print df.head(3) 
#print df.dtypes

# pandas takes all of the numerical columns and quickly calculated the
# mean, std, minimum and maximum value
print df.describe()
print df['Age'].mean()
print df[ ['Sex', 'Pclass', 'Age'] ] 
print df[df['Age'] > 60]
print df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]#
print df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

# distribuition of males classes that do not survived  
print 'MALES DID NOT SURVIVED BY CLASS:'
for i in range(1,4):
    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) & (df['Survived'] == 0)])

# distribuition of males classes that do  survived  
print 'MALES DID  SURVIVED BY CLASS:'
for i in range(1,4):
    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) & (df['Survived'] == 1)])

#df['Age'].hist()
#P.show()

## FILL IN NULL VALUES AND CREATE NUMERIC COLUMNS IN THE MODEL

df['SexFill'] = df['Sex'] 
# add new column to describe gender as 0 (female) or 1 (male)
df['SexFill'] = df['SexFill'].map( {'female': 0, 'male': 1} ).astype(int)

 
df['EmbarkedFill'] = df['Embarked']
# fill values that are null with dummy values TODO 
df.loc[ (df.Embarked.isnull()),'EmbarkedFill'] = 'H' 
# add new column to describe port of Embarked as 0 (C = Cherbourg), 1 (Q = Queenstown), 2 (S = Southampton)
df['EmbarkedFill'] = df['EmbarkedFill'].map( {'C': 0, 'Q': 1, 'S': 3, 'H': 4} ).astype(int)


df['AgeFill'] = df['Age']
# fill values that are null with medians
median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['SexFill'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.SexFill == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

# let save also the family size
df['FamilySize'] = df['SibSp'] + df['Parch']

# Pclass had a large effect on survival, and it's possible Age will too
df['Age*Class'] = df.AgeFill * df.Pclass

#df['Age*Class'].astype(int).hist()
#P.show()

df.dtypes[df.dtypes.map(lambda x: x=='object')]

# drop columns we will not use
df = df.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 

train_data = df.values






