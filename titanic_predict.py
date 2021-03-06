# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 11:37:49 2014

@author: dlaraujo
"""

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
import pylab as P
import re
import math as m
from sklearn.cross_validation import cross_val_score

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# columns not to be used in the model
variables_to_drop = ['Name', 'Sex', 'Ticket', 'PassengerId', 'Age', 'Fare', 'Age*Class', 'SibSp', 'Parch']

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
#if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
#    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values
median_embarked = {}

# i is the Gender (2 values), j is the Pclass (3 values) 
for i in range(0, 2):
    for j in range(0, 3):
        median_embarked[(i,j)] = train_df[(train_df['Gender'] == i) & \
                                        (train_df['Pclass'] == j+1)]['Embarked'].dropna().mode().values

#fill Embarked that are null
for i in range(0, 2):
    for j in range(0, 3):
            train_df.loc[ (train_df.Embarked.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1),\
                    'Embarked'] = median_embarked[(i,j)]

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
# This is the easy way
"""median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age"""

# A smarted way using sex, pclass, embarked to derive median_age in each case
median_ages = {} # np.zeros((3,3))

# i is the Gender (2 values), j is the Pclass (3 values) and k is Embarked (3 values)
for i in range(0, 2):
    for j in range(0, 3):
        for k in range(0, 3):          
            pos = i + j + k    
            median_ages[(i,j, k)] = train_df[(train_df['Gender'] == i) & \
                                        (train_df['Pclass'] == j+1) & \
                                        (train_df['Embarked'] == k)]['Age'].dropna().median()

#fill Age that are null
for i in range(0, 2):
    for j in range(0, 3):
        for k in range(0, 3):   
            train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1) & (train_df.Embarked == k),\
                    'Age'] = median_ages[(i,j,k)]


#train_df['Age'].astype(int).hist()
#P.show()

# create additional AgeInterval
train_df.loc[ (train_df['Age'] < 20), 'AgeInterval' ]  =  0
train_df.loc[ (train_df['Age'] >= 20) & (train_df['Age'] < 40), 'AgeInterval' ]  =  1
train_df.loc[ (train_df['Age'] >= 40) & (train_df['Age'] < 60), 'AgeInterval' ]  =  2
train_df.loc[ (train_df['Age'] >= 60) & (train_df['Age'] < 80), 'AgeInterval' ]  =  3
train_df.loc[ (train_df['Age'] >= 80) , 'AgeInterval' ]  =  4

# create additional FareInterval
train_df.loc[ (train_df['Fare'] < 10), 'FareInterval' ]  =  0
train_df.loc[ (train_df['Fare'] >= 10) & (train_df['Fare'] < 20), 'FareInterval' ]  =  1
train_df.loc[ (train_df['Fare'] >= 20) & (train_df['Fare'] < 30), 'FareInterval' ]  =  2
train_df.loc[ (train_df['Fare'] >= 30), 'FareInterval' ]  =  3

# create additional Title column      
nameSplit =  train_df['Name'].str.split(',').apply(pd.Series, 2)
firstName = nameSplit[1]
title = firstName.str.split('.').str[0]
title = title.map(lambda x: x.lstrip(' ').rstrip('aAbBcC'))

titles_dic = {'Col': 0,
 'Don': 0,
 'Dr': 1,
 'Master': 1,
 'Miss': 2,
 'Mr': 0,
 'Mrs': 2,
 'Ms': 2,
 'Rev': 0,
 'Mme': 0,
 'Capt': 1, 
 'Col': 0, 
 'Don': 1,
 'Jonkheer': 0,
 'Lady': 2,
 'Major': 1, 
 'Mlle': 2,
 'Sir': 1,
 'the Countess': 1
 }
 


train_df['Title'] = title.map( titles_dic ).astype(int)     # Convert all Titles strings to int

# create additional Cabin column
train_df['Cabin'] = train_df['Cabin'].str.extract('(?P<letter>[ABCDEF])')
train_df.loc[ (train_df.Cabin.isnull()), 'Cabin'] = 0
train_df['Cabin'] = train_df['Cabin'].map( {0:0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6})

# lets create additional columns to the model
train_df['FamilySize'] = train_df.SibSp + train_df.Parch # family size
train_df['Age*Class'] = train_df.Age * train_df.Pclass # higher value less likely to survive

train_df.loc[ (train_df['Age*Class'] < 40), 'Age*ClassI' ]  =  0
train_df.loc[ (train_df['Age*Class'] >= 40) & (train_df['Age*Class'] < 80), 'Age*ClassI' ]  =  1
train_df.loc[ (train_df['Age*Class'] >= 80) & (train_df['Age*Class'] < 120), 'Age*ClassI' ]  =  2
train_df.loc[ (train_df['Age*Class'] >= 120), 'Age*ClassI' ]  =  3

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(variables_to_drop, axis=1) 

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
#train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

#train_df['TitleMen'] = 

#train_df['FamilySize'].astype(int).hist()
#P.show()
#train_df['Age*Class'].astype(int).hist()
#P.show()


# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
#if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
#    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

median_embarked = {}

# i is the Gender (2 values), j is the Pclass (3 values) 
for i in range(0, 2):
    for j in range(0, 3):
        median_embarked[(i,j)] = test_df[(test_df['Gender'] == i) & \
                                        (test_df['Pclass'] == j+1)]['Embarked'].dropna().mode().values

#print median_embarked
for i in range(0, 2):
    for j in range(0, 3):
            test_df.loc[ (test_df.Embarked.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1),\
                    'Embarked'] = median_embarked[(i,j)]

# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#All the ages with no data -> make the median of all Ages based on the sex, pclass, embarked
# A smarted way using sex, pclass, embarked to derive median_age in each case
median_ages = {} # np.zeros((3,3))


# i is the Gender (2 values), j is the Pclass (3 values) and k is Embarked (3 values)
for i in range(0, 2):
    for j in range(0, 3):
        for k in range(0, 3):  
            median_ages[(i,j, k)] = test_df[(test_df['Gender'] == i) & \
                                        (test_df['Pclass'] == j+1) & \
                                        (test_df['Embarked'] == k)]['Age'].dropna().median()
 
#print median_ages
for i in range(0, 2):
    for j in range(0, 3):
        for k in range(0, 3):   
            test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1) & (test_df.Embarked == k),\
                    'Age'] = median_ages[(i,j,k)]

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values


# create additional AgeInterval
test_df.loc[ (test_df['Age'] < 20), 'AgeInterval' ]  =  0
test_df.loc[ (test_df['Age'] >= 20) & (test_df['Age'] < 40), 'AgeInterval' ]  =  1
test_df.loc[ (test_df['Age'] >= 40) & (test_df['Age'] < 60), 'AgeInterval' ]  =  2
test_df.loc[ (test_df['Age'] >= 60) & (test_df['Age'] < 80), 'AgeInterval' ]  =  3
test_df.loc[ (test_df['Age'] >= 80) , 'AgeInterval' ]  =  4

# create additional FareInterval
test_df.loc[ (test_df['Fare'] < 10), 'FareInterval' ]  =  0
test_df.loc[ (test_df['Fare'] >= 10) & (test_df['Fare'] < 20), 'FareInterval' ]  =  1
test_df.loc[ (test_df['Fare'] >= 20) & (test_df['Fare'] < 30), 'FareInterval' ]  =  2
test_df.loc[ (test_df['Fare'] >= 30), 'FareInterval' ]  =  3

# create additional Title column      
nameSplit =  test_df['Name'].str.split(',').apply(pd.Series, 2)
firstName = nameSplit[1]
title = firstName.str.split('.').str[0]
title = title.map(lambda x: x.lstrip(' ').rstrip('aAbBcC'))

test_df['Title'] = title.map( titles_dic ).astype(int)     # Convert all Titles strings to int

# create additional Cabin column
test_df['Cabin'] = test_df['Cabin'].str.extract('(?P<letter>[ABCDEF])')
test_df.loc[ (test_df.Cabin.isnull()), 'Cabin'] = 0
test_df['Cabin'] = test_df['Cabin'].map( {0:0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6} ).astype(int)

    # lets create additional columns to the model
test_df['FamilySize'] = test_df.SibSp + test_df.Parch # family size
test_df['Age*Class'] = test_df.Age * test_df.Pclass # higher value less likely to survive

#print(np.unique(test_df['FamilySize']))
#print(np.unique(test_df['Age*Class']))

test_df.loc[ (test_df['Age*Class'] < 40), 'Age*ClassI' ]  =  0
test_df.loc[ (test_df['Age*Class'] >= 40) & (test_df['Age*Class'] < 80), 'Age*ClassI' ]  =  1
test_df.loc[ (test_df['Age*Class'] >= 80) & (test_df['Age*Class'] < 120), 'Age*ClassI' ]  =  2
test_df.loc[ (test_df['Age*Class'] >= 120), 'Age*ClassI' ]  =  3


# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(variables_to_drop, axis=1) 

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
#forest = RandomForestClassifier(n_estimators=200, max_features = 3, max_depth=None, min_samples_split=1)
forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

score = forest.score(train_data[0::,1::], train_data[0::,0] )

print "SCORE: " + str(score)

#labels = train_df["Survived"].values
#features = train_df[train_df.keys()].values
 
#et_score = cross_val_score(forest, features, labels, n_jobs=-1).mean()
 

#et_score = cross_val_score(et, features, labels, n_jobs=-1).mean()
#print("{0} -> ET: {1})".format(train_df.keys(), et_score))

predictions_file = open("davidforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'



