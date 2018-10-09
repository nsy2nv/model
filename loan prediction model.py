# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 00:15:48 2018

@author: Nsikan Udo
"""

import pandas as pd
import numpy as np #for mathematics calculations
import seaborn as sns #for data visualization
import matplotlib.pyplot as plt #for plotting graphs
#%matplotlib inline
#import warnings #To ignore any warnings
#warning.filterwarnings('ignore')

#Reading data into the program

train = pd.read_csv('C:\\Users\ALHASNA AGENCY\\Downloads\\train_u6lujuX_CVtuZ9i.csv')
test = pd.read_csv('C:\\Users\ALHASNA AGENCY\\Downloads\\test_Y3wMUE5_7gLdaTN.csv')

#Make a copy of the data set
train_original = train.copy()
teat_original = test.copy()

#Doing EDA for the data set

print(train.columns)
print(test.columns)

print(train.info)
print(train.describe())

print(train.dtypes)
print(test.shape)
print(len(train) + len(test))

""" We look at the target variable. Since it is a categorical variable, 
We examine it frequency table, percentage distribution and bar plot """

my_target = train['Loan_Status'].value_counts()
print(my_target)

# Normalize can be set to True to print proportions instead of number 
my_target_prop = train['Loan_Status'].value_counts(normalize=True)
print(my_target_prop)

#Plot bar plot of Loan_Status
train['Loan_Status'].value_counts().plot.bar(title='Loan Status')
plt.show()

#Plot bar for independent categorical variables

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize = (20,10), title = 'Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.show()

#Plot bar for independent variable (Ordinal)

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

plt.show()

#Plot Independent variables (Numeric)

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize = (16,5))
plt.show()

#Segregate the ApplicantIncome by Education

train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")

#Ploting Categorical Independent Variables vs Target Variabe

Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()


"""Numerical Independent Variable vs Target Variable
We will try to find the mean income of people for which the loan has been approved 
vs the mean income of people for which the loan has not been approved."""

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

#Bin the Applicant Incomes based on values

bins= [0,2500,4000,6000,81000]
group = ['Low','Average','High','Very High']
train['Income_bin'] = pd.cut(train['ApplicantIncome'], bins, labels = group)

Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
plt.xlabel('ApplicantIncome')
p = plt.ylabel('Percentage')
print(train['Income_bin'])

#We will analyze the coapplicant income and loan amount variable in similar manner
bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome')
P = plt.ylabel('Percentage')

#Let us combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status.

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')

#Let’s visualize the Loan amount variable.

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

#Drop the bins and replace Y with 1 and N with 0

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

#Now lets look at the correlation between all the numerical variables

matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
     
train.isnull().sum()    

#filling in missing categorical variables
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

















































