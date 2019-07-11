
# coding: utf-8

# In[947]:

import pandas as pd
df= pd.read_csv('Loan_Training.csv')
df.head()


# In[948]:

#create a copy of the original data
df1 = df.copy()

#drop columns that have more than 3 N.A values
df1 = df1.dropna(thresh=25)


#delete columns that are irrelevant
del df1['Loan Title']
del df1['Loan ID']
del df1['City']
del df1['Earliest CREDIT Line']
print(df1.shape)
import numpy as np

#changing values of certain columns from percentage (%) to decimal

df1['Interest Rate'] = df1['Interest Rate'].replace('%','',regex=True).astype('float')/100
df1['Debt-To-Income Ratio'] = df1['Debt-To-Income Ratio'].replace('%','',regex=True).astype('float')/100    
df1['Revolving Line Utilization'] = df1['Revolving Line Utilization'].replace('%','',regex=True).astype('float')/100


df1.head()

#normalizing the values of education

df1['Education'] = df1['Education'].replace(np.nan, 'Not Applicable', regex=True)
df1['Education'] = df1['Education'].replace(['n/a','none','None','NONE','undefined'], 'Not Applicable', regex=True)
searchForUni = ['uni','A&M','UCLA','V.C.U','unc','vcu','sdsu','san diego','Penn State','SUNY','Technology','UC','Dartmouth','FMU']
df1.loc[df1['Education'].str.contains('|'.join(searchForUni),case=False),'Education'] = 'University'
searchForCollege = ['college','ccny','NYCCT','coll']
df1.loc[df1['Education'].str.contains('|'.join(searchForCollege),case=False),'Education'] = 'College'
searchForVocational = ['tech','institute','Academy','Navy','vocational']
df1.loc[df1['Education'].str.contains('|'.join(searchForVocational),case=False),'Education'] = 'Vocational/Military School'

getTheRest = ['Not Applicable','University','College','Vocational/Military School']
df1.loc[~df1['Education'].isin(getTheRest),'Education'] = 'High School and Others'  



# In[949]:




# In[951]:




# In[952]:

#normalizing values of Employment Length

df1['Employment Length'] = df1['Employment Length'].replace('n/a','< 1 year',regex=True)



# In[ ]:




# In[954]:

#changing values of FICO RANGE : NaN into mode
df1['FICO Range'] = df1['FICO Range'].replace(np.nan, '680-684', regex=True)


# In[955]:




# In[956]:

#helper functions to generate random numbers that follow a normal distribution with mean and sigma determined from the exisiting
#values inthe column
def NaNgenerateRandom(df):
    mu = df.mean()
    std = df.std()
    a = df.values
    m = np.isnan(a) #no. of NANs
    a[m] = np.random.normal(mu,std,size=m.sum())
    a[m] = [int(round(x)) for x in a[m]]
    a[m] = [abs(x) for x in a[m]]
    return df
#to ensure that 0.5 does not get rounded up to 1
def NaNgenerateRandomPercent(df):
    mu = df.mean()
    std = df.std()
    a = df.values
    m = np.isnan(a) #no. of NANs
    a[m] = np.random.normal(mu,std,size=m.sum())
    a[m] = [abs(x) for x in a[m]]
    return df


# In[957]:

#more normalizing
df1['Months Since Last Record'] = NaNgenerateRandom(df1['Months Since Last Record'])


# In[958]:




# In[959]:

#Randomize NaN with 0-5 based on their respective probabilities in the columns.
odds = [0.95405,0.04445,0.0014,0.00005,0.00005]
df1['Public Records On File'] = df1['Public Records On File'].replace(np.nan,np.random.choice(np.arange(0, 5), p=odds),regex=True)


# In[ ]:




# In[961]:

df1['Months Since Last Delinquency'] = NaNgenerateRandom(df1['Months Since Last Delinquency'])


# In[962]:




# In[963]:

df1['Delinquencies (Last 2 yrs)'] = NaNgenerateRandom(df1['Delinquencies (Last 2 yrs)'])


# In[964]:




# In[965]:

#randomize based on existing probabilities of the values in the column
df1['Delinquent Amount'].value_counts()
odds = ['0'] * 19968 +['334'] * 1 + ['3941.0'] * 1
df1['Delinquent Amount'] = df1['Delinquent Amount'].replace(np.nan,np.random.choice(odds),regex=True)


# In[966]:

df1['Revolving Line Utilization'] = NaNgenerateRandomPercent(df1['Revolving Line Utilization'])


# In[967]:

df1['Accounts Now Delinquent'].value_counts()
odds = [0.99,0.01]
df1['Accounts Now Delinquent'] = df1['Accounts Now Delinquent'].replace(np.nan,np.random.choice(np.arange(0, 2), p=odds),regex=True)


# In[968]:

df1['Revolving CREDIT Balance'] = NaNgenerateRandom(df1['Revolving CREDIT Balance'])


# In[969]:


df1['Inquiries in the Last 6 Months'] = NaNgenerateRandom(df1['Inquiries in the Last 6 Months'])


# In[970]:

df1['Total CREDIT Lines'] = NaNgenerateRandom(df1['Total CREDIT Lines'])


# In[971]:

df1['Open CREDIT Lines'] = NaNgenerateRandom(df1['Open CREDIT Lines'])


# In[972]:

#ensure no negative values
df1.loc[(df1['Monthly Income']< 0,'Monthly Income')] = 0


# In[973]:

df1['Debt-To-Income Ratio'] = NaNgenerateRandomPercent(df1['Debt-To-Income Ratio'])
df1['Interest Rate'] = NaNgenerateRandomPercent(df1['Interest Rate'])


# In[974]:

df1.loc[(df1['Total Amount Funded']< 0, 'Total Amount Funded')] = 0


# In[975]:

df1.loc[(df1['Monthly PAYMENT']< 0, 'Monthly PAYMENT')] = 0


# In[976]:

df1.loc[(df1['Amount Requested']< 0, 'Amount Requested')] = 0
df1.loc[(df1['Amount Funded By Investors']< 0, 'Amount Funded By Investors')] = 0


# In[977]:

#get rid of the 'months' behind 36 and 60
odds = ['36 months'] * 16725 + ['60 months'] * 3275
df1['Loan Length'] = df1['Loan Length'].replace(np.nan,np.random.choice(odds),regex=True)
df1['Loan Length'] = df1['Loan Length'].replace('months','',regex=True)


# In[ ]:




# In[ ]:

#write to file


# In[980]:

df1.to_csv('cleanTrain.csv', encoding='utf-8',index= False)


# In[981]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[721]:



