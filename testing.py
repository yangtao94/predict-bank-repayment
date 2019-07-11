
# coding: utf-8

# In[186]:

import pandas as pd
df= pd.read_csv('Loan_ToPredict.csv')
df.head()


# In[220]:

#create a copy of the original data
df1 = df.copy()

#threshold = 0.2 * 20000
#df1 = df1.dropna(axis=1,thresh= threshold)
#df1 = df1.dropna(thresh=25)


#delete columns that are irrelevant
del df1['Loan Title']
#del df1['Loan ID']
del df1['City']
del df1['Earliest CREDIT Line']
#del df1['Education']
#del df1['Months Since Last Record']
print(df1.shape)
import numpy as np
#changing values of certain columns
df1['Revolving Line Utilization'] = df1['Revolving Line Utilization'].replace(np.nan,'0%',regex = True)
df1['Interest Rate'] = df1['Interest Rate'].replace(np.nan,'11.14%',regex = True)
df1['Debt-To-Income Ratio'] = df1['Debt-To-Income Ratio'].replace(np.nan,'0%',regex = True)


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



# In[ ]:




# In[221]:

#df1['Education'].value_counts()


# In[222]:

#df1['FICO Range'].unique()


#for column in df1:
#    print(column, " unique values!")
#    print(df1[column].unique())


# In[223]:

#normalizing values of Employment Length
df1['Employment Length'].unique()
df1['Employment Length'] = df1['Employment Length'].replace('n/a','< 1 year',regex=True)

df1['Employment Length'].unique()


# In[224]:

df1['Employment Length'].unique()


# In[225]:

#changing values of FICO RANGE : NaN into mode
df1['FICO Range'] = df1['FICO Range'].replace(np.nan, '680-684', regex=True)

df1['FICO Range'].value_counts()


# In[226]:

#df1['Months Since Last Record'].describe()
#change values of month since last record to mode


# In[227]:

def NaNgenerateRandom(df):
    mu = df.mean()
    std = df.std()
    a = df.values
    m = np.isnan(a) #no. of NANs
    a[m] = np.random.normal(mu,std,size=m.sum())
    a[m] = [int(round(x)) for x in a[m]]
    a[m] = [abs(x) for x in a[m]]
    return df

def NaNgenerateRandomPercent(df):
    mu = df.mean()
    std = df.std()
    a = df.values
    m = np.isnan(a) #no. of NANs
    a[m] = np.random.normal(mu,std,size=m.sum())
    a[m] = [abs(x) for x in a[m]]
    return df


# In[228]:

df1['Months Since Last Record'] = NaNgenerateRandom(df1['Months Since Last Record'])


# In[229]:

#df1['Months Since Last Record'].describe()


# In[230]:

#Randomize NaN with 0-5, with their respective probabilities
odds = [0.95405,0.04445,0.0014,0.00005,0.00005]
df1['Public Records On File'] = df1['Public Records On File'].replace(np.nan,np.random.choice(np.arange(0, 5), p=odds),regex=True)


# In[231]:

#df1['Public Records On File'].value_counts()


# In[232]:

df1['Months Since Last Delinquency'] = NaNgenerateRandom(df1['Months Since Last Delinquency'])
#df1['Months Since Last Delinquency'].describe()


# In[233]:

#df1['Delinquencies (Last 2 yrs)'].value_counts()


# In[234]:

df1['Delinquencies (Last 2 yrs)'] = NaNgenerateRandom(df1['Delinquencies (Last 2 yrs)'])


# In[235]:

#df1['Delinquencies (Last 2 yrs)'].value_counts()


# In[236]:

df1['Delinquent Amount'].value_counts()
odds = ['0'] * 8230 +['440'] * 1 + ['27.0'] * 1
df1['Delinquent Amount'] = df1['Delinquent Amount'].replace(np.nan,np.random.choice(odds),regex=True)


# In[237]:

df1['Revolving Line Utilization'] = NaNgenerateRandomPercent(df1['Revolving Line Utilization'])
#df1['Revolving Line Utilization'].value_counts()


# In[238]:

df1['Accounts Now Delinquent'].value_counts()
odds = [0.99,0.01]
df1['Accounts Now Delinquent'] = df1['Accounts Now Delinquent'].replace(np.nan,np.random.choice(np.arange(0, 2), p=odds),regex=True)


# In[239]:

df1['Revolving CREDIT Balance'] = NaNgenerateRandom(df1['Revolving CREDIT Balance'])


# In[240]:

#df1['Inquiries in the Last 6 Months'].value_counts()
df1['Inquiries in the Last 6 Months'] = NaNgenerateRandom(df1['Inquiries in the Last 6 Months'])
#df1['Inquiries in the Last 6 Months'].value_counts()


# In[241]:

df1['Total CREDIT Lines'] = NaNgenerateRandom(df1['Total CREDIT Lines'])
#df1['Total CREDIT Lines'].value_counts()


# In[242]:

df1['Open CREDIT Lines'] = NaNgenerateRandom(df1['Open CREDIT Lines'])


# In[243]:


df1.loc[(df1['Monthly Income']< 0,'Monthly Income')] = 0


# In[244]:

df1['Debt-To-Income Ratio'] = NaNgenerateRandomPercent(df1['Debt-To-Income Ratio'])
df1['Interest Rate'] = NaNgenerateRandomPercent(df1['Interest Rate'])


# In[245]:



df1.loc[(df1['Total Amount Funded']< 0, 'Total Amount Funded')] = 0


# In[246]:



df1.loc[(df1['Monthly PAYMENT']< 0, 'Monthly PAYMENT')] = 0


# In[247]:

df1['Loan Length'].value_counts()


# In[248]:

odds = ['36 months'] * 16725 + ['60 months'] * 3275
df1['Loan Length'] = df1['Loan Length'].replace(np.nan,np.random.choice(odds),regex=True)
df1['Loan Length'] = df1['Loan Length'].replace('months','',regex=True)
df1['Loan Length'].unique()


# In[249]:

df1.loc[(df1['Amount Requested']< 0, 'Amount Requested')] = 0
        
df1.loc[(df1['Amount Funded By Investors']< 0, 'Amount Funded By Investors')] = 0


# In[250]:

df1.head()


# In[251]:

df1.to_csv('cleanToPredict.csv', encoding='utf-8',index= False)


# In[252]:

df3= pd.read_csv('cleanToPredict.csv')
df3.head()


# In[ ]:




# In[ ]:




# In[ ]:




# In[721]:



