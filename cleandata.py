#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


#select data
df = pd.read_csv('rawdata.csv')
leisure = df[['hQ1ra','hQ1rb','hQ1re','hQ1rp','hQ1rq','hQ3ra','hQ3rb']]
mindset = df[['Q30rae','Q30raf','Q30rah','Q30rak','Q30ral','Q30ran','Q30rao','Q30raq','Q30rat','Q30rbm']]
food = df.iloc[:,56:187]
demographic = df[['S2','S3','D5','D6','D9','D10ra','D10rb','D11']]
exercise = df['Q6']
store = df.iloc[:,187:196]

#convert categorical to numerical
for i in range(mindset.shape[1]):
    mindset.iloc[:,i] = mindset.iloc[:,i].map({'Strongly agree':4, 'Agree somewhat':3, 'Disagree somewhat':2,'Strongly disagree':1})
mindset['Q30rbm'] = mindset['Q30rbm'].map({1:4, 2:3, 3:2,4:1})
for i in range(food.shape[1]):
    food.iloc[:,i] = food.iloc[:,i].map({'Every day':365,'Several times a week':156,'Several times a month':72,'Once a month':12,'Several times a year':5,'Once a year or less':1,'Never':0})   
for i in range(store.shape[1]):
    store.iloc[:,i] = store.iloc[:,i].map({'Two times a week or more':100,'About once a week':50,'About once every two weeks':24,'About once a month':12,'Less than monthly':6,'Not in the past several months':0})
 

exercise = df['Q6'].map({'Rarely or never':-2, 'Occasionally, but less than once a month':-1, 'A few times each month':1,'1 time per week':2,"2-3 times per week":3,"4 or more times per week":5})


demographic['D5'] = demographic['D5'].map({'Some high school or less':1, 'Graduated high school':2, 'Trade or technical school':3,'Some college or Associate degree':4,"Graduated college/Bachelor's degree":5,"Attended graduate school or received Advanced degree (Master's, Ph.D.)":6,'Decline to answer':0}).astype(int)
demographic['D6'] = demographic['D6'].map({'Less than $15,000':1, '$15,000 but less than $25,000':2, '$25,000 but less than $35,000':3,'$35,000 but less than $50,000':4,"$50,000 but less than $75,000":5,"$75,000 but less than $100,000":6,'$100,000 but less than $200,000':7,'$200,000 but less than $300,000':8,'$300,000 but less than $500,000':9,'$500,000 or over':10,'Decline to answer':0}).astype(int)
demographic['inch'] = demographic['D10ra'] *12 + demographic['D10rb']
demographic['BMI'] = demographic['D9'] /demographic['inch']/ demographic['inch'] *703
demographic['BMI'] = demographic['BMI'].round(2)
demographic['S2'] = demographic['S2'].map({'Female':1,'Male':0})

#cleandata without drop
cleandata = pd.concat([leisure,mindset,food,exercise,demographic,store],axis=1)

#drop outlier and unanswered value
df1 = cleandata[cleandata['BMI'] < 50]
df2 = df1[df1['BMI'] > 10]
df3 = df2[(~df2['D5'].isin([0]))]
df4 = df3[(~df3['D6'].isin([0]))]

df4.to_csv(r'/Users/apple/Desktop/df4.csv', index = False)


#correlation matrix
sns.heatmap(leisure.corr(),annot=True,fmt='.2f',cmap= 'coolwarm',annot_kws={'size':7, 'color':'black'})
sns.heatmap(mindset.corr(),annot=True,fmt='.2f',cmap= 'coolwarm',annot_kws={'size':7, 'color':'black'})


#kmeans
sns.set()
from sklearn.cluster import KMeans
kmeans.inertia_
wcss = []
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(finalDf) #finalDf is from PCA result
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel
plt.show()

kmeans = KMeans(4) 
kmeans.fit(finalDf)
identified_clusters = kmeans.fit_predict(finalDf)
data_with_clusters = finalDf.copy()
data_with_clusters['Cluster'] = identified_clusters

plt.scatter(data_with_clusters.iloc[:,0], data_with_clusters.iloc[:,1], c=data_with_clusters['Cluster'], cmap='rainbow')

plt.xlim(-5,5)
plt.ylim(-6,6)
plt.xlabel('BMI')
plt.ylabel('pca mindset')
plt.show()

#export kmeans result
r1 = finalDf[(kmeans.labels_ ==0)]
r2 = finalDf[(kmeans.labels_ ==1)]
r3 = finalDf[(kmeans.labels_ ==2)]
r4 = finalDf[(kmeans.labels_ ==3)]
r1 = pd.DataFrame(data = r1)
r1['cluster'] = 1
r2 = pd.DataFrame(data = r2)
r2['cluster'] = 2
r3 = pd.DataFrame(data = r3)
r3['cluster'] = 3
r4 = pd.DataFrame(data = r4)
r4['cluster'] = 4

BMI_mindset = pd.concat( [r1,r2,r3,r4], axis=0 )
BMI_mindset['id'] = BMI_mindset.index
BMI_mindset.to_csv(r'/Users/apple/Desktop/BMI_mindset.csv', index = False)




