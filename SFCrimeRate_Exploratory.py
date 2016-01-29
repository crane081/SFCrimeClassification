# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:06:54 2016

This code uses Naive Bayes Classifier (Multinomial NB) 

for Kaggle challenge: San Francisco crime classification

https://www.kaggle.com/c/sf-crime/data?test.csv.zip

@author: hhcrane
"""

import warnings
warnings.filterwarnings("ignore") #ignore all warnings

from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd

df=pd.read_csv('train.csv')

from pdb import set_trace as bp
from sklearn.naive_bayes import MultinomialNB

#check columns:
#df.columns

#check missing data:
#df.isnull().sum()

#%% Define target and features
y=df.Category #target: category of crime
N_c=np.unique(y) #unique categories
N_y=len(y) #number of samples
print('%d categories of crimes in %d samples' % (len(N_c),N_y))
#%% Exploratory analysis:  Crime category distribution
from collections import Counter

category=df.Category
Crime=Counter(category).most_common() #use Crime to check the occurces of each crime
Crime_dec=[]
for key,value in Crime:
    Crime_dec.append(key)
    
Count_crime=np.zeros(len(Crime))
t=0
for i in Crime_dec:
    Count_crime[t]=Counter(category).get(i)/len(y)
    t=t+1

# plot crime by category distribution
import matplotlib.pyplot as plt
plt.bar(np.arange(len(Crime)),Count_crime)
plt.xticks(np.arange(len(Crime)), Crime_dec,ha='left',rotation=90)
plt.title('Crime category Distribution (sorted)')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

#%% compute the probability of each crime happen:
# on each day of the week: P(crime|weekday)
# in each PdDistrict: P(crime|PdDistrict)
# at what time during the day (0-24) 
# in which month 
# in which year 
# with what resolution 
DayofWeek=df.DayOfWeek.values
PdDistrict=df.PdDistrict.values
Resolution = df.Resolution

a=pd.to_datetime(df.Dates)
hour=a.apply(lambda x: x.hour)
df['Hour']=hour
month=a.apply(lambda x: x.month)
df['month']=month
year=a.apply(lambda x: x.year)
df['year']=year

Counter(DayofWeek).most_common() #list days of crimes in decreasing order
Counter(PdDistrict).most_common() #list district of crimes in decreasing order
Counter(hour).most_common() #list hours of crimes in decreasing order
Counter(Resolution).most_common() #list hours of crimes in decreasing order
Counter(month).most_common() #list hours of crimes in decreasing order
Counter(year).most_common() #list hours of crimes in decreasing order

#%% By weekday
Days=Counter(DayofWeek).most_common()
Days_dec=[]
for key,value in Days:
    Days_dec.append(key)
    
Count_days=np.zeros(7)
t=0
for i in Days_dec:
    Count_days[t]=Counter(DayofWeek).get(i)/len(DayofWeek)
    t=t+1

# plot crime by day distribution
import matplotlib.pyplot as plt
plt.bar(np.arange(7),Count_days)
plt.xticks(np.arange(7), Days_dec,ha='left', rotation=45)
plt.title('Crime Weekday Distribution (sorted)')
plt.show()
#%% By PD district
Dist=Counter(PdDistrict).most_common()
Dist_dec=[]

for key,value in Dist:
    Dist_dec.append(key)
    
Count_dist=np.zeros(len(Dist_dec))
t=0
for i in Dist_dec:
    Count_dist[t]=Counter(PdDistrict).get(i)/len(DayofWeek)
    t=t+1
 
# plot crime by PDdistrict distribution
import matplotlib.pyplot as plt
plt.bar(np.arange(len(Dist_dec)),Count_dist)
plt.xticks(np.arange(len(Dist_dec)), Dist_dec,ha='left', rotation=45)
plt.title('Crime PdDistrict Distribution (sorted)')
plt.show()

#%% By hours in the day
Hours=Counter(hour).most_common()
Hour_dec=[]

for key,value in Hours:
    Hour_dec.append(key)
    
Count_hour=np.zeros(len(Hour_dec))
t=0
for i in Hour_dec:
    Count_hour[t]=Counter(hour).get(i)/len(DayofWeek)
    t=t+1
 
# plot crime by PDdistrict distribution
import matplotlib.pyplot as plt
plt.bar(np.arange(len(Hour_dec)),Count_hour)
plt.xticks(np.arange(len(Hour_dec)), Hour_dec,ha='left', rotation=45)
plt.title('Crime Hour Distribution (sorted)')
plt.show()

#%% By month
Months=Counter(month).most_common()
Month_dec=[]

for key,value in Months:
    Month_dec.append(key)
    
Count_month=np.zeros(len(Month_dec))
t=0
for i in Month_dec:
    Count_month[t]=Counter(month).get(i)/len(DayofWeek)
    t=t+1
 
# plot crime by PDdistrict distribution
import matplotlib.pyplot as plt
plt.bar(np.arange(len(Month_dec)),Count_month)
plt.xticks(np.arange(len(Month_dec)), Month_dec,ha='left', rotation=45)
plt.title('Crime Month Distribution (sorted)')
plt.show()


#%% By resolution
Reso=Counter(Resolution).most_common()
Reso_dec=[]

for key,value in Reso:
    Reso_dec.append(key)
    
Count_reso=np.zeros(len(Reso_dec))
t=0
for i in Reso_dec:
    Count_reso[t]=Counter(Resolution).get(i)/len(DayofWeek)
    t=t+1
 
# plot crime by PDdistrict distribution
import matplotlib.pyplot as plt
plt.bar(np.arange(len(Reso_dec)),Count_reso)
plt.xticks(np.arange(len(Reso_dec)), Reso_dec,ha='right', rotation=45)
plt.title('Crime Resolution Distribution (sorted)')
plt.show()

#%% create Day or Night based on timestamp
#day: 7am-7pm, night: 7pm-7am
from datetime import time
day_start=time(7)
day_end=time(19)
night_start=time(19)
night_end=time(7)

periods = {'day':[day_start, day_end], 'night':[night_start, night_end]}

def f(x, periods=periods):
    for k, v in periods.items():
        if x.hour >= v[0].hour and x.hour < v[1].hour:
            return k
    return 'unknown_period'
a=pd.to_datetime(df.Dates)

df['periods'] = np.where((day_start.hour <= a.apply(lambda x: x.hour)) & (a.apply(lambda x: x.hour) <= day_end.hour), 'day', 'night')     

#%% plot crimes by categories on the map using coordinates
import matplotlib.pyplot as plt

#id=np.where(df.X<-122.0)[0]
unique_y=np.unique(df.Category)

for i in unique_y:
    id=np.where(df.Category==i)[0]
#    labels=df.periods[id] #group by time of the day
    labels=df.PdDistrict[id] #group by PdDistrict

    X=df.X[id]
    Y=df.Y[id]
    
    df1=pd.DataFrame(dict(x=X,y=Y,label=labels))
    
    groups=df1.groupby('label')
    fig,ax=plt.subplots()
    ax.margins(.05)
    
    for name,group in groups:
        ax.plot(group.x,group.y,marker='o',linestyle='')
#    plt.legend()
    #plt.show()
    
    map = Basemap(
        projection = 'cyl',
        lon_0=-122.44576,
        lat_0=37.752303,
        llcrnrlon=-122.52469,
        llcrnrlat=37.69862,
        urcrnrlon=-122.33663,
        urcrnrlat=37.82986,
        resolution = 'h')
        
    mapdata=np.loadtxt('sf_map_copyright_openstreetmap_contributors.txt')
    plt.imshow(mapdata, cmap = plt.get_cmap('gray'), extent=[-122.52469, -122.33663, 37.69862, 37.82986])
    plt.title(i)    
    plt.show()
      
    bp() #use quit() to quit ipdb status, use c to continue

#%% convert variables
x1=pd.get_dummies(df[['DayOfWeek','PdDistrict','Resolution','Descript']])
#x1=pd.get_dummies(df[['DayOfWeek','PdDistrict']])

X=np.hstack((x1,df[['Hour','month','year']]))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y = le.fit_transform(y) #transform original target to numbers


#%% split to training and testing data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = .3)

#%% Multinomial Naive Bayes
clf=MultinomialNB()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
prob=clf.predict_log_proba(X_test) #log prob for each test point
acc=sum(pred==y_test)/len(y_test)
print(acc) #.99

"""
Note: test data only have Dates (year, month, Hour), DayofWeek, PdDistrict,
Address, X and Y. 

"""

