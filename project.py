# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 07:27:37 2020

@author: P Srihari
"""

import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import csv
import matplotlib.pyplot as pyplt 
from sklearn.metrics import *
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn import tree
source = open('inplog.log', "r")
dest = open('testlog.csv',"w")
header = csv.DictWriter(
    dest, fieldnames=["size", "param","code","label","url"])
header.writeheader()

def get_data(source):
	reg = '([(\d\.)]+) - - \[(.*?)\] "(.*?)" (\d+) (.+) "(.*?)" "(.*?)"'
	data = {}
	for l in source:
		l=l.replace(',','_')
		l = re.match(reg,l).groups()
		s = str(l[4]).rstrip('\n')
		rc = l[3]
		u = l[2]
		p = len(u.split('&'))
		ulen = len(u)
		if '-' in s:
			s = 0
		else:
			s = int(s)
		if (int(rc) > 0):
			vals = {}
			vals['s'] = int(s)
			vals['p'] = int(p)
			vals['len'] = int(ulen)
			vals['rc'] = int(rc)
			data[u] = vals
	return data


def data_lab(d,ld):
	for i in d:
		attack = '0'
		samples = ['%3b','honeypot', 'union', '%3c' , 'eval', 'sql' '%3e', 'xss']
		if any(sample in i.lower() for sample in samples):
			attack = '1'
		row = str(d[i]['len']) + ',' + str(d[i]['p']) + ',' + str(d[i]['rc']) + ',' + attack + ',' + i + '\n'
		ld.write(row)
	print (str(len(d)) + ' were read as test data')

data_lab(get_data(source),dest)
source.close()
dest.close()


def d_val(csv_data):
        t = 'label'
        f = csv_data.columns.drop([t])
        features = csv_data[f]
        labels = csv_data[t]
        return labels, features

train_d = pd.read_csv("trainlog.csv")
del train_d['url']
train_d.dropna()
test_d = pd.read_csv("testlog.csv")
del test_d['url']
test_d.dropna()

x_test , x_train = d_val(train_d)
y_test, y_train = d_val(test_d)

print("training data \n", train_d.head())
print("training data shape " ,train_d.shape)

print("testing data \n", test_d.head())
print("testing data shape " ,test_d.shape)

pyplt.scatter(range(train_d.shape[0]), np.sort(train_d['label'].values))
pyplt.xlabel('requests')
pyplt.ylabel('Status')
pyplt.title("Anomaly status")
pyplt.show()

print("\n\nxxxxxxxxx K-means Clustering xxxxxxxxx\n")

sse = {}
for i in range(1, 10):
    elb = KMeans(n_clusters=i, max_iter=1000).fit(train_d)
    sse[i] = elb.inertia_ 
pyplt.figure()
pyplt.plot(list(sse.keys()), list(sse.values()))
pyplt.xlabel("Number of cluster")
pyplt.ylabel("SSE")
pyplt.show()

kmeans = KMeans(n_clusters=2, max_iter=125).fit(test_d)

print("\n Created clusters are \n")
pyplt.title('K Means cluster')
pyplt.scatter(test_d.iloc[:,1],test_d.iloc[:,1])
pyplt.show()

print( "\n\nxxxxxxxxx Isolation Forest Classifier xxxxxxxxx\n")
isf=IsolationForest(n_estimators=10, max_samples='auto')
isf.fit(x_train,x_test)
y_pred = isf.predict(y_train)
for i in range(len(y_pred)):
    if(y_pred[i] == -1):
        y_pred[i] = 0
print ( "The acurracy score is ",accuracy_score(y_test,y_pred))


print ("\n\nxxxxxxxxx Logistic Regression Classifier xxxxxxxxx\n")
lr = LogisticRegression()
lr.fit(x_train, x_test)

y_pred = lr.predict(y_train)
print("The system is trained")
print("The accuracy of the system is " ,lr.score(y_train, y_test ))

cm = confusion_matrix(y_test, y_pred)
print("\n Consfusion Matrix")
print(cm)
sns.heatmap(cm, annot=True)

logit_roc_auc = roc_auc_score(x_test, lr.predict_proba(x_train)[:,1])
fpr, tpr, thresholds = roc_curve(x_test, lr.predict_proba(x_train)[:,1])
pyplt.figure()
pyplt.plot(fpr, tpr)
pyplt.plot([0, 1], [0, 1],'r--')
pyplt.xlim([0.0, 0.8])
pyplt.ylim([0.0, 0.9])
pyplt.xlabel('FP')
pyplt.ylabel('TP')
pyplt.show()

print ("\n\nxxxxxxxxx Decision Tree Classifier xxxxxxxxx\n")
dtc = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
dtc = dtc.fit(x_train, x_test)
y_pred = dtc.predict(y_train)
print ("decision tree accuracy is ", accuracy_score(y_test, y_pred))
pyplt.figure(figsize=(25,10))
tree.plot_tree(dtc);