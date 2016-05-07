'''Gradient Boosted Logistic Regression '''
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.decomposition import PCA,TruncatedSVD
print "Loading data.."
cnt = 0
unsatisfiedCnt = 0
satisfiedCnt = 0
trainds = [ ]
trainlabels = [ ]
testds = [ ]
testlabels = [ ]
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
    	if cnt == 0:
            cnt = cnt+1
            continue
        f = float(row[len(row)-1])
        l = [float(x) for x in row[:len(row)-1]]
    	if cnt <= 15000:
            testds.append(l)
            testlabels.append(f)
    	else:
            trainds.append(l)
            trainlabels.append(f)
        cnt = cnt+1

print "Finished loading data..."
dtrain = xgb.DMatrix(trainds,trainlabels)
dtest = xgb.DMatrix(testds,testlabels)
dtestLabels = dtest.get_label();
param = {'max_depth': 5,'gamma ':0.8,'booster':'gbtree','max_delta_step':10,'alpha':0.4,'lambda':0.4, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric':'error'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 5
bst = xgb.train(param, dtrain, num_round,watchlist)
# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print "error: " + str((sum(1 for i in range(len(preds)) if int(preds[i] >= 0.5) != labels[i])/ float(len(preds))));
x,y,thresholds =roc_curve(testlabels,preds,1)
plt.figure()
plt.plot(x,y)
plt.show()

print (auc(x,y))

print "MCCR"
count1 = 0
count2 = 0
correct1 = 0
correct2 = 0
for index in range (len(preds)):
    out = preds[index]
    target = testlabels[index]
    if target == 0:
        count1 = count1 + 1
        if out < 0.5:
			correct1 = correct1 + 1
    if target==1:
        count2 = count2 + 1
        if out >=0.5:
			correct2 = correct2 + 1

print correct1
print correct2
print count1
print count2
print min (float(correct1)/count1,float(correct2)/count2)
