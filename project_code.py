'''
Imports
=======
'''
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
import os, gc, shutil
import cv2, SimpleCV

'''
Data Preparation
================
'''
# global convenience variables:
slice_length = 5
# prior_slices = 3      # not currently used
# discount_wgt = 0.5    # not currently used
mins_ahead = [1, 2, 3, 5, 10, 15, 30, 60]

'''
Data Load and Timestamp Preprocessing
=====================================
'''
# import previously-saved 20-day batch of SPX data from correct directory:
cols = ['DATE', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
raw_data = pd.read_table('rawdata/SPX_001_Jan_8_2015.txt', sep=',', header=6, names=cols)

# function to convert timestamps from Unix Epoch to local exchange time:
def convert_ts(stamp):
    stamp = int(stamp) # making sure
    return pd.datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stamp)), '%Y-%m-%d %H:%M:%S')

# convert timestamps in raw data:
converted = []
for i in range(len(raw_data)):
    if raw_data['DATE'][i][0] == 'a':
        converted.append(convert_ts(raw_data['DATE'][i][1:len(raw_data['DATE'][i])]))
    else:
        converted.append(converted[(i-1)] + datetime.timedelta(minutes=1))

# make into DatetimeIndex for use in dataframe:
dti = pd.DatetimeIndex(converted)
clean_cols = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
clean = DataFrame(raw_data.as_matrix(columns=clean_cols), \
    index=dti, columns=clean_cols)

'''
Create Graphs for Image Recognition Analysis
============================================
'''
# create graphs from data of specified time slice:
for i in range(len(raw_data) - slice_length):
    if os.path.isfile('graphics/' + str(i + slice_length) + '.png') == False:
        clean[['CLOSE', 'HIGH', 'LOW', 'OPEN']][i:i+slice_length].plot(legend=False)
        plt.axis('off')
        plt.savefig('graphics/' + str(i + slice_length) + '.png', dpi=25)
        plt.close('all')
        if i % 25 == 0:
            print 'Still processing: i = ' + str(i) + ' and time is ' + time.ctime()
            gc.collect()
    else:
        pass

'''
Prep Graphs for SimpleCV
========================
'''

# copy 70% of each of up/down graphs to training directories
# copy 30% of each of up/down graphs to test directories
up_5s = sum(y['5_ahead'])
dn_5s = len(y) - up_5s
cutoff_up = int((0.7 * up_5s)//1)
cutoff_dn = int((0.7 * dn_5s)//1)
train_up_path = '../data/supervised/up5/'
train_dn_path = '../data/supervised/down5/'
test_up_path = '../data/unsupervised/up5/'
test_dn_path = '../data/unsupervised/down5/'
up_iter = 0
dn_iter = 0

# copy first 70% of each class to individual corresponding directory
for i in range(len(y) - 5):
    cur_file = 'graphics/' + str(i + 5) + '.png'
    if y['5_ahead'][i] == 1:
        if up_iter < cutoff_up:
            shutil.copy(cur_file, train_up_path)
        else:
            shutil.copy(cur_file, test_up_path)
        up_iter = up_iter + 1
    else:
        if dn_iter < cutoff_dn:
            shutil.copy(cur_file, train_dn_path)
        else:
            shutil.copy(cur_file, test_dn_path)
        dn_iter = dn_iter + 1

'''
Feature Creation: Raw Pixel Data
================================
'''
# convenience dict:
rng = np.arange(256)
revd = rng[::-1]
px_dict = dict(zip(rng,revd))

# mapping function to remap pixel values using the above-created dict
def dict_map(item):
    global px_dict
    return px_dict[item]

# instantiate a list container for unrolled pixel data
graph_data = []
# instantiate DTI for the graph data (to join on clean)
graph_dti = clean.index[4:-1]   # image generation loop missed the last minute somehow

# cv2 loop to read graphical data:
for i in range(len(raw_data) - slice_length):
    path = 'graphics/' + str(i + slice_length) + '.png'
    gs_img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    unrolled = []
    for row in range(gs_img_data.shape[0]):
        unrolled = unrolled + map(dict_map, list(gs_img_data[row,:]))
    graph_data.append(unrolled)
    if i % 100 == 0:
        print 'Still processing: i = ' + str(i) + ' and time is ' + time.ctime()
        gc.collect()

gdata = DataFrame(data=graph_data, index=graph_dti)

# eliminate empty feature columns:
to_drop = []
for i in range(gdata.shape[1]):
    if any(gdata.iloc[:,i] > 0):
        pass
    else:
        to_drop.append(i)
gdata.drop(gdata.columns[to_drop])

for i in range(len(to_drop)):
    gdata.drop(gdata.columns[to_drop[i]], axis=1, inplace=True)
    if i % 25 == 0:
        print 'i = ' + str(i) + ' at ' + time.ctime()

gdata.info()    # check what it looks like size-wise afterwards
gdata.shape     # check dims afterwards
gdata.to_pickle('../data/gdata_pickle')         # pickle it since drop loop took 4 hours
gdata = pd.read_pickle('../data/gdata_pickle')  # moved out of git repo; check dir

'''
Feature Creation: Additional Basic Data
=======================================
'''
# generate a few additional basic features from clean data:
# rolling mean, std, min, max of close, and (high-low) ^2 for each observation
RM_CLOSE = pd.rolling_mean(clean.CLOSE, slice_length)
RSTD_CLOSE = pd.rolling_std(clean.CLOSE, slice_length)
RMIN_CLOSE = pd.rolling_min(clean.CLOSE, slice_length)
RMAX_CLOSE = pd.rolling_max(clean.CLOSE, slice_length)
HL_SQD = Series([(clean.iloc[row][1] - clean.iloc[row][2]) ** 2 for row in range(len(clean))], index=clean.index)

# inner join of the above-created Series on the "clean" DF; drop rows with NaN values
X_basic = clean.join([RM_CLOSE, RSTD_CLOSE, RMIN_CLOSE, RMAX_CLOSE, HL_SQD], how='inner')
X_basic = X_basic.dropna()

# create y_basic based on this new dataframe
ahead_basic = []

for i in range(len(X_basic) - max(mins_ahead)):
    current_row = [1 if X_basic.iloc[i+mins_ahead[j],0] > clean.iloc[i,0] else 0 for j in range(len(mins_ahead))]
    ahead_basic.append(current_row)

y_basic = DataFrame(ahead_basic, columns = [str(mins_ahead[i]) + '_ahead' for i in range(len(mins_ahead))])

# trim down X_basic to match y_basic's reduced length
X_basic = X_basic[:-60]


'''
Generate Target Data
====================
'''

# simple binary loop; flexible to whatever is specified in mins_ahead:
ahead = []

for i in range(len(clean) - max(mins_ahead)):
    current_row = [1 if clean.iloc[i+mins_ahead[j],0] > clean.iloc[i,0] else 0 for j in range(len(mins_ahead))]
    ahead.append(current_row)

y = DataFrame(ahead, columns = [str(mins_ahead[i]) + '_ahead' for i in range(len(mins_ahead))])

'''
Modeling: Basic Data
====================
'''
# define 70% cutoff point manually for train/test split
train_inds = int((0.7 * len(X_basic))//1)

#split into train/test sets
Xb_train = X_basic.iloc[:train_inds, :]
yb_train = y_basic.iloc[:train_inds, :]
Xb_test = X_basic.iloc[train_inds:, :]
yb_test = y_basic.iloc[train_inds:, :]

# null accuracy rates for basic dataset (dict may not be great for this)
null_rates_basic = list(y_basic.mean())

##########################
# AdaBoost on basic data #
##########################
ADB_Scores = list(np.zeros(y_basic.shape[1]))

for i in range(y_basic.shape[1]):
    from sklearn.ensemble import AdaBoostClassifier
    adb = AdaBoostClassifier(n_estimators = 100)
    adb.fit(Xb_train, yb_train.iloc[:,i])
    ADB_Scores[i] = adb.score(Xb_test, yb_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(ADB_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(ADB_Scores[i]) + '\t' + str(ADB_Scores[i] - null_rates_basic[i])

###############################
# Random Forest on basic data #
###############################
RF_Scores = list(np.zeros(y_basic.shape[1]))

for i in range(y_basic.shape[1]):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 20)
    rf.fit(Xb_train, yb_train.iloc[:,i])
    RF_Scores[i] = rf.score(Xb_test, yb_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(RF_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(RF_Scores[i]) + '\t' + str(RF_Scores[i] - null_rates_basic[i])

#############################
# Naive Bayes on basic data #
#############################
NB_Scores = list(np.zeros(y_basic.shape[1]))

for i in range(y_basic.shape[1]):
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(Xb_train, yb_train.iloc[:,i])
    NB_Scores[i] = nb.score(Xb_test, yb_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(NB_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(NB_Scores[i]) + '\t' + str(NB_Scores[i] - null_rates_basic[i])

# this performs SUPER poorly, but that's not particularly surprising on these data
# may do significantly better on pixel data, but still, not a very promising model

#####################################
# Logistic Regression on basic data #
#####################################
LR_Scores = list(np.zeros(y_basic.shape[1]))

for i in range(y_basic.shape[1]):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(Xb_train, yb_train.iloc[:,i])
    LR_Scores[i] = lr.score(Xb_test, yb_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(LR_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(LR_Scores[i]) + '\t' + str(LR_Scores[i] - null_rates_basic[i])

# this blows even random forest out of the water on the basic data - interesting

#####################
# SVM on basic data #
#####################
SVM_Scores = list(np.zeros(y_basic.shape[1]))

for i in range(y_basic.shape[1]):
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(Xb_train, yb_train.iloc[:,i])
    SVM_Scores[i] = svc.score(Xb_test, yb_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(SVM_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(SVM_Scores[i]) + '\t' + str(SVM_Scores[i] - null_rates_basic[i])


'''
Modeling: Graphical Data with SimpleCV for Image Recognition
============================================================
'''
# point to ImageSets of up/down training data; extract line features
up5_imgs = SimpleCV.ImageSet('../data/supervised/up5')
up5_lines = [x.findLines() for x in up5_imgs]
dn5_imgs = SimpleCV.ImageSet('../data/supervised/down5')
dn5_lines = [x.findLines() for x in dn5_imgs]
# store training data and targets
temp_data = []
temp_targets = []

for x in up5_lines:
    coord1mean = np.mean([obs[0] for obs in x.coordinates()])
    coord2mean = np.mean([obs[1] for obs in x.coordinates()])
    temp_data.append([np.mean(x.length()), np.mean(x.angle()), coord1mean, coord2mean])
    temp_targets.append(1)

for x in dn5_lines:
    coord1mean = np.mean([obs[0] for obs in x.coordinates()])
    coord2mean = np.mean([obs[1] for obs in x.coordinates()])
    temp_data.append([np.mean(x.length()), np.mean(x.angle()), coord1mean, coord2mean])
    temp_targets.append(0)

# cleanup and properly index
X = np.array(temp_data)
y = np.array(temp_targets)

X = DataFrame(X, columns = ['mean_len', 'mean_angle', 'c1mean', 'c2mean'])
y = Series(y, name=['target'])

X.dropna(inplace=True)
y = y.ix[X.index]

# ready to train
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

clf = LinearSVC()
clf.fit(X, y)
clf2 = LogisticRegression()
clf2.fit(X, y)

up5_test = SimpleCV.ImageSet('../data/unsupervised/up5')
up5_test_lines = [x.findLines() for x in up5_test]
dn5_test = SimpleCV.ImageSet('../data/unsupervised/down5')
dn5_test_lines = [x.findLines() for x in dn5_test]

to_predict_up = []
to_predict_dn = []

for x in up5_test_lines:
    coord1mean = np.mean([obs[0] for obs in x.coordinates()])
    coord2mean = np.mean([obs[1] for obs in x.coordinates()])
    grph = [np.mean(x.length()), np.mean(x.angle()), coord1mean, coord2mean]
    to_predict_up.append(grph)

to_predict_up = DataFrame(to_predict_up, columns = ['mean_len', 'mean_angle', 'c1mean', 'c2mean']).dropna()

up_preds_clf = clf.predict(to_predict_up)
up_preds_clf2 = clf2.predict(to_predict_up)

for x in dn5_test_lines:
    coord1mean = np.mean([obs[0] for obs in x.coordinates()])
    coord2mean = np.mean([obs[1] for obs in x.coordinates()])
    grph = [np.mean(x.length()), np.mean(x.angle()), coord1mean, coord2mean]
    to_predict_dn.append(grph)

to_predict_dn = DataFrame(to_predict_dn, columns = ['mean_len', 'mean_angle', 'c1mean', 'c2mean']).dropna()

dn_preds_clf = clf.predict(to_predict_dn)
dn_preds_clf2 = clf2.predict(to_predict_dn)

# good at predicting on up, not so good at predicting on down -- interesting
# normally I'd say this reflects upward bias in stock prices, but the time scales
# are so small that it doesn't make much sense here
print 'SVM on up data: ' + str(np.mean(up_preds_clf)) + '\n' + 'LR on up data: ' + str(np.mean(up_preds_clf2)) + '\n' + 'SVM on down data: ' + str(np.mean(dn_preds_clf)) + '\n' + 'LR on down data: ' + str(np.mean(dn_preds_clf2))

'''
Modeling: Graphical Data (Raw)
==============================
'''
## Null Accuracy Rates for comparison:
null_rates = list(y.mean())
X = gdata[:-60] # trim to account for max mins ahead target
X = X.join(clean, how='inner')
y = y[5:]       # trim to account for nonexistent starting entries of graphical data

# define 70% cutoff point for train/test split manually
train_inds = int((0.7 * len(X))//1)

#split into train / test sets
X_train = X.iloc[:train_inds, :]
y_train = y.iloc[:train_inds, :]
X_test = X.iloc[train_inds:, :]
y_test = y.iloc[train_inds:, :]


# AdaBoost
for i in range(y.shape[1]):
    from sklearn.ensemble import AdaBoostClassifier
    adb = AdaBoostClassifier(n_estimators = 100)
    adb.fit(X_train, y_train.iloc[:,i])
    ADB_Scores[i] = adb.score(X_test, y_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(ADB_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(ADB_Scores[i]) + \
    '\t' + str(ADB_Scores[i] - null_rates_basic[i])


# Random Forest

# Naive Bayes
NB_Scores = list(np.zeros(y.shape[1]))

for i in range(y.shape[1]):
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, y_train.iloc[:,i])
    NB_Scores[i] = nb.score(X_test, y_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(NB_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(NB_Scores[i]) + \
    '\t' + str(NB_Scores[i] - null_rates_basic[i])

# Logistic Regression
LR_Scores = list(np.zeros(y_basic.shape[1]))

for i in range(y.shape[1]):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train.iloc[:,i])
    LR_Scores[i] = lr.score(X_test, y_test.iloc[:,i])

# compare to null accuracy rates; difference in accuracy:
for i in range(len(LR_Scores)):
    print str(list(y_basic.mean())[i]) + '\t' + str(LR_Scores[i]) + \
    '\t' + str(LR_Scores[i] - null_rates_basic[i])
