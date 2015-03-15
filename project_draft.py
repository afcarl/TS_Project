'''
Imports:
========

'''

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import gc
import shutil
import cv2
import SimpleCV

'''
Declare some global variables:
==============================

slice_length:   a time slice length in minutes to be used to define the primary
                features of each observation

                try: 5, 10, 15 to start

prior_slices:   how many slices prior to the current slice to get summary stats
                for, to be included with the CURRENT slice as additional features
                (not implemented below yet)

                try: 1, 3, 5, 7 to start

discount_wgt:   a factor by which to downweight prior_slices data

                try: 0.5, 0.3, 0.1 to start
                also probably read some literature on this to understand why /
                how to do it properly
                (also not implemented below yet)

mins_ahead:     how many minutes ahead to set known targets for each observation
                to train against (algos may do better at certain times ahead)

                try: 1, 2, 3 to start; maybe 5, 10, 15, 30, 60 will yield better results

'''

slice_length = 5
# prior_slices = 3      # not currently used
# discount_wgt = 0.5    # not currently used
mins_ahead = [1, 2, 3, 5, 10, 15, 30, 60]

'''
Fetching and importing raw data:
================================

Max trailing days of tick data is 20 from Google Finance -- started collecting
full sets from January 8, 2015; can add forward from there and re-run generic
data cleaning pipeline and analysis.

Data are being captured for SPX (could try some others but this seems to be a
practical choice for a proof-of-concept like this... except for lack of volume data),
and are stored in /rawdata as .txt files.

Minute-tick data are being obtained from Google Finance for given tickers for a specified historical range using the following URL format to scrape for data:

http://www.google.com/finance/getprices?i=[PERIOD]&p=[DAYS]d&f=d,o,h,l,c,v&df=cpct&q=[TICKER]

* [PERIOD]: Interval/frequency in seconds (60 is the most granular that the data are available for)
* [DAYS]: Integer number of days back in time to fetch data for
* [TICKER]: A valid ticker symbol that you could search for on Google Finance

Saved data filename scheme is:
[Ticker]_[sequence number, starting at 001]_[Mon]_[Day]_[Year].txt
where the date is the end date for that file (first file is mislabeled).

'''

# column format is constant for data fetched from this Google Finance API
cols = ['DATE', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
# only have one file for right now so not worrying about splicing things together yet
# assuming we are in the directory where this file is located, w/rawdata subdir
raw_data = pd.read_table('rawdata/SPX_001_Jan_8_2015.txt', sep=',', header=6, names=cols)

# now have multiple files; need to read in both -- probably best to do separately, and join on datetimeindex

'''
Cleaning the raw data:
======================
Tasks:
        - Figure out which of the columns we want to do stuff with
            - At least one price, volume (probably), some range calculation based on H/L
        - Define graphical parameters (based on summary statistics of variance of slice_length windows over the dataset for max values)
        - Convert the slice_length windows into sparse matrices of data based on graphical parameters
        - Create a new dataframe with the following features:
            - prior_slices summary stats, discounted by discount_wgt
            - sparse matrix data for current slice
            - targets (one for each mins_ahead that can fit into current days' data)
'''

# ugly, presumably redundant function to convert Unix Epoch stamps:
def convert_ts(stamp):
    stamp = int(stamp) # making sure
    return pd.datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stamp)), '%Y-%m-%d %H:%M:%S')

# could also use some refactoring but this works for now:

converted = []

# loop to convert the raw data time column to a proper human-readable timestamp:

for i in range(len(raw_data)):
    if raw_data['DATE'][i][0] == 'a':
        converted.append(convert_ts(raw_data['DATE'][i][1:len(raw_data['DATE'][i])]))
    else:
        converted.append(converted[(i-1)] + datetime.timedelta(minutes=1))

# make into DatetimeIndex for use in dataframe:

dti = pd.DatetimeIndex(converted)
clean_cols = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
clean = DataFrame(raw_data.as_matrix(columns=clean_cols), index=dti, columns=clean_cols)

'''
Basic loop to graph slices of the time series
=============================================

This loop takes about 4 hours to run on my 5-year-old laptop; pickle of the DF
from the output is available here (1.x GB):

https://www.dropbox.com/s/2m5pp7gtev36jrd/gdata_pickle?dl=0

[update notes here: describe the two individual loops for graphing / CV]

'''

# splitting the loop in two -- graph-making half (try to fine-tune this a bit):
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

## Redo cv2 loop below.  Be VERY CAREFUL to align data properly... don't be off by one.

# create a dict to reverse the pixel values captured in an image so that grayscale
# actually produces sparsity instead of a bunch of 255 values
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


# image processing loop -- cv2 half:
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

# run this once (if it is working properly) and simply capture which cols are empty, then drop
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
TODO:

- Drop the invalid rows (based on time of day) from gdata
- Rows for first 5 mins of every day beyond the first are invalid due to cross-day wrapping

This isn't worth the computational effort; delete invalid images and re-generate the gdata DF instead
(or ignore this problem for now)
'''

invalid_times = ['09:31:00', '09:32:00', '09:33:00', '09:34:00']

for i in range(len(gdata)):
    if str(gdata.index[i])[-8:] in invalid_times:
        print "Dropping row at index " + str(gdata.index[i]) + ' at ' + time.ctime()
        gdata.drop(gdata.index[i], inplace=True)

'''
Index and join generated image data to clean financial data
===========================================================

After getting correct DTI in place, inner join the two DFs on the index
'''

test = clean.join(gdata, how='inner')

'''
Generate target data for model training
=======================================

NB: targets being generated from forward data means we will lose a few
train / test examples on the near-term end of the time series
'''

# stupidly simple binary loop; flexible to whatever is specified in mins_ahead:
ahead = []

for i in range(len(clean) - max(mins_ahead)):
    current_row = [1 if clean.iloc[i+mins_ahead[j],0] > clean.iloc[i,0] else 0 for j in range(len(mins_ahead))]
    ahead.append(current_row)

y = DataFrame(ahead, columns = [str(mins_ahead[i]) + '_ahead' for i in range(len(mins_ahead))])

'''
TODO:
- think about something cleverer than a binary target (maybe 4-5 classes?)
- try to train an actual classifier (logreg, rf, nb, adaboost)
- compare some cross-cutting metric for different classifiers (AUC, F-1 score, etc.) [does sklearn "Score" do this automatically?]
'''


###################################
# Image Recognition with SimpleCV #
###################################


# copy 70% of everything with a 1 in ahead_5 target to supervised/up5, 0 to supervised/down5
# copy 30% of everything with a 1 in ahead_5 target to unsupervised/up5, 0 to unsupervised/down5

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

# images are now ready for SimpleCV training and testing
# this blobs example does nothing useful; wrong sort of thing for our graphs
# try findLines() or findEdges()

up5_imgs = SimpleCV.ImageSet('../data/supervised/up5')
#up5_blobs = [x.findBlobs()[0] for x in up5_imgs]
up5_lines = [x.findLines() for x in up5_imgs]
dn5_imgs = SimpleCV.ImageSet('../data/supervised/down5')
#dn5_blobs = [x.findBlobs()[0] for x in dn5_imgs]
dn5_lines = [x.findLines() for x in dn5_imgs]
temp_data = []
temp_targets = []
target_names = ["Down", "Up"]

# for x in up5_blobs:
#     temp_data.append([x.area(), x.height(), x.width()])
#     temp_targets.append(1)

for x in up5_lines:
    coord1mean = np.mean([obs[0] for obs in x.coordinates()])
    coord2mean = np.mean([obs[1] for obs in x.coordinates()])
    temp_data.append([np.mean(x.length()), np.mean(x.angle()), coord1mean, coord2mean])
    temp_targets.append(1)

# for x in dn5_blobs:
#     temp_data.append([x.area(), x.height(), x.width()])
#     temp_targets.append(0)

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
#up5_test_blobs = [x.findBlobs()[0] for x in up5_test]
up5_test_lines = [x.findLines() for x in up5_test]
dn5_test = SimpleCV.ImageSet('../data/unsupervised/down5')
#dn5_test_blobs = [x.invert().findBlobs()[0] for x in dn5_test]
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

to_predict_dn = DataFrame(to_predict_dn, columns = ['mean_len', 'mean_angle', \
    'c1mean', 'c2mean']).dropna()

dn_preds_clf = clf.predict(to_predict_dn)
dn_preds_clf2 = clf2.predict(to_predict_dn)

# good at predicting on up, not so good at predicting on down -- interesting
# normally I'd say this reflects upward bias in stock prices, but the time scales
# are so small that it doesn't make much sense here
print 'SVM on up data: ' + str(np.mean(up_preds_clf)) + '\n' + 'LR on up data: ' + str(np.mean(up_preds_clf2)) + '\n' + 'SVM on down data: ' + str(np.mean(dn_preds_clf)) + '\n' + 'LR on down data: ' + str(np.mean(dn_preds_clf2))


'''
Classifier instances to try, w/rationale:
=========================================

Processing to try:
- Lasso
- Ridge regression

AdaBoost
========
Seems like the pixel data is a good candidate for weak learning, since it is generated via CV and deliberately includes multiple graphed lines per image (hard to determine a priori which predictors might be best to ignore / best to boost)

Random Forest
=============
Again, due to large size of the data and non-intuitive nature of the feature columns extracted from the pixel data (pixel data has meaning as a whole, but individual cols are not straightforward), seems like an obvious choice for classification

Naive Bayes
===========
Partially interested in this as a control -- the data are definitely not independent, so the performance SHOULD be worse here I think.  If independence assumption violation doesn't matter, Naive Bayes may actually work well, since the generated pixel features are somewhat arbitrary and in some sense lend themselves well to the prior/posterior belief paradigm

Logistic Regression
===================
Another check to make sure we're not getting too fancy for no useful reason

SVM
===
Alternative classification approach for basic data; unlikely to work for graphical data due to having more features than observations

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


## COMPARISON TO EACH OTHER

'''
Plan B
======

Alternative, more basic dataset (not using graphically-generated parameters), that can be used either in lieu of or as a basis of comparison for the more complex data.

'''
# generate a few additional basic features from clean data:
# rolling mean, std, min, max of close, and (high-low) ^2 for each observation
RM_CLOSE = pd.rolling_mean(clean.CLOSE, slice_length)
RSTD_CLOSE = pd.rolling_std(clean.CLOSE, slice_length)
RMIN_CLOSE = pd.rolling_min(clean.CLOSE, slice_length)
RMAX_CLOSE = pd.rolling_max(clean.CLOSE, slice_length)
HL_SQD = Series([(clean.iloc[row][1] - clean.iloc[row][2]) ** 2 for row in range(len(clean))], index=clean.index)

## Add others if I think of something clever
# binary variables based on some T/F condition vs. a lookback, maybe ?

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

# create Train and Test sets MANUALLY
# cannot use randomization, else cheating by "looking into the future"
# however, CAN use pd.rolling_apply or similar to create "rolling" CV folds -- look into this
# see if it is possible to use Pipeline for this... not sure

# define 70% cutoff point manually
train_inds = int((0.7 * len(X_basic))//1)

#split into train / test sets
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
    print str(list(y_basic.mean())[i]) + '\t' + str(ADB_Scores[i]) + \
    '\t' + str(ADB_Scores[i] - null_rates_basic[i])

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
    print str(list(y_basic.mean())[i]) + '\t' + str(RF_Scores[i]) + \
    '\t' + str(RF_Scores[i] - null_rates_basic[i])

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
    print str(list(y_basic.mean())[i]) + '\t' + str(NB_Scores[i]) + \
    '\t' + str(NB_Scores[i] - null_rates_basic[i])

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
    print str(list(y_basic.mean())[i]) + '\t' + str(LR_Scores[i]) + \
    '\t' + str(LR_Scores[i] - null_rates_basic[i])

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
    print str(list(y_basic.mean())[i]) + '\t' + str(SVM_Scores[i]) + \
    '\t' + str(SVM_Scores[i] - null_rates_basic[i])


## COMPARISON NOTES VS COMPLEX DATA

'''
Notes for model improvement:

- Try centering graphical data in a consistent way (not sure how default graph behavior works... can test this out a bit with some spot examples) -- if it is being generated "consistently" by matplotlib in some fashion, this is a lower-order concern, but still worth looking into
- Try to ensure consistent scale in graphical data (I don't think this is correct right now) -- see note above on this as well
- Use volatility data
- On graphical data, remove any empty columns (may exist due to image margins)
- Try different tuning parameters for slice_length
- think more about prior time data -- weighting factor for historical data, inclusion of prior preds as features in current observation (possibly also weighted)
- think about using GridSearchCV to tune parameters (and double-check that tuning was worthwhile using a non-cheating method) -- could use these as "optimized defaults" to be periodically reevaluated in the future
'''





'''
Various cruft from earlier mucking about:
=========================================

Keeping these around for now, just in case...

dl_datestamp = time.strftime("%c")              # to datestamp downloads

raw_data.to_pickle('data/df_pickle')            # to maintain a static copy of
raw_data = pd.read_pickle('data/df_pickle')     # raw data as feeds change

# code used to create quick graph for 1/28 progress report, basic exploration:

raw_data[['CLOSE', 'HIGH', 'LOW', 'OPEN']][10:30].plot()
plt.savefig('example_plot.png')

# memmap code; not sure if sklearn can read memmaps the way it can read ndarrays/dfs though

graph_data = np.memmap('graph_data', dtype='float64', mode='w+', \
     shape=(len(clean) - slice_length, 30000))  # for 150x200 px image

# trying to concat Series rather than lists; this now seems pointless:

pd.concat([Series(unrolled), graph_data], axis=1)   # maybe not this one

# main image processing loop: (ORIGINAL VERSION - SPLITTING THIS UP)
for i in range(len(raw_data) - slice_length):
   # plot the length of the time slice
       # various matplotlib code goes here -- NB no need to actually plot the figure onscreen
       # below is example code -- need better params to make it as clean a visual as possible
       clean[['CLOSE', 'HIGH', 'LOW', 'OPEN']][i:i+slice_length].plot(legend=False)

   # save the graph image to disk
       plt.axis('off')
       plt.savefig('graphics/obs_graph.png', dpi=25) # transparency doesn't seem to help w/grayscale values
       plt.close('all')

   # load in pixel data using cv2
       gs_img_data = cv2.imread('graphics/obs_graph.png', cv2.IMREAD_GRAYSCALE)

   # access pixel values and unroll
       unrolled = []
       for row in range(gs_img_data.shape[0]):
           unrolled = unrolled + map(dict_map, list(gs_img_data[row,:]))

   # append the unrolled row of pixel observations to the graph_data object (TBD... DataFrame, memmap, list?)
       graph_data.append(unrolled)

'''
