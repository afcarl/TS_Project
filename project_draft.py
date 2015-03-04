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
import cv2

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

                try: 1, 2, 3 to start; maybe 5, 10, 15, 30, 60 will yeild better results

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

'''

# create a dict to reverse the pixel values captured in an image so that grayscale
# actually produces sparsity instead of a bunch of 255 values

rng = np.arange(256)
revd = rng[::-1]
px_dict = dict(zip(rng,revd))

# mapping function to remap pixel values using the above-created dict

def dict_map(item):
    global px_dict
    return px_dict[item]

# main image processing loop:

'''
TODO:

Fix this loop.  Currently it doesn't know when it is hitting the end of a day, and is just wrapping around across days (not desired behavior for graphs, though probably not the worst thing in the world since it is at least "contiguous across trading hours")

Can add a day counter and logic fork (double check this math...)

days = 1 (outside loop)

# if about to exceed end of trading day:
if i % <number of trading mins per day - slice_length*days> == 0:
    # move ahead to start of next day
    i = i + slice_length
    # increment days counter
    days = days + 1
else:
    (do the loop activity already described below)

This would only work if you are starting at the first minute of a day.  Alternatively to this "if" condition, could check the DTI to see what time it is and calc from there; might be better.

===================
Another thing TODO:
===================

Try to generate the index AS the loop is running

graph_dti = pd.DatetimeIndex() (outside loop)

graph_dti.append(clean.ix[i])   # I'm sure this isn't correct, seems too easy

'''
# instantiate a list container for unrolled pixel data
graph_data = []
# instantiate days counter (if using the logic described above)
days = 1
# instantiate DTI for the graph data (to join on clean)
graph_dti = pd.DatetimeIndex()

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
Things to try to improve this loop:

# it is quite slow - one thing to look into, can I force it to store as sparse by
# changing some values of the grayscale?  Could shift to binary (0 for 255, 1 for
# everything else), or just "reverse" the scale (with a dict?) so that 255 maps to
# 0 and things map to their reflection in the set (have now done this above)

# another thing to try: tweak dpi kwarg in plt.savefig() (have now tried this too)
'''

# pickling so that I don't have to run this again:

gdata = DataFrame(graph_data)
gdata.to_pickle('data/gdata_pickle')
# reading pickle appears to be very slow for some reason
gdata = pd.read_pickle('data/gdata_pickle') # moved out of git repo; check dir

'''
Index and join generated image data to clean financial data
===========================================================

First need to generate a DatetimeIndex that matches the correct pixel rows
(first row of gdata = slice_length'th row of clean data)
so the DTI for gdata will be slice_length shorter than the one for clean data

After getting correct DTI in place, inner join the two DFs on the index
'''

# this isn't working for some reason:
# gdata_indexed = gdata.set_index(list(list(clean.index[:-5])))

# alternative: add a new col to gdata of values from clean.index; join on col to index
# also doesn't appear to be working right now; I'm assuming memory issues
fake_idx = Series(list(clean.index[:-5]))
test = pd.concat([gdata, fake_idx], axis = 1)

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
- MAKE SURE TO CHECK AGAINST NULL ACCURACY RATE
- compare some cross-cutting metric for different classifiers (AUC, F-1 score, etc.)
'''


'''
Classifier instances to try, w/rationale:
=========================================

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

'''
# this will be the part where I instantiate a few models in sklearn
# and then train them on the EARLIER 70% (say) of the dataset

# cannot use regular train_test_split because then the algos would be "cheating"
# by having some future data in the training mix, even if it didn't know it was
# future data

# AdaBoost

# Random Forest

# Naive Bayes

# Logistic Regression

## COMPARISON TO EACH OTHER

'''
TODO:

Build X_basic dataframe of more basic data to train on.  Include things like:

- rolling average
- rolling std (volatility metric)
- prior window max / min

Redo classification models to see if the simpler data performs better/worse/no different
'''

## Null Accuracy Rates for comparison:

null_rates = y.mean().to_dict()

'''
Plan B
======

Alternative, more basic dataset (not using graphically-generated parameters), that can be used either in lieu of or as a basis for comparison to the more complex data.

'''
# generate a few additional basic features from clean data

# rolling mean of close
RM_CLOSE = pd.rolling_mean(clean.CLOSE, slice_length)

# rolling standard deviation of close
RSTD_CLOSE = pd.rolling_std(clean.CLOSE, slice_length)

# rolling min of close
RMIN_CLOSE = pd.rolling_min(clean.CLOSE, slice_length)

# rolling max of close
RMAX_CLOSE = pd.rolling_max(clean.CLOSE, slice_length)

# (high - low)^2 for each observation (vol measure of prices quoted within that minute)
HL_SQD = Series([(clean.iloc[row][1] - clean.iloc[row][2]) ** 2 for row in range(len(clean))])

# inner join of the above-created Series on the "clean" DF
X_basic = clean.join([RM_CLOSE, RSTD_CLOSE, RMIN_CLOSE, RMAX_CLOSE, HL_SQD], how=inner)

# AdaBoost on basic data

# Random Forest on basic data

# Naive Bayes on basic data

# Logistic Regression on basic data

## COMPARISON NOTES VS COMPLEX DATA

'''
Notes for model improvement:

- Try centering graphical data in a consistent way (not sure how default graph behavior works... can test this out a bit with some spot examples) -- if it is being generated "consistently" by matplotlib in some fashion, this is a lower-order concern, but still worth looking into
- Try to ensure consistent scale in graphical data (I don't think this is correct right now) -- see note above on this as well
- Use volatility data
- On graphical data, remove any empty columns (may exist due to image margins)
- Try different tuning parameters for slice_length
- think more about prior time data -- weighting factor for historical data, inclusion of prior preds as features in current observation (possibly also weighted)
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


'''
