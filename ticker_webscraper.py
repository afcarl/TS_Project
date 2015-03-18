# imports
import time
import urllib
import os

# manually define tickers of interest
tickers = ['AAPL', 'DGS', 'DTN', 'EWJ', 'FNCL', 'FTEC', 'IBB', 'IVV', 'IXJ', 'JXI', 'ONEQ', 'SPX', 'QQQ', 'VEU', 'VGT', 'VTI']

# define function to pull arbitrary ticker data:
def get_ticker(ticker):
    # create url string for Google Finance API:
    turl = 'http://www.google.com/finance/getprices?i=60&p=20d&f=d,o,h,l,c,v&df=cpct&q=' + str(ticker)
    # set destination dir; create if doesn't exist
    dest = 'rawdata/' + str(ticker) + '/'
    if not os.path.exists(dest):
        os.makedirs(dest)
    # set file name
    now = str(time.ctime())
    fname = str(ticker) + '_' + now[4:7] + '_' + now[8:10] + '_' + now[-4:] + '.txt'
    # request data & save to destfile; overwriting is not an issue as we always want latest
    urllib.urlretrieve(turl, dest + fname)
    # notify
    print 'Fetched ' + ticker + ' data to file: ' + dest + ' at ' + now[11:19]


# run loop periodically to fetch ticker data:
for ticker in tickers:
    get_ticker(ticker)
    time.sleep(2)
