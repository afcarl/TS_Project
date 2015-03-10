## Data Dictionary

Please see below for a description of the features used in this analysis.

DON'T FORGET UNITS.

### Global Variables

- slice_length:     A window of time in minutes.  Used in calculating or generating rolling features.
- mins_ahead:       A number of minutes to look ahead in order to determine target values for classification.  Used in generating target values only.

### Price Variables

- OPEN:     Price of the security (in USD) or index value at the start of the current minute.
- HIGH:     Highest price of the security (in USD) or index value during the current minute.
- LOW:      Lowest price of the security (in USD) or index value during the current minute.
- CLOSE:    Price of the security (in USD) or index value at the end of the current minute.
- VOLUME:   (If used), the # of shares of the security traded during the current minute.

### Derivative Price Variables

- RM_CLOSE:     Rolling mean of the CLOSE price variable over a window of slice_length minutes.
- RSTD_CLOSE:   Rolling standard deviation of the CLOSE price variable over a window of slice_length minutes.
- RMIN_CLOSE:   Rolling minimum value of the CLOSE price variable over a window of slice_length minutes.
- RMAX_CLOSE:   Rolling maximum value of the CLOSE price variable over a window of slice_length minutes.
- HL_SQD:       (HIGH - CLOSE)^2 for each minute interval, as an indicator of volatility.

### Graphical Variables

Each graphical feature is the pixel intensity value (on an inverted scale) as read from the corresponding PNG image of a graph of the trailing slice_length minutes of price data (OHLC).

The pixel values have been unrolled row-wise for each image and concatenated to form a full set of features per observation.

Blank columns (representing margins of the images) have been dropped from the dataset during processing.
