## Machine Technical Analysis
#### Data Science, General Assembly
**Alex Lee
3/13/2015**

**Note to readers:** companion code for this paper is available in the GitHub repository where it is hosted.

### Problem and Hypothesis

In finance, so-called "technical analysis" attempts to divine insights about future movements in the price of a security by examining the visual characteristics of recent price movements and attempting to identify "telling" patterns.  While many (including your author) believe technical analysis as it is performed by humans to be essentially bunk, the application of machine learning to the same problem poses an interesting question: can algorithms for processing an abstraction of visual price data actually perform well in predicting future price movements?

While technical analysis relies on pattern recognition at base, humans are subject to biases in seeing patterns where the pattern is either dubious or very possibly meaningless in the context that the human is trying to evaluate.  Algorithms are a lot harder to fool in this sense, as they do not know that they are looking at price data or visualizations of price movements -- where a human analyst or trader might have a gut feeling about where prices were moving on a certain trading day and be subject to subsequent bias in evaluating data ("... well it LOOKS like it's about to trend up again..."), an algorithm will not.

This paper charts the progress of attempts to answer this question, including methodology, modeling, and comparison of results across base cases as well as models that were not based on *visual* pattern analysis for evaluating the same problem.  

### Data Sourcing and Description

Data for this project were obtained from a Google Finance web-based API that provides up to 20 days' worth of historical price data at the minute-tick level, for any ticker that Google Finance has data for.  Data are given for the open, high, low, and closing (OHLC) prices for each tick (representing the respective prices during that 1-minute interval, in the case of the data I used); data are also available for volume for certain tickers (though not for indices such as SPX which I used for my modeling).

I chose to use SPX data as a proof-of-concept set, as SPX indexes overall market performance.  The same techniques used to model SPX could be used for any other security for which one could pull similar data.

**Note:** More detailed descriptions of the individual data features are available in the data_dictionary.md file located in the same GitHub repository as this document.

### Data Processing and Exploration

As the data obtained from Google Finance were directly from the API, they were very clean.  The primary pre-processing required was to convert the initially cryptic Unix-timestamp-keyed index to human-readable times for proper alignment of target data as well as features created from the data.  This was also necessary to allow for efficient merging of data pulled from the API in the future to create contiguous datasets.

Data exploration was relatively simple for my purposes, as the primary "feature" of interest that I was looking for was suitable variability in the visual graphs of all 4 price movements to present an algorithm with meaningfully discernible data.  For time slices of as little as 5-10 minutes, this appeared to be the cased based on exploratory visual analysis.

### Feature Creation

From the basic OHLC price data, a number of additional features were calculated for use in a non-graphical "base case" model.  These included rolling mean, standard deviation, minimum, and maximum across a window which was defined to be the same as the time slice length used to generate the graphical features.  In addition to these rolling features, I calculated the squared difference between the high and the low prices for each observation, to capture some measure of relative volatility in the price data over time.

Creating features from the cleaned price data for use in graphical approaches was one of the most difficult parts of the overall analytic process.  The general conceptual pipeline that I built out was to graph rolling time slices of prices in a "clean" format (e.g. no axis labels, tick marks, titles, etc., just the pure price lines over that window) for all four OHLC prices, then to read back a grayscale version of the saved graph image as a matrix of pixel intensities (which was then unrolled, row by row, to create a vector of pixel features for each observation, or point in time).  

This resulted in a much larger dataset -- graphing each time slice as a 150x200 pixel image, each observation went from having 4 features (OHLC prices) to having 30,000 (one for each pixel of the image).

Target data (used for both the basic data as well as the graphical data) were generated from the basic OHLC price data in a simple binary format.  For a specified number of minutes ahead, a target variable was set to 1 if the closing price at that time was higher than the closing price for the current observation, and 0 otherwise.  This was repeated across a range of forward-looking targets to assess performance of models across different time horizons.

### Modeling and Rationale

Models were divided into two broad categories, described in more detail below: as a baseline, the OHLC price data along with additional features described above were fed into a set of standard and ensemble classifiers.  The graphical data were fed into some of these same classifiers, as well as alternative models.  All models were trained on a split of the data that occurred entirely before the test data, so as not to unfairly gain information about the future during the training phase.

#### Classification Modeling on Simple OHLC Price Data

The OHLC price data and derivative features were modeled using a variety of classifiers, to try to optimize for performance as well as to verify assumptions.  Individual models will be addressed in turn.

**AdaBoost:** As a general purpose ensemble classifier, I expected this, along with Random Forests, to perform relatively well.  Representative scores for a model trained on 5-minute time slices are below:

- **1 minute ahead:** .60 vs. null accuracy rate of .49, for an 11-point boost
- **2 minutes ahead:** .59 vs. null accuracy rate of .49, for a 10-point boost
- **3 minutes ahead:** .56 vs. null accuracy rate of .49, for a 10-point boost

Scores fell off for further lookahead periods after that, except for 15 minutes ahead, which had about an 8-point boost over a null accuracy rate of .50.

**Random Forest:** Like AdaBoost, I expected this robust classifier to perform relatively well.  Some representative scores for a model trained on 5-minute time slices:

- **1 minute ahead:** .72 vs. null accuracy rate of .49, for a 23-point boost
- **2 minutes ahead:** .67 vs. null accuracy rate of .49, for an 18-point boost
- **3 minutes ahead:** .69 vs. null accuracy rate of .49, for a 20-point boost
- **5 minutes ahead:** .64 vs. null accuracy rate of .50, for a 14-point boost
- **10 minutes ahead:** .61 vs. null accuracy rate of .51, for a 10-point boost
- **15 minutes ahead:** .58 vs. null accuracy rate of .50, for an 8-point boost

As expected, Random Forest modeling performed very well indeed, even when looking further into the future than the time slice length it was trained on.  Interestingly, scores for 30 and 60 minutes ahead converged almost completely to base accuracy, whereas for AdaBoost they did not.  For the applications that I can think of for these models, though, Random Forest appears preferable to AdaBoost.

**Support Vector Machine:** One of two main non-ensemble classifiers that I tested.  As the simple OHLC price data were not very high dimensional, I expected non-ensemble classifiers (SVM and Logistic Regression) to still perform well on the data.  Some representative scores for a model trained on 5-minute time slices:

- **1 minute ahead:** .81 vs. null accuracy rate of .49, for a 32-point boost
- **2 minutes ahead:** .76 vs. null accuracy rate of .49, for a 27-point boost
- **3 minutes ahead:** .71 vs. null accuracy rate of .49, for a 22-point boost
- **5 minutes ahead:** .68 vs. null accuracy rate of .50, for an 18-point boost
- **10 minutes ahead:** .61 vs. null accuracy rate of .51, for a 10-point boost
- **15 minutes ahead:** .59 vs. null accuracy rate of .50, for a 9-point boost

For nearly every lookahead period, SVMs outperformed Random Forests, occasionally by a large margin (1 and 2 minutes ahead).  

**Logistic Regression:** Another non-ensemble classifier; some representative scores for a model trained on 5-minute time slices:

- **1 minute ahead:** .84 vs. null accuracy rate of .49, for a 35-point boost
- **2 minutes ahead:** .80 vs. null accuracy rate of .49, for a 31-point boost
- **3 minutes ahead:** .76 vs. null accuracy rate of .49, for a 27-point boost
- **5 minutes ahead:** .72 vs. null accuracy rate of .50, for a 22-point boost
- **10 minutes ahead:** .65 vs. null accuracy rate of .51, for a 14-point boost
- **15 minutes ahead:** .65 vs. null accuracy rate of .50, for a 15-point boost
- **30 minutes ahead:** .61 vs. null accuracy rate of .52, for a 9-point boost

Logistic regression outperformed SVMs for every lookahead period that I modeled.  I was slightly surprised by this performance, but very happy with the results.

Overall, the model performance falling off past the short-lookahead marks makes intuitive sense; the models were trained on only 5 minutes' worth of data per observation, so predictive power might be expected to fall off quickly for ranges beyond that.  To confirm this intuition, a couple of the models we re-run on 10-minute time slices, and scores for predictions past 3 minutes ahead did indeed rise.  There is much more that could be tuned with respect to relative lengths of time slices and lookahead periods, though such tuning would be highly dependent on the specifics of the use case for the predictions.

In addition, the relative under-performance of the ensemble models may be explained by the general uniformity of the dataset, as well as possible greater need to finely tune those models.  All data points for the modeling described above were ultimately related to just one variable: price.  As such, it perhaps should not be surprising that a logistic regression model gave the overall best performance.

#### CV Modeling

TBD based on ANN / PyBrain modeling.  Non-ANN models for processing pixel data performed poorly, even models like NB which I expected would perform better.

### Challenges and Successes

- Feature creation was a huge pain
- Forgetting about ANN
- Trying to implement ANN
- Using image recognition libraries

### Looking Forward: Future Expansions and Applications

Definitely still think this has promise; productionizing certain parts of it would be interesting and worthwhile (webscraping, functions to automate some of the preprocessing, batching out more robust graphing jobs, etc.)

### Conclusion: Key Learnings

I need a better computer.  This was good practice learning to integrate lots of libraries into analysis.  Also real conclusions go here.
