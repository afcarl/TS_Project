## Machine Technical Analysis
#### Data Science, General Assembly
#### Alex Lee, 3/13/2015

**Note to readers:** companion code for this paper is available in the GitHub repository where it is hosted.

### Problem and Hypothesis

In finance, so-called "technical analysis" attempts to divine insights about future movements in the price of a security by examining the visual characteristics of recent price movements and attempting to identify "telling" patterns.  While many (including your author) believe technical analysis as it is performed by humans to be essentially bunk, the application of machine learning to the same problem poses an interesting question: can algorithms for processing an abstraction of visual price data actually perform well in predicting future price movements?

While technical analysis relies on pattern recognition at base, humans are subject to biases in seeing patterns where the pattern is either dubious or very possibly meaningless in the context that the human is trying to evaluate.  Algorithms are a lot harder to fool in this sense, as they do not know that they are looking at price data or visualizations of price movements -- where a human analyst or trader might have a gut feeling about where prices were moving on a certain trading day and be subject to subsequent bias in evaluating data ("... well it LOOKS like it's about to trend up again..."), an algorithm will not.

This paper charts the progress of attempts to answer this question, including methodology, modeling, and comparison of results across base cases as well as models that were not based on *visual* pattern analysis for evaluating the same problem.  

### Data Sourcing and Description

Data for this project were obtained from a Google Finance web-based API that provides up to 20 days' worth of historical price data at the minute-tick level, for any ticker that Google Finance has data for.  Data are given for the open, high, low, and close price for each tick (representing the respective prices during that 1-minute interval, in the case of the data I used); data are also available for volume for certain tickers (not indices such as SPX which I used for my modeling, though).

**Note:** More detailed descriptions of the individual data features are available in the data_dictionary.md file located in the same GitHub repository as this document.

### Data Processing and Exploration

As the data obtained from Google Finance were directly from the API, they were very clean.  The primary pre-processing required was to convert the initially cryptic Unix-timestamp-keyed index to human-readable times for proper alignment of target data as well as features created from the data.  This was also necessary to allow for efficient merging of data pulled from the API in the future to create contiguous datasets.

Data exploration was relatively simple for my purposes, as the primary "feature" of interest that I was looking for was suitable variability in the visual graphs of all 4 price movements to present an algorithm with meaningfully discernible data.  For time slices of as little as 5-10 minutes, this appeared to be the cased based on exploratory visual analysis.

### Feature Creation

this was the most arduous part - describe final implementation of 2 loops (graphing, image processing)

### Modeling and Rationale

This section is divided into two sections etc. etc.

#### Basic Modeling

Pretty good results for some of these actually

#### CV Modeling

TBD based on ANN / PyBrain modeling.  Non-ANN models for processing pixel data performed poorly, even models like NB which I expected would perform better.

### Challenges and Successes

- Feature creation was a huge pain
- Forgetting about ANN
- Trying to implement ANN

### Looking Forward: Future Expansions and Applications

Definitely still think this has promise; productionizing certain parts of it would be interesting and worthwhile (webscraping, functions to automate some of the preprocessing, batching out more robust graphing jobs, etc.)

### Conclusion: Key Learnings

I need a better computer.  Also real conclusions go here.
