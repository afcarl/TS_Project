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

Creating features from the cleaned price data for use in graphical approaches was one of the most difficult parts of the overall analytic process.  The general conceptual pipeline that I built out was to graph rolling time slices of prices in a "clean" format (e.g. no axis labels, tick marks, titles, etc., just the pure price lines over that window) for all four OHLC prices, save the image, then to read back a grayscale version of the saved graph image as a matrix of pixel intensities (which was then unrolled, row by row, to create a vector of pixel features for each observation, or point in time).  

This resulted in a much larger dataset -- graphing each time slice as a 150x200 pixel image, each observation went from having 4 features (OHLC prices) to having 30,000 (one for each pixel of the image).

For feature extraction for image recognition, due to constraints imposed by the library I was using (SimpleCV), I also had to segment the created graphs into distinct directories for binary classes, train and test sets.  From these graphs I was then able to extract features based on length, angle, and position of lines detected in my graphs to build a dataset for predictive modeling.

Target data (used for both the basic data as well as the graphical data) were generated from the basic OHLC price data in a simple binary format.  For a specified number of minutes ahead, a target variable was set to 1 if the closing price at that time was higher than the closing price for the current observation, and 0 otherwise.  This was repeated across a range of forward-looking targets to assess performance of models across different time horizons.

### Modeling and Rationale

Models were divided into two broad categories, described in more detail below: as a baseline, the OHLC price data along with additional features described above were fed into a set of standard and ensemble classifiers.  The graphical data were fed into some of these same classifiers, as well as alternative models for image recognition.  All models were trained on a split of the data that occurred entirely before the test data, so as not to unfairly gain information about the future during the training phase.

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

#### Image Data Modeling

Image data was used for two types of modeling applications: one used features extracted by computer vision software from the graph images created during the data processing phase, and the other used the raw pixel data as features themselves.

**Extracted Features Modeling:**

For image recognition using graphical feature extraction, the SimpleCV library was used to detect lines in the data for 5-minute time slices, and features of those lines were modeled for predicting price movement 5 minutes into the future.  The features extracted and used were average line length, average line angle, and average x- and y-coordinates of the detected lines.  These features were fed into two models, a Support Vector Machine and a Logistic Regression model, and then tested.

**SVM:** The SVM model correctly predicted about 78% of the "up" target data that it was fed after training.  However, it only correctly classified about 22% of the "down" target data.

**Logistic Regression:** The Logistic Regression model correctly predicted about 77% of the "up" target data, and only about 20% of the "down" target data.   I

I am not sure of the reason for this discrepancy between predictive accuracy for "up" cases as opposed to "down" in the context of these models, as they were trained on a combination of both data.  I also find it hard to attribute it to "general upward bias" in the stock market -- on a long time horizon this may indeed be a factor, but for the target data that these models were trained on, only 49% of the targets were "up".

This linear feature data was added to the simple OHLC price data and the models in the prior section were re-run, but adding these features directly to the model did not improve performance.  However, given that both classes of models "work well enough", they could be more usefully ensembled together by a mechanism such as majority vote.

**Raw Pixel Data Modeling:**

For the raw pixel data, AdaBoost, Naive Bayes, and Logistic Regression models were trained on the data.  None of these, however, yielded any useful results -- with 30,000 features, I expect that no single feature was ever significant enough to yield a decent prediction, so even a boosted method like AdaBoost was unable to pick up much signal.  

One class of model that may yield somewhat better results is the Artificial Neural Network, which was not tested on these data for lack of time (and for lack of an ANN model in scikit-learn to try easily) -- however, the nature of the ANN algorithm suggests to me that it may learn usable features from these pixel data in a way that other models tested could not.  This is something that I would like to test in the future, but for the time being, the raw pixel data has been something of a blind alley.

### Challenges and Successes

This project proved challenging for a number of reasons.  Graphical feature creation was rather arduous, and involved more Python programming than data analysis per se (though this was a very useful learning experience).  In addition, processing some of the graphical data was difficult on my old personal computer hardware.  The raw pixel dataset was over 100 million data points, and nearly a gigabyte in memory, so I think it was pushing the limit of what one could reasonably do on a home laptop.  Some of the processing loops took hours to run (though they only needed to be run once to create the graphics and dataframes that I was using), but this hindered the analytic process somewhat.

Using image recognition and extracted feature modeling was also difficult, as it was not a topic covered in class and I had to learn it all very quickly, as I discovered SimpleCV quite late in the process of this project.  Figuring out what types of features to try to extract from the images was difficult, as the documentation was not as robust as that of, say, scikit-learn.  In addition, some types of features that looked as though they might be useful (Haar-type features, for instance), required building one's own Haar cascade from training data, and while I had the "raw material" (processed image files) for that, there was no package that I could find for doing so without great difficulty.

The primary successes that I had with this project were two-fold: one was in learning how to productionize certain parts of my analytic pipeline for easy re-training and re-use of data and models, which helped in quickly obtaining results on my more basic OHLC price data set.  The second success was in the performance of some of the models themselves: the best-performing among them offered 20-35 point gains in predictive accuracy over the null accuracy rate, which is, in my mind, too high to be ignored.  These models were not as tuned as they could have been, either, so I imagine that even higher performance is possible with these.

Finally, I counted it as a major success that in the end, I was able to train CV-based models that did have rather good performance (for at least one predictive use case) -- as this was the original intent of the project, I was glad to have achieved something in that regard, even if the models did not outperform simpler ones based on non-graphical features.

### Looking Forward: Future Expansions and Applications

I believe that this idea of "machine technical analysis" still has promise, as I was only able to scratch the surface of what I had initially envisioned for these types of models.  Additional models, such as Artifical Neural Networks, may provide substantial benefit, as may things like custom Haar cascades based on the image set that I processed.  Ensembles of graphical and price-based models may also outperform what I have created here, and I think there is significant headroom for tuning many aspects of my data production and modeling processes.

There is a clear business application for the results shown in this paper: if not an element of a fully-automated trading scheme, the information provided by the best-performing model could at least be used as a "cognitive augmentation" tool for human traders.  If a model can predict with 85% accuracy which side of a short-term trade it is best to be on, that would seem to be very valuable information that could help any trading operation.  Surely similar econometric models exist to provide this kind of information, but they may not have the flexibility or "data agnosticism" of a machine learning-based technique.  ML techniques such as the one demonstrated here are also very flexible across a variety of use cases: to suit different term trading strategies, models can be re-trained on different time slices of data, and/or different forward-looking targets, with relative ease.  I expect that sooner rather than later, financial firms will catch on to the power that the machine learning tool set has to offer their businesses.

### Conclusion: Key Learnings

I learned a great number of things while pursuing this project.  Many of these fell under the umbrella of general Python programming, how to re-use code effectively, how to effectively incorporate existing libraries into one's own code, how to optimize certain memory- and/or CPU-intensive operations.  I learned enough about computer vision to keep me interested in pursuing its applications to the types of data that I am interested in, and how to quickly build out machine learning model sets in scikit-learn, which will no doubt prove useful in a number of contexts.  

The primary conclusion that I feel I can draw from this project is that, abstracting away details about exact model selection and tuning, feature optimization, etc., machine learning and predictive modeling absolutely have use in quantitative finance.  While I did not particularly suspect the contrary prior to starting this project, building something with my own hands that proves it to me beyond the shadow of a doubt was both satisfying and highly instructive.  There are many possibilities waiting to be uncovered in this space, and through what I have learned over the course of this project, I feel well-equipped to explore them.
