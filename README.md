# *Tanzanian Wells: Assessing Water Source Failure With Machine Learning*
**Steven Zych - August 2020**

# Introduction

This project looks at housing prices in King County, Washington--which includes Seattle, Bellevue, Renton, Tacoma, and a handful of smaller cities and countryside. Housing data was sourced from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction), manipulated and cleaned in Python as a Pandas dataframe, and ultimately modelled in an iterative approach using StatsModels and Scikit-learn.

More specifically, this project looks at best choices for first-time buyers in the Seattle area, and aims to answer the question:
> "What factors bring down the price of a house the most, with least impact to overall quality?"

Obviously, this is a fairly subjective question, but categories will be posited later that aim to answer which variables count as having "least impact" on the quality of a home.

All in all, the following packages were used:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Datetime
- Pickle
- Imbalanced-Learn

# EDA and Cleaning 

## Column Names

The first (and perhaps most important) part of cleaning this data was sorting through the columns, identifying their meaning, and determining their usefulness to the project. The full list is as follows:
`['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']`
Columns with `15` at the end refer to the values for the nearest 15 neighbors, waterfront is a yes-no column of whether or not you're on it, and view *claims* to represent whether or not the home has been viewed, yet the values range from 0-4. (Almost every value was 0, and this column was later dropped.)

## Data Types, Duplicates, And Missing Values

The `date` column was changed from strings to datetimes, `sqft_basement` was purged of an erroneous `?` value in many of its rows, and `date` was further manipulated into being represented as `day_of_year` (1-365) so that the model could use it better for predictions (as opposed to interpteting `02-15-2011`, for example).

There were no fully-duplicated rows in the data that needed to be expunged, but there were 117 rows with duplicate ID's--likely homes that were resold. These rows were dropped from the data as they represented only 0.01% of it.

The columns `waterfront` and `yr_renovated` contained thousands of missing values each, both sets of which were replaced with the most likely outcome--0. This indicated "Not on the waterfront" and "Never been renovated" respectively.

With the data cleaned, a baseline model could be built.

# Baseline Model

The first model takes all the data as-is, and uses every non-price column to predict the `price` column. No special modifications made, just plain and simple. The model's R-squared value was **0.692.** For the uninitiated, an R-squared value of **1** would mean that the linear regression model matches onto the predicted value (`price` here) *perfectly.* This is pretty unheard of. The next key metric, RMSE (root mean squared error, or, give-or-take how far off the model's guesses are) is **$213,446.**  For a baseline model however, the R-squared of 0.692 is *not bad,* but we *can* do better. The following models aim to do just that.

![Scatter Plot Of Predicted And Actual Price For Baseline Model](/readme_images/model_base.PNG)

# Model 2: Mean Normalization of Continuous Variables

For this model I first sorted out the continuous variables from the categorical ones. The continuous variables were `['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_basement', 'yr_built', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'day_of_year']`. For each of these variables, a histogram was plotted to observe whether or not the data showed a normal distribution. In this case, only `['price', 'sqft_living15', 'sqft_living']` were deemed "normal enough," and the remaining columns were all mean-normalized (transformed such that their means were now 0, and their minimums and maximums were roughly -1 and 1 respectively).

This model produced a dip in R-squared, down to **0.689,** and I suspected it may have been due to outliers in the data affecting the mean normalizations. Take a look at this plot to further see what I mean. Those lonely dots far down to the left are our price outliers, multi-million-dollar homes whose mere presence in the data throws the model into a tailspin.

![Scatter Plot Of Predicted And Actual Price For Mean Model 2](/readme_images/model_cont_var.PNG)

This impacted the RMSE of the model ever so slightly, bringing it down just a hair to **$213,306.** That's still a pretty massive margin. The subsequent models attempt to address these concerns.

# Model 3: Removing Outliers

It wasn't just `price` that had some extreme outliers. The columns `['price', 'bedrooms', 'sqft_lot', 'sqft_living', 'sqft_basement', 'sqft_lot15']` all had outliers as well (I'm looking at *you,* house with 33 bedrooms). Rows containing an outlier in any of these columns were dropped, and the model was reran with everything else unchanged.

It produced an R-squared value of **0.693** and and RMSE of **$137,906.** This was a considerable improvement on RMSE, and teh R-squared returned to where it was in the baseline.

![Scatter Plot Of Predicted And Actual Price For Model 3](/readme_images/model_remove_outlier.PNG)

# Model 4: Categorical Variables As Dummies

This model was based on the previous model, and involved data manipulation opposite that in model 2. IN other words, I focused on organizing the *categorical variables* in the dataset. Things like `condition`, `grade`, and `waterfront` that were either a binary category or didn't make sense on their own. (A grade of 3 has no meaning when abstracted from whatever rubric it's based on. Is it good? Bad?)

In short, the variables `condition` and `grade` were made into multiple columns of dummy variables (with the first column dropped of course) and `zipcode` was replaced altogether by sorting `lat` and `long` variables into 4 city sectors, where their converging point sits just south of Mercer Island.

This produced a bump in R-squared, up to **0.721** (which can be expected when we're adding so many variables to the model) and an RMSE of **$133,961.** The R-squared was definitely better, and the RMSE hardly changed. This model was kept moving forward.

![Scatter Plot Of Predicted And Actual Price For Model 4](/readme_images/model_cat_vals.PNG)

# Model 5: Selective Predictors

Unsure of a novel change to apply to the data, I modelled it with a twist: I only fed in a variable if its coefficient in the model was at least 6 digits long. In other words, I only modelled that which had a massive impact on price. I though this would strengthen the model, but it didn't at all. R-squared was dropped to **0.675** (worse than the baseline) and RMSE was brought back up some to **$144,649.** Seeing no significant value to this model, it was scrapped. It did, however, produce interesting clumping in the prediction data (as seen in the horizontal clouds below). I did not have time to investigate this behavior.

![Scatter Plot Of Predicted And Actual Price For Model 5](/readme_images/model_selected_predictor.PNG)

# Model 6: Only P-Values Under 0.05

This final model was based on model 4, and was mostly born out of the premise that none of your model's predictors should have a p-value over 0.05. It performed similarly to model 4, with an identical R-squared of **0.721** and a slightly-higher RMSE of **$135,925.** Seeing as this model was effectively the same as the previous best model, and it showed reduced complexity from dropping those low-p-value columns, it was kept, and bestowed the honor of **final model.**

![Scatter Plot Of Predicted And Actual Price For Model 6](/readme_images/model_p_above_005.PNG)

# Recommendations

After iterating through all these models, we return to the initial question:
> "What factors bring down the price of a house the most, with least impact to overall quality?"

I've done the dirty work behind the scenes, and can offer you this summary. The following conditions all have *negative* correlations with price, meaning that if they are met, that house will be a  better deal. Here's what to do:

1. **Buy south of Mercer Island,** which puts you outside of the downtown region, into sectors 3 and 4.
1. **Buy where the neighbors have big yards.** Generally, these homes are in less urban areas (again, not downtown) and consequently can spread out. This is based on `sqft_lot15`.
1. **Buy new within reason.** Interestingly enough, `yr_built` has a negative correlation with price as well. Many older homes may be prized for their more aesthetic architecture or history.
1. **Don't get a basement.** This one surprised me, but basement-having correlates positively with price, so.
1. **Don't live on the waterfront,** but you should know that.

# Future Research

When next I aid first-time buyers in house-hunting in the greater Seattle area, I'd like to devote more time to feature-engineering and/or data-scraping, to get a better assessment of location's effect on price, especially proximity to areas of high cultural significance.

In this research, the only spatial data points used to predict price were latitude and longitude which--while salient--only provide broad strokes of information. Predicting instead by zipcodes and aggregated zipcodes (such that they represent whole neighborhoods, suburbs, and discrete cities) would potentially be a greater predictor of price since similar areas would be grouped with similar areas. In other words, the reputation of "living in Renton" vs. "living in Bellevue," whatever that may mean, would be more present in the data when these spatial groups are recreated in it.

Two final spatial considerations would be proximity to cultural institutions and main streets. Does living next to Pike Place Market drive the price of your house up? What about the Olympic Sculpture park? How about if you live on Pine Street near all the cool restaurants? Intuition says "Yes" to all these, but the data has yet to speak.

# Conclusion

Thank you for lending me your time and consideration, and--more importantly--your trust. It is my sincere hope that the predictions and models set forth in this paper are beneficial to all the first-time buyers reading. Happy house hunting!