# *Tanzanian Wells: Assessing Water Source Failure With Machine Learning*
**Steven Zych - August 2020**

# Introduction

This project looks at the condition of water wells in the East-African country of Tanzania, with the aim of building a machine learning model that predicts the condition of any given well. Predictions are made from a set of independent variables such as `funder`, `yr_built`, `region`, and so on. All of the data is available for free at [this competition link](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/).

The models in this project were based exclusively on the files named `train_labels.csv` and `train_vals.csv`, though a third `target_vals.csv` is also available for competition entries. The data was manually cleaned in the notebook `cleaning.ipynb` and made into a Pandas DataFrame. This DataFrame (boasting 59,000 rows and 31 features) was then brought to the `modeling.ipynb` notebook where four main ML algorithms were used. A general overview of the cleaning process and all ML models are provided in the following sections.

To be explicit, this is a **ternary classification problem,** where the three possible classes are **functional, functional-needs-repairs, and nonfunctional.** The aim of this project is to make a model that accurately applies these three labels, and later to investigate **what features have the greatest effect on predictions.** In doing so, the Tanzanian government (as well as independent aid organizations) can be aided in distributing resources and help to communities in need of clean, accessible water.

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

## General Cleaning

The data cleaning on this project was fairly straightforward. Nothing too odd or frustrating was present in the data, aside from the oft-expectable case of 0's being used in an unclear manner (Are they 0's? Are they placeholders?). A handful of columns that were almost entirely NaN values or 0's were dropped, including `num_private`, which was 99% 0-values, and `scheme_name`, which was over 50% NaN.

Beyond this, the brunt of the data cleaning involved sorting through the numerous columns in this dataset with `_class` and `_group` suffixes to see where there were redundancies. Count plots were made for the members of any set (such as `extraction`, `extraction_type`, etc.) to see which member of that set should be kept, and the redundant features were dropped. A feature that produced examples such as the plot below (high variance in class distribution across values, note orange bar in "other") were kept.

![Extraction Count Plot](/images/extraction_bar.PNG)

This thinned the herd down to 31 features. These features were, unfortunately, largely categorical and had to be made into dummy variables. Value counts were printed for all categorical columns and inspected. Dummy variables were made in such a way as to minimize the amount needed. If, for example, the most common value was at 80% frequency and the next was only at 10% frequency, there would only be a dummy variable made for that most-common value only. From here, the data was ready to be modelled on. But before we talk about that, I'd like to share a few EDA insights with the following images.

## EDA Insights

The count plot below shows the relationship between a well's source method and its functionality. We see that rivers and natural springs tend to perform quite well, whereas machine-drilled boreholes and shallow wells do quite poorly--along with lakes. This seems to suggest that more natural methods of water extraction are more successful in the long run.

![Water Source Count Plot](/images/source_bar.PNG)

This next count plot shows the same sort of relationships but in regard to the quality of water. It is in a way unsurprising that water described as "salty," "milky," or "unknown" could correlate with a misfunctioning pump. There is a serious imbalance problem here in that "good" water accounts for most instances, but it is still worth noting.

![Water Quality Count Plot](/images/water_bar.PNG)

Lastly--and this is my favorite image in this document--here is a basic scatter plot of wells in Tanzania (with their colors signifying quality--green good, blue okay, red bad) superimposed on a Google map of the country. It is obvious that, while not extremely discrete, there *are* geographic impacts on well functionality.

![Scatter Plot Of Well Function Over Tanzania](/images/tanzanian_wells.jpg)

# Machine Learning Models

## Baseline Model

Due to the presence of some geographic clustering in the data, a KNearestNeighbors classifier was used to set a baseline. No hyperparameters were tweaked here, and an unimpressive **accuracy of 52%** was produced. Also, the model labeled nonfunctional pumps as functional twice as often as it labelled them correctly (true label 2, predicted as 0 below). Since this is the **most important class** to get right, this is unacceptable.

![Baseline Model Confusion Matrix](/images/base_matrix.PNG)

From here, the four following models were trained in the same manner:
1. A Pipeline case was instantiated with a StandardScaler and a default-version of the given classifier.
1. A parameter grid was made for this model.
1. A GridSearchCV was made for the model and run.
1. The best parameters found by the GridSearchCV instance were used to fit, train, and score the current model.

## K-Nearest Neighbors

The above methodology was applied to a KNN classifier. The only parameter tested was number of neighbors. This produced the following performance:

![KNN Score](/images/score_knn.PNG)

The key observation here is that we now see an **accuracy of 0.70.** Recall on class 2 barely changed, and class 1's precision and recall are both abysmally low. This second piece will be a theme throughout. Perhaps, the class imbalance was causing this problem. So the data was SMOTE'd and used to retrain the same setup of classifier.

![KNN Score With SMOTE](/images/score_knn_smote.PNG)

Strangely enough, this made things worse! This **lowered the overall accuracy** and **increased class 1's recall,** which I'm not too concerned about. Since the main improvement is minimal and less-important, and the data is now synthetic, SMOTE was not used for the rest of the models.

## Decision Tree

Again, the same basic method was followed. Max depth, minimum samples per split, and minimum samples per leaf were all tested in the grid search. Expectations were lower, but wrongly so:

![Tree Score](/images/score_tree.PNG)

This model showed a **2 percentile point increase in accuracy** as well as a **10 percentile point increase in class 2 recall.** This was now the model to be compared against.

## Random Forests

The RF model ended up being the best performer, and was tested with the same parameters as the decision tree model.

![RF Model Score](/images/score_rf.PNG)

This model showed up the competition with an **accuracy of 77% (a 5pp increase), and class 2 precision and recall of 78% and 72% respectively.** We also see a **decrease in misclassifications for class 0** (functional wells), and relatively little change for the liminal class 1.

## Gradient Boosting

The gradient boosted model was trained just for thoroughness sake, and grid searched on the parameters of max depth and learning rate. It performed nearly as well as the random forest model:

![GB Model Score](/images/score_grad.PNG)

We see a few metrics drop by a percentage point or two, but **generally the same results as the previous model.**

## Modeling Summary

The final model, as revealed previously, was decided to be **the RandomForestClassifier.** This model was chosen for a few reasons beyond its higher performance. If this model were to be built on more and/or reused for well data elsewhere in the world, the lower computational cost and hyperparameter tuning required to build RF models beats out GB. Moreover, GB models can tend towards overfitting data due to their sequential nature of training. If this well-classification model were to be used in other situations, perhaps by the same aid organizations who were kept in mind through its training, the overfitting problem could rear its ugly head.

# Conclusions

## Recommendations

While this project doesn't exactly have any *business* recommendations, per se, they are a couple key insights that can be used in the field when assessing well functionality and predicting its cause(s). This plot below illustrates feature importance in predicting a well's functionality. In other words, it shows what data is most vital in getting that critical guess right. (Orange bars are just to draw attention to most significant features, they are not discrete categories.)

![Feature Importances For Well Classification](/images/rf_model_importances.PNG)

A well's **latitude, longitude, and height** all play a clear role in its classification. So do it's **construction year** and--interestingly enough--the **day of year** the *recording* was made. Lastly, the **extraction class** plays a key role, specifically when it's been glossed as "other."

> Note: I am not sure why ID is a powerful predictor here. But I suspect it may have a relationship to the time the ID was assigned, though I was unable to identify any datetime patterns myself.

It is no surprise that a well's geospatial data provides important predictors, seeing as the previous scatter plot showed clustering. It *is,* however, a surprise that the day a well's status was recorded plays a role, as does the elusive "other" form of extraction. As such I encourage the well managers and water committees of the world to:
1. **Pay special attention to data-entry bias throughout the year.** Perhaps record-takers draw conclusions more stricly when weather is harsh, for example.
1. **Carefully monitor the usage of non-standard extraction methods.**
1. **Try to avoid building wells at GPS heights that correlate with well failure.**

## Future Research

In my future research, I'd like to dig deeper into the GB and RF models both and see how much hyperparameter tuning can improve their performance, given the time and resources. Moreover, I'd like to see how well this well-classifier performs *outside* of Tanzania. Do other East African countries follow similar trends in well functionality? What about somewhere geologically distinct, like the American Midwest?

More specifically, I'd also like to do more research into this "day of year" phenomenon, as well as *which* "other" extraction methods are most problematic, and whether or not their failure correlates with other variables. (Do certain "other" extractors perform better when installed in different seasons? What about at different elevations?)

## Thank You

I would like to thank the government of Tanzania for funding this research. I look forward to continuing this business relationship. *Asante sana!*
