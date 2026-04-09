# Term-Deposit-Subscription-Predictor

## Source

To view the source code, please refer to this [Jupyter Notebook](./prompt_III.ipynb)

## Data Visualization and Preparation

Based on different visualizations shown in the notebook, most of the categorical data have an `unknown` value. In most cases, there's an insignificant amount to the point where I could just drop those rows without having it affect the data significantly. For the `default` column, there is a significant amount of `unknown` values. For this column, I replaced all `unknown` values with `no` since it's the most common value, with `yes` accounting for less than 1% of the total data. There were also some interesting patterns shown in the categorical graphs such as, most calls were made in `may`, and there doesn't seem to be a correlation between the `day_of_week` and the target variable. In terms of data preparation, I used label encoding for most categorical columns since most of them contained a lot of unique values and they have an order to them. For the rest of the columns, I used one-hot encoding since there wasn't any coherent order between the values.

In terms of numerical columns, there were a decent number of outliers throughout all columns, which meant that I had to do do some normalization on those columns. In addition, because there's a value of `999` in the `pdays` column to signify no prior contact, I had to modify the column by adding a binary column, `is_contacted`, and then changed the value of `pdays` from `999` to the median `pdays` value to prevent my models from developing a bias (for decision tree model, I left the `999` value in the data). 

The heatmap with the economic indicators (e.g., `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) shows some interesting correlations. Some of which include that `emp.var.rate` is positively correlated with `euribor3m`. Similarly with `emp.var.rate` and `nr.employed` & `euribor3m` and `nr.employed`.

Finally, when plotting the distribution of the target variable, there was significantly more data samples with a value of `no`. Because of this I prepared two sets of data for my models to train on, the original prepped data, and an expanded and prepped dataset using SMOTE to increase the number of samples that have a value of `yes`. This is to increase the precision and recall of models in the positive case.

For the final data preparation, I tailored the data to the model. For Logistic Regression and SVM models, I standardized the data in order for all features to contribute equally. For KNN, I normalized the data since this model relies on distance between points. For the decision tree, I didn't perform any standardization or normalization nor did I remove the `999` values since this model can successfully handle those cases by just adding an addition branch to cover that case.

## Models and Results

I trained 4 different models, Logistic Regression, KNN, Decision Tree, and SVM. Originally, I trained all these models with the default parameters and when looking at their accuracy, all of them besides the Decision Tree model barely did better than the baseline.

I then ran grid search on all 4 models to test different hyperparameters on both the original dataset and the extended dataset with SMOTE. This time instead of just looking at the accuracy, I looked at the classification report to see the precision, recall and f1-score which helps identify how well each model does at predicting `yes` cases.

When comparing the two classification reports for each model, for all models when using the SMOTE data, although the accuracy slightly decrease the precision, recall, and f1-score all increased by a significant amount, meaning it's able to predict more of the `yes` cases. When the models were trained on the non-SMOTE data, the precision, recall, and f1-score were all pretty low even though their accuracy was high. This is due to the imbalanced classes. Since there aren't that many data samples where the target variable value was `yes`, it's harder for the models to identify patterns and successfully and accurately predict the `yes` cases.

Overall, the models all performed similar to each other in terms of precision and recall without SMOTE data, performing quite well when predicting `no` values but all had poor performance when trying to accurately predict `yes` values. When using SMOTE data, all models had their precision and recall for the `no` cases drop slightly while they increased by a significant amount for the `yes` cases. Out of the 4 models, although they were all quite close to each other, the decision tree model performed the best with the SMOTE data with a precision of 0.77 and recall of 0.53 for the `yes` cases and a precision of 0.89 and recall of 0.96 for the `no` cases.