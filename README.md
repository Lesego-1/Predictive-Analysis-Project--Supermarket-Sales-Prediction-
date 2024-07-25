# Predictive Analysis on Supermarket Sales Dataset

# Problem being solved
Predict gross income, gross margin percentage and ratings and find best model for predictions.

# Process
1. The data is loaded into a pandas dataframe from the raw_data folder.
2. Data is processed and encoded to maximize model performance.
3. Three Machine Learning Models are used to predict gross income and ratings.
4. The models' performances are evaluated and output to determine which is best to use for this problem.
5. The data is stored into a sqlite database.

# Results
The Ridge Regression Model had the lowest Mean Squared Error and the highest R2 value. This shows that it has the best balance of minimizing error and providing information on the variance of the data.