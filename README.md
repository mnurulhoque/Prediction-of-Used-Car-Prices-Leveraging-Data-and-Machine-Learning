# Prediction-of-Used-Car-Prices-Leveraging-Data-and-Machine-Learning

## OVERVIEW OF THE PROJECT
This project aims to develop a machine learning model capable of predicting the resale price of used cars listed on platforms like Craigslist. By leveraging historical data and machine learning algorithms, the project seeks to achieve the following objectives:

**i. Accurate Price Predictions:** Develop a predictive model to estimate car prices based on key features such as vehicle specifications (e.g., odometer, make, model, condition, transmission type) and seller information (e.g., location, fuel type, cylinders).

**ii. Feature Importance and Interpretability:** Identify and interpret the key features driving car price predictions using SHAP values.

This project bridges the gap between subjective car valuations and objective, data-driven pricing models, making car pricing more transparent, consistent, and fair for stakeholders.

## OVERVIEW OF THE DATASET
The dataset includes a variety of features describing cars listed on Craigslist. It contains 26 columns, including both numerical and categorical features. Some notable features include:

-`year`: The manufacturing year of the car.

-`odometer`: The total mileage of the car.

-`manufacturer`: The car's manufacturing company.

-`model`: The specific model of the car.

-`fuel`: The type of fuel used (e.g., gas, diesel, electric).

-`transmission`: The type of transmission (automatic, manual, etc.).

-`lat` and `long`: Geographic location coordinates of the car listing.

The target variable for the prediction task is `price`, which represents the sale price of each car. 

The dataset has been downloaded from Kaggle. Here is the Link: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

## INITIAL DATA EXPLORATION
To understand the structure of the dataset, the following actions were performed:

**Dataset Information:** The dataset has 14,101 rows and 26 columns. It includes numerical data (year, odometer, lat, long) and categorical data (manufacturer, model, fuel, transmission).

**Summary Statistics:** Basic statistics such as mean, standard deviation, minimum, and maximum values were calculated for numerical columns like year and odometer. For example, the mean year of cars is approximately 2011, and the average mileage (odometer) is 101,914.

**Missing Values:** A significant number of missing values were detected in several columns, such as condition, cylinders, drive, and size.
A missing values heatmap was generated to visually identify columns with high percentages of missing data.

![Missing values before cleaning](https://github.com/mnurulhoque/Prediction-of-Used-Car-Prices-Leveraging-Data-and-Machine-Learning/blob/main/Missing%20values%20before%20cleaning.png)

## DATA PREPROCESSING

**Handling Missing Data:** Given the presence of missing values in key columns, preprocessing steps were applied:
Columns such as url, region_url, VIN, image_url, description, and county were deemed irrelevant to the prediction task and removed.
We handled missing values in the dataset by imputing numerical and categorical columns with appropriate statistical measures. For numerical columns, identified using select_dtypes(include=[np.number]), missing values are replaced with the median of the column. The median is chosen because it is robust to outliers, ensuring the central tendency of the data remains unaffected by extreme values. This imputation is performed in-place, directly modifying the original dataset. For categorical columns, identified using select_dtypes(include=[object]), missing values are replaced with the mode of the column, which represents the most frequently occurring value. Using the mode for categorical data ensures the imputed values align with the most common category, preserving the original distribution. This method avoids the loss of information associated with dropping rows or columns with missing data and is dynamic, automatically identifying and imputing columns based on their data types. Overall, this approach effectively preserves data integrity and ensures the dataset is ready for analysis or modeling.
The heatmap of the missing values after data cleaning has been given below: 

![Missing values after cleaning](https://github.com/mnurulhoque/Prediction-of-Used-Car-Prices-Leveraging-Data-and-Machine-Learning/blob/main/Missing%20values%20after%20cleaning.png)

**Balancing the Dataset:** Before balancing, the distribution of the target variable in the training data shows a significant imbalance across the defined bins. As evident in the first chart ("Distribution of Target Variable in Training Data"), one bin (e.g., bin 1) dominates the dataset with the highest frequency, while other bins, particularly bin 0, have considerably fewer samples. This imbalance can skew machine learning models, causing them to become biased towards the majority class, ultimately reducing their performance on underrepresented classes.

To address this, upsampling is used to balance the dataset. The process involves first combining the training features (X_train) and target variable (y_train) into a single DataFrame. Each target value is grouped into bins (e.g., bin column), which highlights the imbalance. Then, the resample function is applied to the underrepresented bins, duplicating their samples until each bin reaches the same number of samples as the majority class (the maximum bin count). The upsampling is performed with replacement, ensuring that the original distribution of each class within the bins is preserved while increasing their representation. The balanced dataset is then constructed by concatenating all the resampled subsets for each unique bin.

After balancing, as shown in the second chart ("Distribution of Target Variable After Balancing"), the frequencies of all bins are equalized. This improvement ensures that the dataset no longer exhibits class imbalance, which helps to mitigate biases during model training. As a result, the model can learn equally from all classes, improving its ability to generalize and predict accurately across all categories. This enhancement is particularly important for tasks requiring fair and representative predictions across different classes.

**Encoding Categorical Variables:** We implemented mean encoding to transform categorical features into numerical representations based on their relationship with the target variable (price). A list of categorical features, such as manufacturer, model, condition, fuel, transmission, drive, type, state, and cylinders, is specified for encoding. For each feature in this list, the code calculates the mean of the target variable (price) grouped by the unique categories of that feature using the groupby method. This results in a mapping where each category is replaced by the average price of the samples belonging to that category. The map function then applies this mapping to transform the original categorical values in the dataset to their corresponding mean-encoded values. By doing so, mean encoding captures the statistical relationship between each category and the target variable, enhancing the dataset's predictive power while converting categorical data into a numerical format suitable for machine learning models. This approach can be particularly useful for models that benefit from numerical features but requires careful handling to prevent data leakage, such as during cross-validation.

**Feature Scaling:** Numerical features like year and odometer were normalized using the QuantileTransformer method. This transformation ensures that the numerical values follow a normal distribution, improving model performance.

**Feature Engineering:** To enhance the predictive power of the model, an interaction feature year_odometer was created by multiplying year and odometer. Additionally, the target variable price was log-transformed using log1p to reduce skewness and stabilize variance.

## MACHINE LEARNING MODEL PIPELINE DESIGN, TRAINING, AND OPTIMIZATION 

### Model Selection:
Four machine learning models were selected for evaluation:

i. Random Forest: An ensemble model that builds multiple decision trees to improve accuracy.

ii. XGBoost: A gradient-boosting algorithm optimized for speed and performance.

iii. LightGBM: A lightweight boosting algorithm suitable for large datasets.

iv. CatBoost: A gradient-boosting algorithm specifically designed for categorical features.

### Model Justification and Fine-Tuning:
The models were chosen based on their proven performance with tabular datasets and their ability to handle numerical and categorical features (Varma et al., 2023; Wang et al., 2022). To improve performance, hyperparameter tuning was performed using Optuna, an automated optimization framework:

### Model Evaluation and Cross-Validation:
Model evaluation was performed using a 5-fold Cross-Validation strategy to ensure that the models generalize well across unseen data. Performance was measured using:

**i. Root Mean Squared Error (RMSE):** Measures prediction error magnitude. It is sensitive to large errors because it squares the residuals, making it an excellent measure when large deviations in price prediction are critical to detect (Hodson, 2022). Compared to Mean Absolute Error (MAE), RMSE penalizes large errors more severely, which aligns with the goal of minimizing significant mispredictions.

**ii. R² Score:** Explains the proportion of variance captured by the model. This metric indicates how well the model explains variance in the target variable. It helps evaluate the goodness-of-fit for regression models and compares model performance across iterations.

### Baseline Model Performance:
Each model was trained on the preprocessed dataset, and their performance was evaluated using Root Mean Squared Error (RMSE) and R² score. The baseline results are as follows:

### Model Performance After Optimization:
After hyperparameter tuning, the optimized performance results are as follows:

### Best Model Selection:
The best-performing model for predicting car prices is XGBoost, which achieved the lowest RMSE (Root Mean Square Error) of 0.4249 and the highest R² (coefficient of determination) of 0.7954 among the optimized models. These metrics indicate that XGBoost not only provides the most accurate predictions by minimizing the average squared differences between predicted and actual values but also explains approximately 79.54% of the variance in the target variable. Compared to the other models, including Random Forest, LightGBM, and CatBoost, XGBoost strikes the best balance between precision and generalizability, making it the optimal choice for this predictive task.

## MACHINE LEARNING MODEL INTERPRETABILITY 
The SHAP summary plot provides insights into the impact of features on the XGBoost model's predictions for car price prediction. The horizontal axis represents the SHAP values, which indicate the impact of each feature on the model's output, while the vertical axis lists the features in descending order of importance.
The most influential features are model, year, and odometer, as they have the widest range of SHAP values, meaning their variations significantly affect the predictions. For example, model shows both positive and negative SHAP values, indicating that different models can either increase or decrease the predicted price. Similarly, year shows that newer vehicles generally lead to higher predicted prices (positive SHAP values), while older vehicles decrease it.
Features such as latitude (lat), manufacturer, and type also play a substantial role, although their impacts are less pronounced than the top features. These features capture geographical and categorical distinctions that affect pricing. On the other hand, features like state, fuel, and condition have lower SHAP values, suggesting they contribute less to the model's predictions.
The color gradient represents feature values, where red indicates high values and blue indicates low values. For instance, for year, red points (newer cars) have positive SHAP values, contributing to higher predictions, while blue points (older cars) decrease predictions. This detailed visualization helps interpret the model's behavior and understand the factors influencing car price predictions.

This LIME explanation provides a detailed breakdown of a single prediction for car price, offering insights into how specific features influenced the model's output. The model predicted a price of 21,286.28, while the actual price was 16,094.99, indicating some lower predicted value. The prediction falls within a local range of possible values, from approximately 6.45 to 12.61. Key features influencing the prediction are displayed as either negative (blue), pulling the predicted value down, or positive (orange), pushing it up. The feature model contributes significantly in a negative direction, decreasing the predicted price by about 13,252.36 units, while condition and cylinders have strong positive contributions, adding approximately 4,545.74 and 2,716.17 units, respectively. Features such as drive and year also play a role, with drive slightly decreasing the prediction by 2,089.95 units and year (2013) increasing it by 2,020.40 units. Other features, including manufacturer, state, and type, have minor impacts on the prediction. The table on the right lists the actual input values for these features, providing context for their contributions. Overall, this explanation highlights the transparency of the model's decision-making process and identifies the most influential factors for this specific car's predicted price.

This LIME explanation illustrates the model's prediction for a specific car, providing insights into the features that influenced the output. The model predicted a price of 30,958.29, while the actual price was 36,494.99, showing a moderate higher predicted value. The prediction falls within a local range, from a minimum of 5.95 to a maximum of 12.66. The key influencing features are categorized into negative (blue), reducing the predicted price, and positive (orange), increasing it. Among the features, odometer and year make the largest positive contributions, adding 28,137.00 and 20,014.00 units, respectively, to the prediction. Features such as cylinders and condition also push the prediction higher, contributing 2,716.17 and 4,545.74 units, respectively. On the other hand, model has a significant negative impact, decreasing the prediction by 13,252.36 units. Features like latitude (lat), longitude (long), and drive also slightly reduce the predicted price but with less influence compared to the top contributors. The table on the right provides the actual values for each feature, adding context to the analysis. Overall, this explanation highlights how critical features like odometer, year, and model drive the model's predictions, offering transparency into its decision-making process.

## Conclusions
The project successfully developed a machine learning model to predict car prices based on key car attributes. The XGBoost model outperformed other models, achieving the lowest RMSE of 0.4249 and the highest R² score of 0.7954. By leveraging SHAP values, we identified model, year, and odometer as the most influential predictors. 
Overall, the XGBoost model provides a reliable and interpretable solution for predicting car prices. Platforms like Craigslist can integrate this model to offer fair pricing suggestions to users, enhancing transparency and trust in the online used car market. Future work could focus on integrating additional features like car accident history or service records to improve the model's accuracy further.
