# Car Price Prediction - Linear Regression

This repository contains a **Linear Regression Model** project for predicting the selling prices of cars based on various features. The project involves data preprocessing, model training, evaluation, and insightful visualizations.

## Key Features
- **Data Preprocessing**: Handled categorical variables using `LabelEncoder` and prepared the data for modeling.
- **Model Training**: Built and trained a **Linear Regression** model using the training data.
- **Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: Measures average prediction error.
  - **Mean Squared Error (MSE)**: Penalizes larger prediction errors.
  - **R-squared (R2) Score**: Explains the proportion of variance captured by the model.
- **Visualizations**:
  - Scatter plot of actual vs. predicted prices.
  - Residual plot for heteroscedasticity analysis.
  - Distribution of residuals for error normality checks.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: For data manipulation and preprocessing.
- **Seaborn & Matplotlib**: For data visualization.
- **scikit-learn**: For building and evaluating the regression model.

## Project Objectives
- Predict the selling price of cars based on their features.
- Analyze model performance and identify any patterns in prediction errors.
- Visualize insights from the dataset and model.

## Repository Structure
- `car data.csv`: The dataset containing car features and prices.
- `car_price_prediction.py`: The Python script implementing the linear regression model.
- `README.md`: Project documentation and overview.

## Visual Insights
1. **Actual vs. Predicted Prices**: Shows how well the model predicts selling prices.
2. **Residual Plot**: Ensures random distribution of residuals to validate regression assumptions.
3. **Residual Distribution**: Confirms errors follow a normal distribution.

## How to Run the Code
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
