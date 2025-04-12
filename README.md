# Project: Regression of Used Car Prices with Deep Learning

### Overview
The main goal of this project was to gain hands-on experience with deep learning and artificial neural networks (ANNs) for used car price prediction. Using a dataset with various features such as make, model, year, mileage, and fuel type, we aimed to develop a machine learning model that accurately predicts car prices. The project involved performing exploratory data analysis (EDA), feature engineering, and building a deep learning model using neural networks for regression tasks.

---

### Objectives
1. **To develop a deep learning model using artificial neural networks (ANNs) to predict used car prices.**
2. **To practice implementing techniques like regularization, early stopping, and hyperparameter tuning in deep learning models.**
3. **To handle data preprocessing, feature engineering, and optimize model performance through experimentation with different techniques.**
4. **To assess the performance of the deep learning model using appropriate evaluation metrics and fine-tune it for improved accuracy.**

---

### Steps Followed

#### **1. Data Loading and Preprocessing**
- **Dataset**: The raw data was loaded from `../data/raw/cars.csv`.
- **Initial Exploration**: A general analysis of the dataset was performed to understand the features and the relationships between them. Visualizations were used extensively to gain insights into the data.
- **Target Variable**: The target variable, `price`, was selected for regression tasks.
- **Missing Values**: I filled missing values for most features, but `fuel_type` was left out as it showed weak correlation with the target variable, `Price`.
- **Feature Engineering**: 
  - I examined the features `price` and `mileage` for skewness, kurtosis, and outliers.
  - After analyzing the distributions, I decided to keep these features in their raw form, as no significant issues were found.
  - Categorical variables were encoded using a `qual_mapping` technique for appropriate preprocessing.

---

#### **2. Feature Engineering**
To improve the model's accuracy, various feature engineering techniques were applied:

1. **Handling Missing Data**: Missing values for features like `fuel_type` were not imputed because they didn't significantly impact the target variable’s correlation.
2. **Outlier Handling**: I performed outlier detection for `price` and `mileage` and kept them in their original form as no major issues were found after analysis.
3. **Categorical Data**: Categorical columns such as `brand`, `model`, and `fuel_type` were encoded using custom mappings.

---

#### **3. Deep Learning Model Development**

I implemented a deep learning model using **Artificial Neural Networks (ANNs)** for used car price prediction. The model architecture and training process were designed to optimize performance and reduce overfitting.

- **Model Architecture**: 
  - The model consisted of 5 dense layers with ReLU activation functions.
  - Dropout layers were used to prevent overfitting by randomly setting a fraction of input units to zero during training.
  - The model utilized **StandardScaler** for feature standardization, which helped improve the training process.
  - **Adam optimizer** was used to improve convergence speed and accuracy.

- **Regularization and Callbacks**: 
  - Early stopping was used to halt training once the validation loss stopped improving, preventing unnecessary overfitting.
  - `ReduceLROnPlateau` was used to reduce the learning rate when the model’s performance plateaued during training, which helped achieve better optimization.

- **Model Training**:
  - The model was trained for **100 epochs** and evaluated at each step using the **RMSE (Root Mean Squared Error)** metric to assess its accuracy.
  - The model's weights were saved in `.h5` format for future use and testing.

---

#### **4. Model Evaluation**

The model’s performance was evaluated using multiple regression metrics, primarily **RMSE**, to understand how well it predicted car prices.

- **Early Stopping** and **ReduceLROnPlateau** were both instrumental in optimizing the model's training process.
- After training, the model was tested using a separate test set to evaluate its ability to generalize.
- The final model, trained with the best possible parameters, was saved for future use and deployment.

---

### Challenges and Solutions

1. **Handling Missing Data**:
   - **Challenge**: Some features had missing values, but they didn't significantly correlate with the target variable.
   - **Solution**: I chose not to impute the `Fuel_Type` variable, focusing on the most important features.

2. **Overfitting**:
   - **Challenge**: The model risked overfitting due to the complexity of neural networks.
   - **Solution**: Dropout regularization was applied, and early stopping was used to halt training once the model's performance stopped improving.

3. **Feature Scaling**:
   - **Challenge**: Ensuring that the deep learning model's inputs were appropriately scaled for faster and more accurate convergence.
   - **Solution**: StandardScaler was applied to standardize the features, improving the model's training efficiency.

4. **Optimization**:
   - **Challenge**: Fine-tuning the model's hyperparameters to achieve the best performance.
   - **Solution**: Used callbacks like **ReduceLROnPlateau** to adjust the learning rate dynamically during training.

---

### Tools and Libraries Used
- **Programming Language**: Python
- **Libraries**:
  - pandas, numpy (Data manipulation)
  - scikit-learn (Feature engineering and preprocessing)
  - tensorflow.keras (Deep learning model development)
  - matplotlib, seaborn (Visualization)

### File Structure
```plaintext
Project/
├── data/
│   ├── raw/
│   ├── processed/
├── models/
│   ├── car_price_prediction_model.h5
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
│   ├── test_data_processing.ipynb
│   ├── final_test_predictions.ipynb
├── submission/
│   ├── sample_submission.csv
├── README.md
├── environment.yml
├── requirements.txt
```
---

### Installation
To set up the project environment:

Using Conda:
```bash
conda env create -f environment.yml
conda activate regression-of-used-car-prices-with-deep-learning
```

Using Pip:
```bash
pip install -r requirements.txt
```

---

### Conclusion
Through this project, I gained valuable practical experience working with deep learning models, particularly in the context of predicting used car prices. By using neural networks, regularization techniques, and hyperparameter tuning, I was able to optimize the model for better accuracy. This project provided insight into handling real-world data challenges like missing values, feature scaling, and overfitting.

Future improvements could involve experimenting with more advanced neural network architectures or additional optimization techniques to further enhance model performance.

---

### Acknowledgements
Special thanks to Kaggle for providing the dataset and the inspiration for this challenge.

---

---
