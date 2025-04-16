# **Employee Leave Prediction Using Machine Learning**

This project uses machine learning models to predict employee leave behavior based on various features such as education, experience, and demographics. The dataset is used for classification tasks to understand patterns that lead employees to take leave or not. The models implemented include Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM).

---

## **Project Overview**

The goal of this project is to:
1. Perform **Exploratory Data Analysis (EDA)** to better understand the dataset.
2. Preprocess the data for machine learning tasks, including handling missing values, encoding categorical features, and balancing the dataset.
3. Train and evaluate various classification models to predict whether an employee will take leave or not.
4. Select the best performing model and visualize its performance.

---

## **Dataset Description**

The dataset contains employee-related data, including the following features:

- **Education**: The educational qualifications of employees.
- **Joining Year**: The year each employee joined the company.
- **City**: The city where the employee works.
- **Payment Tier**: Categorization of employees into different salary tiers.
- **Age**: The age of the employee.
- **Gender**: The gender of the employee.
- **Ever Benched**: Whether an employee has ever been temporarily without assigned work.
- **Experience in Current Domain**: Years of experience in their current field.
- **Leave or Not**: The target variable indicating whether the employee took leave (1) or not (0).

---

## **Steps Taken**

### **1. Data Preprocessing**
- **Handling Missing Values**: Null or missing values were handled by either removing or imputing them.
- **Encoding Categorical Variables**: Categorical features like `Education`, `City`, and `Gender` were encoded using `LabelEncoder`.
- **Feature Scaling**: Features were scaled using `StandardScaler` to ensure that all features are on the same scale.
- **Data Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the dataset, addressing class imbalances before splitting the data.
  
### **2. Model Training and Evaluation**
- **Models Implemented**: The following models were trained and evaluated:
  - Logistic Regression
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  
  Each model was trained on the resampled and scaled dataset. Performance metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score** were calculated for each model.

- **Model Selection**: Based on evaluation metrics, **Support Vector Machine (SVM)** emerged as the best performing model for this dataset.

### **3. Confusion Matrix and Classification Report**
- For the best model (SVM), a **Confusion Matrix** and **Classification Report** were generated to analyze performance in detail.

### **4. Hyperparameter Tuning**
- Grid Search was used to tune the hyperparameters of the SVM model to improve its performance further.

---

## **Technologies Used**
- **Python**: For data manipulation and modeling.
- **Pandas**: For data cleaning and manipulation.
- **Scikit-learn**: For machine learning models and evaluation.
- **Imbalanced-learn**: For data balancing using SMOTE.
- **Matplotlib & Seaborn**: For data visualization.

---

## **How to Run the Project**

1. **Clone the repository**.

2. **Install the required libraries**.

3. **Run the script**.

This will run the model training, evaluation, and produce the confusion matrix and classification report for the best model (SVM).

---

## **Results**

- **Best Performing Model**: Support Vector Machine (SVM)
- **Evaluation Metrics**: 
    - Accuracy, Precision, Recall, F1-Score for all models were calculated and compared.

- **Confusion Matrix** and **Classification Report** for the **SVM model** show that the model performed well in predicting employee leave behavior.

---

## **Conclusion**

This project successfully predicted employee leave behavior using machine learning models. By preprocessing the data, balancing the dataset, and evaluating various models, SVM was selected as the best model. The project demonstrates the importance of data preprocessing, model selection, and evaluation in achieving optimal results.

---
