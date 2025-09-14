# 🩺 Kidney Stone Prediction (Educational Project)

This project demonstrates how to build a **machine learning pipeline** for predicting kidney stones based on urine analysis.  
It covers data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, visualization, and deployment with Streamlit.  

⚠️ **Note:** This project is for **educational purposes only**.  
It is **not a medical tool** and should not be used for real-world diagnosis or treatment.

---

## 🚀 Try the App
[Try it on Streamlit Cloud](<your-link-here>)

---

## 📊 Dataset
- The dataset was obtained from Kaggle.  
- It contains urine analysis results with features such as pH, specific gravity, red blood cells, white blood cells, etc.  
- **Target variable:** Presence or absence of kidney stones.  

![](images/1.png)
---

## 🧹 Data Preprocessing

Steps included:

* Handling missing values
* Encoding categorical variables
* Normalization / scaling
* Train-test split

---

## 🔎 Exploratory Data Analysis (EDA)

* Distribution plots for numerical features
* Count plots for categorical features
* Correlation heatmap

📷 Example plots (replace with your paths):

![](images/2-1.png)  
![](images/2-2.png)  
![](images/2-3.png)  
![](images/2-4.png)  
![](images/2-5.png)  


---

## 🛠️ Feature Engineering

* Outlier detection and handling
* Derived features from urine analysis readings
* Feature importance analysis

---

## 🤖 Model Training

The following models were trained and compared:

* Logistic Regression
* Support Vector Machine (SVM)
* k-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting

---

## 📈 Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC

📷 Example metrics visualization:

![](images/3-1.png)  
![](images/3-2.png)  
![](images/3-3.png)  
![](images/3-4.png)  
![](images/3-5.png)

---

## 🌳 Decision Trees & Ensemble Models

* Visualization of trained Decision Trees
* Feature importance from Random Forest / Gradient Boosting

📷 Example placeholders:

![](images/4-1.png)  
![](images/4-2.png)  
![](images/4-3.png)  
![](images/4-4.png)  
![](images/4-5.png)  
![](images/4-6.png)  
![](images/4-7.png)

---

## 💻 Streamlit App

A Streamlit app was built for interactive prediction:

* Users can input urine test results
* Model outputs prediction (kidney stone present or not)

📷 Placeholder for screenshot:

![](images/streamlit_app.png)

---

## 📌 Disclaimer

This project is **educational only**.
It should **not** be used for real medical decision-making.
Always consult a healthcare professional for diagnosis and treatment.
