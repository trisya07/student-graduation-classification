# 🎓 On-Time Graduation Classification

A classification project to predict whether students graduate on time based on academic data using the Random Forest algorithm.

## 📁 Project Structure
```
├── ms_lulusan_fix.csv # Student graduation information
├── transkip_nilai.csv # Transcript data containing grades per semester
├── prediksi kelulusan random forest.ipynb # Main notebook for preprocessing, analysis, and modeling
```

## 🧾 Project Overview

This project aims to classify students as graduating **on time** or **not on time** based on their academic performance. The classification model used is **Random Forest**, and the workflow includes data merging, cleaning, feature engineering, and model evaluation.

## 🔄 Workflow

1. **Data Merging**  
   - Calculate Semester GPA (IPS) from `transkip_nilai.csv`
   - Merge with `ms_lulusan_fix.csv` using a common student identifier

2. **Preprocessing**  
   - Handle missing values, encode categorical features, detect and remove outliers

3. **Exploratory Data Analysis (EDA)**  
   - Analyze trends and distributions related to graduation status

4. **Modeling**  
   - Train and evaluate a Random Forest classifier
   - Use metrics such as accuracy, precision, recall, and F1-score

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Google Colaboratory

## 📌 Notes

This project is for academic and research purposes only.  
Feel free to use or modify it, but please give appropriate credit.
