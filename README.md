🎓 Student Performance Prediction App

A machine learning web app built using Streamlit that predicts a student's exam score based on various lifestyle and academic factors.


Features:

Predicts exam score using a trained ML model

Interactive web interface with Streamlit

Real-time predictions based on user input

Clean and user-friendly UI


Model

Algorithm: Linear Regression

Trained on student performance dataset

Includes preprocessing:

Encoding categorical variables

Handling missing values

Feature selection

📂 Project Structure

student-performance-app/
│
├── app.py                # Streamlit app
├── train_model.py        # Model training script
├── best_model.pkl        # Saved trained model
├── requirements.txt      # Dependencies
└── README.md             # Project documentation


⚙️ Installation

1. Clone the repository

git clone https://github.com/your-username/student-performance-app.git
cd student-performance-app

2. Install dependencies

pip install -r requirements

📊 Input Features

The model uses features like:

Gender

Part-time job

Diet quality

Parental education level

Internet quality

Study habits

Sleep hours

Attendance

Mental health

Extracurricular participation


📈 Output

Predicted Exam Score

Tools Used:

Python

Pandas & NumPy

Scikit-learn

Streamlit

Joblib


Author:
Alia Al-Qadri 
(Data Science & Medical AI Enthusiast)
