# 💳 Razorpay AI Risk Manager

![Razorpay AI Risk Manager](screenshots/risk-manager.png)

## 🤖 AI-Based Fraud Transaction Detection System

Razorpay AI Risk Manager is a machine learning-powered web application designed to detect potentially fraudulent financial transactions.

The application accepts transaction details, performs feature engineering, and uses a trained Random Forest classification model to predict whether a transaction is **NORMAL** or **FRAUD**.

It also calculates a fraud probability and assigns a corresponding risk level:

- 🟢 **LOW** — Low Risk
- 🟡 **MEDIUM** — Medium Risk
- 🔴 **HIGH** — High Risk

---

## 🌐 Live Demo

🚀 **Try the application:**  
https://razorpay-ai-risk-manager-f081.onrender.com

> Note: The application is hosted on Render's free tier, so the first request after inactivity may take some time to load.

---

## 📌 Project Overview

Financial fraud detection is an important problem in digital payment systems.

This project demonstrates how Artificial Intelligence and Machine Learning can be used to analyze transaction patterns and identify potentially fraudulent transactions.

The system processes transaction information, generates additional features, and passes the processed data to a trained machine learning model.

The final prediction includes:

- Fraud / Normal classification
- Fraud probability
- Risk level
- Transaction risk assessment

---

## ✨ Features

- 🤖 AI/ML-based fraud detection
- 🔍 Transaction risk analysis
- 📊 Fraud probability calculation
- 🚨 LOW / MEDIUM / HIGH risk classification
- 🌐 Flask-based web application
- 🧠 Random Forest machine learning model
- ⚙️ Feature engineering
- 💻 User-friendly web interface
- 📈 Model-based prediction
- 🔐 Financial transaction risk assessment

---

## 🔄 System Workflow

```text
                User
                  │
                  ▼
        Enter Transaction Details
                  │
                  ▼
         Flask Web Application
                  │
                  ▼
           Input Processing
                  │
                  ▼
          Feature Engineering
                  │
                  ▼
        Trained Random Forest
             ML Model
                  │
                  ▼
          Fraud Probability
                  │
                  ▼
           Risk Assessment
                  │
          ┌───────┴────────┐
          ▼                ▼
       NORMAL            FRAUD
          │                │
          └───────┬────────┘
                  ▼
             Display Result