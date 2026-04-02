# 🚀 VISA Processing Time Estimator

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-LICENSE-green)

---

## 🔗 Live Demo
🌐 https://visa-processing-time-estimator.vercel.app

---

## 📝 Description

The **VISA Processing Time Estimator** is a data-driven web application built using Python and Machine Learning to predict visa approval status and processing durations.

By analyzing historical visa application data and applying advanced feature engineering techniques, the system provides:
- ⏳ Estimated processing time  
- ✅ Visa approval/rejection prediction  
- 📊 Data insights and analytics  

This project aims to bring **transparency, planning efficiency, and better decision-making** for users applying for international visas.

---

## ✨ Key Features

- 📊 **Exploratory Data Analysis (EDA)**
  - Automated visualizations (heatmaps, distributions, pairplots, etc.)
  - Stored in `outputs/` directory

- 🧠 **Machine Learning Models**
  - Visa Status Classification (Approved / Rejected)
  - Processing Time Regression Model
  - Feature selection using saved pipelines

- 🌐 **Interactive Frontend**
  - Chat-like assistant for user queries
  - Insights dashboard with analytics
  - Full-screen UI for better experience

- 🔌 **REST API (Flask)**
  - Predict visa status & processing time
  - Fetch analytics and stats

- 📁 **Data Handling**
  - Structured dataset for training
  - Website analytics tracking via JSON

---

## 📊 Project Insights (from Website Analytics)

- 👥 Total Visits: **240**
- 📈 Monthly Visits (sample): `[0, 0, 129, 111, 0, ...]`
- ❌ Latest Prediction Verdict: **Rejected**

> ⚠️ Note: Analytics stored in `website_data/analytics.json`

---

## 🛠️ Tech Stack

### Backend
- 🐍 Python  
- Flask (API development)  
- Pandas, NumPy (Data processing)  
- Scikit-learn, XGBoost (ML models)  

### Frontend
- HTML, CSS, JavaScript  
- Chat UI + Dashboard UI  

### Deployment
- Vercel (Frontend)  
- Gunicorn (Production server)  

---

## 📦 Key Dependencies

```

flask: 1.1
flask-cors: 3.0
joblib: latest
pandas: latest
numpy: latest
scikit-learn: latest
gunicorn: latest
xgboost: latest

```

---

## 📁 Project Structure (Updated)

```

.
├── Documentation
│   ├── Venkat_Ratnam_Agile_doc.xls
│   ├── Venkat_Ratnam_Defect_Tracker.xlsx
│   └── Venkat_Ratnam_Unit_Test_Plan.xlsx
├── LICENSE
├── anaconda_projects
│   └── db
│       └── project_filebrowser.db
├── website_data
│   ├── analytics.json
│   ├── feedbacks.json
│   └── model_metrics.json
├── frontend
│   ├── app.js
│   ├── chat.html
│   ├── dashboard.html
│   ├── feedback.html
│   ├── index.html
│   ├── insights.html
│   ├── predict.html
│   ├── stats.html
│   └── styles.css
├── models
│   ├── processing_days_model.pkl
│   ├── processing_days_model_nofeatures.pkl
│   ├── selected_features.pkl
│   ├── visa_status_model.pkl
│   └── visa_status_model_nofeatures.pkl
├── outputs
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── missing_values_heatmap.png
│   ├── numerical_distribution.png
│   ├── pairplot.png
│   ├── processing_time_distribution.png
│   ├── visa_status_avg_processing.png
│   └── visa_status_boxplot.png
├── requirements.txt
└── src
├── app.py
├── dataset
│   └── VisaFile.csv
├── eda.py
├── eda_select_features.py
├── evaluate_models.py
├── inference.py
├── predict.py
├── train_classifier.py
├── train_model.py
├── train_models.py
└── visa_preprocessing.py

````

---


## 🚀 How to Run the Project

### 1️⃣ Setup Environment

```bash
python -m venv venv
````

Activate:

* Windows: `venv\Scripts\activate`
* Mac/Linux: `source venv/bin/activate`

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run EDA (Generate Visualizations)

```bash
python src/eda.py
```

📁 Outputs will be saved in `outputs/`

---

### 4️⃣ Run Backend Server

```bash
python src/app.py
```

OR (Production):

```bash
gunicorn src.app:app
```

---

### 5️⃣ Run Frontend

```bash
cd frontend
python -m http.server
```

Open:

```
http://localhost:8000/chat.html
```

---

## 🔎 What This Project Demonstrates

* End-to-end ML pipeline (EDA ➝ Training ➝ Inference)
* Real-world dataset simulation for visa processing
* Feature engineering & model optimization
* Full-stack integration (ML + API + UI)
* Deployment-ready architecture

---

## ✅ Recommended Future Improvements

* 🔄 Real-time immigration data integration
* 🤖 Explainable AI (XAI) for transparency
* 📊 Advanced analytics dashboard
* 🔐 Authentication & user profiles
* ☁️ Full cloud deployment (Backend + DB)

---

## 👥 Contributing

Contributions are welcome!

1. Fork the repository
2. Clone your fork
3. Create a new branch
4. Commit your changes
5. Push to GitHub
6. Open a Pull Request

---

## 📜 License

This project is licensed under the **LICENSE** License.

---

## ❤️ Acknowledgement

Developed as part of an AI-driven project to improve transparency and efficiency in visa processing predictions.

---

⭐ *If you like this project, consider giving it a star on GitHub!*

```

---
