# CardiAI - Clinical Heart Disease Risk Diagnostics Portal

An end-to-end MLOps web application built with FastAPI, Support Vector Machines (SVM), and SHAP (Explainable AI) to evaluate cardiovascular disease risk profiles and generate automated clinical reports.

## Live Demo
https://cardiai-risk-app.onrender.com

## Key Features
- Machine Learning Engine: Powered by a pre-trained SVM pipeline for heart disease risk estimation.
- Explainable AI (XAI): Integrated dynamic SHAP (KernelExplainer) attribution models to interpret feature importance for clinical decision support.
- Interactive Medical UI: Modern, clinical-grade responsive interface with interactive toggle controls and real-time visualization gauges using ApexCharts.
- Automated Clinical Reporting: Single-page PDF medical report export containing patient vitals summary, diagnostic assessment, and XAI charts.
- Containerized Deployment: Fully dockerized application ready for production cloud environments via Render and Docker.

## Tech Stack
- Backend: FastAPI, Uvicorn, Pydantic
- Machine Learning & XAI: Scikit-Learn, Joblib, Pandas, NumPy, SHAP
- Frontend & Visualization: HTML5, CSS3, JavaScript, ApexCharts, html2pdf.js
- Containerization & Hosting: Docker, Render

## Project Structure
```text
.
├── main.py
├── model.joblib
├── requirements.txt
├── Dockerfile
└── README.md
