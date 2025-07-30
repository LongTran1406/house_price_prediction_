🏠 NSW House Price Prediction
An end-to-end machine learning pipeline for predicting property prices in New South Wales, Australia. This project scrapes real-estate listings, cleans and preprocesses the data, trains and evaluates models, and includes full MLOps integration with DVC, Docker, MLflow, and GitHub Actions.

📂 Project Overview
Scrape property listings from NSW real-estate websites

Clean and transform structured data (numerical & categorical)

Train two models: Linear Regression and XGBoost

Log experiments using MLflow

Version datasets and models using DVC (S3 as remote)

Package with Docker for API deployment

Automate training and deployment with GitHub Actions

⚙️ Tech Stack
Python 3.10

scikit-learn, XGBoost

MLflow for experiment tracking

DVC for dataset/model versioning

Flask API for inference

Docker for containerization

GitHub Actions for CI/CD

AWS S3 for remote storage

📦 Project Structure
.
├── data/
│   ├── raw/                  # Scraped NSW property listings
│   └── processed/            # Cleaned and engineered features
├── models/                   # Saved .pkl models
├── notebooks/                # EDA and experiments
├── src/
│   ├── scraping/             # Web scraping script(s)
│   ├── preprocessing/        # Cleaning, encoding, scaling
│   ├── training/             # Model training + MLflow logging
│   └── utils/                # Helper functions
├── app.py                    # Flask API for inference
├── Dockerfile                # Docker container setup
├── dvc.yaml                  # DVC pipeline stages
├── requirements.txt          # Python dependencies
└── .github/workflows/        # CI/CD configuration

🚀 Getting Started
1. Clone and set up environment
git clone https://github.com/your_username/nsw-house-price-prediction.git
cd nsw-house-price-prediction
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
