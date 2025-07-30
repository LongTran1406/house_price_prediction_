<!-- Back to Top Anchor -->
<a id="readme-top"></a>

<!-- SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- LOGO -->
<br />
<div align="center">
  <h1 align="center">🏠 NSW House Price Prediction</h1>
  <p align="center">
    End-to-end ML pipeline to predict NSW property prices.
    <br />
    <a href="#about-the-project"><strong>Explore the docs »</strong></a>
    <br />
    <a href="#usage">View Usage</a>
    ·
    <a href="https://github.com/your_username/nsw-house-price-prediction/issues">Report Bug</a>
    ·
    <a href="https://github.com/your_username/nsw-house-price-prediction/issues">Request Feature</a>
  </p>
</div>

---

## 📖 About The Project

This project builds an automated pipeline to scrape real estate listings in New South Wales, clean and process the data, train ML models to predict house prices, and deploy the results using modern DevOps tools.

Key Features:
- Scraping from NSW property sites
- Data cleaning and preprocessing
- Model training (Linear Regression, XGBoost)
- MLflow experiment logging
- DVC tracking of data and models
- Docker-based deployment
- CI/CD with GitHub Actions

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### 🛠️ Built With

- Python 3.10
- scikit-learn
- XGBoost
- MLflow
- DVC
- Docker
- GitHub Actions

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🚀 Getting Started

### Prerequisites

- Python >= 3.10
- Git, DVC, Docker
- AWS CLI (for DVC remote)

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your_username/nsw-house-price-prediction.git
   cd nsw-house-price-prediction
