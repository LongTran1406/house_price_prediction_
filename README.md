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

**Key Features:**
- Scraping from NSW property sites
- Data cleaning and preprocessing
- Model training (Linear Regression, XGBoost)
- MLflow experiment logging
- DVC tracking of data and models
- Docker-based deployment with images hosted on AWS ECR
- CI/CD pipeline with GitHub Actions deploying to AWS ECR and/or other cloud services


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🛠️ Built With

- Python 3.10
- scikit-learn
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

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your_username/nsw-house-price-prediction.git
   cd nsw-house-price-prediction
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure DVC remote storage (replace with your S3 bucket or preferred remote):**
   ```bash
   dvc remote add -d myremote s3://your-bucket-name
   dvc pull
   ```

4. **(Optional) Configure AWS CLI if using S3:**
   ```bash
   aws configure
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🧠 Usage

### Data Scraping
Run the scraper to collect raw NSW housing data:
```bash
python src/scraping.py
```

### Data Preprocessing
Clean and preprocess scraped data:
```bash
python src/data_cleaning.py
python src/data_cleaning2.py
```

### Model Training
Train ML models and log experiments:
```bash
python src/model_building.py
```

### Docker Deployment
Build and run the API container:
```bash
docker build -t nsw-price-api .
docker run -p 5000:5000 nsw-price-api
```

### CI/CD
GitHub Actions pipeline automates:
- Data pulling via DVC
- Model training & evaluation
- Auto deploying model on AWS 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---



<p align="right">(<a href="#readme-top">back to top</a>)</p>

---



## 📬 Contact

**The Long Tran**
- 📧 thelong@example.com
- 🔗 [LinkedIn](https://linkedin.com/in/your-profile)
- 💻 [GitHub Project](https://github.com/your_username/nsw-house-price-prediction)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/your_username/nsw-house-price-prediction.svg?style=for-the-badge
[contributors-url]: https://github.com/your_username/nsw-house-price-prediction/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/your_username/nsw-house-price-prediction.svg?style=for-the-badge
[forks-url]: https://github.com/your_username/nsw-house-price-prediction/network/members
[stars-shield]: https://img.shields.io/github/stars/your_username/nsw-house-price-prediction.svg?style=for-the-badge
[stars-url]: https://github.com/your_username/nsw-house-price-prediction/stargazers
[issues-shield]: https://img.shields.io/github/issues/your_username/nsw-house-price-prediction.svg?style=for-the-badge
[issues-url]: https://github.com/your_username/nsw-house-price-prediction/issues
[license-shield]: https://img.shields.io/github/license/your_username/nsw-house-price-prediction.svg?style=for-the-badge
[license-url]: https://github.com/your_username/nsw-house-price-prediction/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/your-profile
