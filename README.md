# Sentiment Analysis with BERT

## Project Overview
This project builds an end-to-end sentiment analysis pipeline using BERT. It includes:
- Data extraction
- Data processing (cleaning + tokenization)
- Model training
- Inference
- CI/testing

## Setup
```
python -m venv venv
venv\Scripts\activate   # Windows
# or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

How to run training
python -c "from src.model import train_model; train_model(['good', 'bad'], [1, 0])"

How to run inference
python src/inference.py --text "I love this movie!"
Team Roles

Prathmesh – data extraction, metrics, docs

Siddhart – data processing, inference, CI

Maharshi – model training


## 🚀 Docker Usage

### Build the image
docker build -t prathmesh210/sentiment-app .

### Run inference
docker run prathmesh210/sentiment-app --text "I love this project"

## 🔄 CI/CD Pipeline

- `test.yml` → Runs pytest on every push or PR  
- `evaluate.yml` → Evaluates model, uploads metrics  
- `build.yml` → Builds and pushes Docker image after tests pass  

## 👥 Team Roles
- Prathmesh → Dockerfile, Docker build workflow, CI/CD docs  
- Siddhart → Docker Compose, Tests workflow  
- Maharshi → Evaluation workflow, model volumes  

Perfect — below is a **READY-TO-COPY & PASTE `README.md`** for your **entire project**, written exactly to match **Part 2 requirements** (README + CI/CD + Docker + URLs + roles).

You can **copy everything from the line below and paste it directly into `README.md`**.

---


# Sentiment Analysis using BERT – End-to-End MLOps Pipeline

## Overview
This project implements an end-to-end **Sentiment Analysis pipeline** using a BERT-based model, following modern **MLOps practices**.  
The pipeline covers data loading, preprocessing, model training and evaluation, inference, containerization with Docker, and continuous integration using GitHub Actions.

The goal of the project is to demonstrate how machine learning systems can be developed, tested, evaluated, and deployed in a reproducible and automated manner.

---

## Project Architecture
The project follows this workflow:

**GitHub Repository → GitHub Actions (Tests → Evaluation → Build & Push) → Docker Hub → Docker Run / Docker Compose**

- Code is version-controlled in GitHub
- CI pipelines automatically run tests and evaluation
- Docker images are built and pushed to Docker Hub
- The application can be executed using Docker or Docker Compose

---

## Setup & Run Locally

### 1. Clone the repository
```
git clone https://github.com/prathmesh210/sentiment-analysis-bert-prathmesh-siddharth-maharshi.git
cd sentiment-analysis-bert-prathmesh-siddharth-maharshi
```

### 2. Create and activate virtual environment

```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# OR
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run tests

```
pytest -v
```

---

## Run Inference Locally (without Docker)

```
python src/inference.py --text "I love this product"
```

The output will display the predicted sentiment label and logits.

---

## Run with Docker

### Build Docker image

```
docker build -t prathmesh210/sentiment-app:latest .
```

### Run Docker container

```
docker run --rm prathmesh210/sentiment-app:latest --text "I love this product"
```

---

## Run with Docker Compose

Docker Compose simplifies running the application with predefined configuration.

### Start services

```
docker compose up --build
```

### Stop services

```
docker compose down
```

---

## CI/CD Pipeline

This project uses **GitHub Actions** for Continuous Integration and Deployment.

### 1. Tests Workflow

* Automatically runs unit tests on every push
* Ensures data loading, preprocessing, model logic, metrics, and inference work correctly

### 2. Evaluate Model Workflow

* Evaluates the trained model
* Computes evaluation metrics such as **Accuracy** and **F1-score**
* Stores metrics as workflow artifacts

### 3. Build and Push Docker Image Workflow

* Builds the Docker image
* Pushes the image to **Docker Hub**
* Ensures the application is deployable and reproducible

---

## Evaluation Metrics

The model is evaluated using:

* **Accuracy**: Measures overall correctness of predictions
* **F1-score**: Balances precision and recall, suitable for binary sentiment classification

These metrics are computed during the evaluation workflow and help validate model performance before deployment.

---

## Team Roles

* **Prathmesh**

  * Dockerfile creation
  * Docker build & push workflow
  * README documentation

* **Siddharth**

  * Docker Compose configuration
  * Tests workflow
  * Global architecture diagram

* **Maharshi**

  * Volume handling
  * Evaluation workflow
  * Metric explanation (Accuracy / F1)

---

## Repository & Docker Image URLs

* **GitHub Repository:**
  [https://github.com/prathmesh210/sentiment-analysis-bert-prathmesh-siddharth-maharshi](https://github.com/prathmesh210/sentiment-analysis-bert-prathmesh-siddharth-maharshi)

* **Docker Hub Image:**
  [https://hub.docker.com/r/prathmesh210/sentiment-app](https://hub.docker.com/r/prathmesh210/sentiment-app)

---

## Conclusion

This project demonstrates a complete MLOps lifecycle for a machine learning application, from development and testing to automated evaluation and containerized deployment.
It highlights best practices for reproducibility, automation, and collaboration in real-world ML systems.


---

