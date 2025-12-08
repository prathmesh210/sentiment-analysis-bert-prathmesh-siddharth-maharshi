# Sentiment Analysis with BERT

## Project Overview
This project builds an end-to-end sentiment analysis pipeline using BERT. It includes:
- Data extraction
- Data processing (cleaning + tokenization)
- Model training
- Inference
- CI/testing

## Setup
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt

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

