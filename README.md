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
