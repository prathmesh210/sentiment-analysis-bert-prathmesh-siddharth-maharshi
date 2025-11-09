## 🧩 Pipeline & Workflow 

To ensure smooth team coordination and progress tracking, a Trello board was created titled:
**"Sentiment Analysis Project – Prathmesh, Siddharth & Maharshi"**.

### 🧱 Workflow
The board had four lists:
- **To Do** – initial feature tasks before development.
- **In Progress** – tasks being actively developed.
- **In Review** – tasks awaiting peer review and testing.
- **Done** – completed, tested, and merged features.

### 🕹 Example Task Flow
1. **Feature:** `data_extraction.py`  
   → moved from *To Do* → *In Progress* → *In Review* → *Done* after Prathmesh completed and reviewed it.  
2. **Feature:** `data_processing.py`  
   → handled by Siddhart → reviewed by Maharshi.  
3. **Feature:** `model.py`  
   → implemented by Maharshi → reviewed by Prathmesh.

### 🖼 Evidence
Below is a screenshot of the final Trello board showing cards in all phases:

![Trello Board](docs/trello_screenshot.png)


## 🤖 Model & Training (by Maharshi)

### 🧩 Model Architecture
The project uses the **`DistilBERT-base-uncased`** model from the Hugging Face Transformers library, 
a lighter and faster variant of BERT optimized for classification tasks.

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
