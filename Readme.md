# Multilingual Support Ticket Classification System 🎯

An end-to-end machine learning pipeline for automatically classifying multilingual customer support tickets into appropriate departments using transformer-based models.

## 🎯 Project Overview

This system automatically routes customer support tickets to appropriate departments based on their content, supporting multiple languages. It uses XLM-RoBERTa for multilingual text classification and provides a REST API for predictions.

### Key Features
- 🔄 Complete ML pipeline from data preparation to production
- 🌐 Multilingual support using XLM-RoBERTa
- 🚀 FastAPI endpoint for real-time predictions
- 🐳 Docker support with multi-stage builds
- 📊 Comprehensive data preprocessing pipeline

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker (optional)
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#authentication) (for downloading dataset)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/support-ticket-classifier.git
cd support-ticket-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
```bash
# Option 1: Using Kaggle API
kaggle datasets download tobiasbueck/multilingual-customer-support-tickets -p dataset/
unzip dataset/multilingual-customer-support-tickets.zip -d dataset/

# Option 2: Manual download
# Download from: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/data
# Place the CSV file in the 'dataset' directory
```

### Running with Docker (Recommended)

1. Build and train the model:
```bash
docker build  -t your-user-name/mbl_assignment .
```



2. Run the image service:
```bash
docker run -p 8000:8000 your-user-name/mbl_assignment
```

2. Push image to docker hub:
```bash
docker push your-user-name/mbl_assignment:latest
```

### Running Locally

1. Train the model:
```bash
python trainer.py
```

2. Start the API server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Documentation

Once the server is running, access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example API Request

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "text": "I need help resetting my password for the application."
}

response = requests.post(url, json=data)
print(response.json())
```

Example response:
```json
{
    "label": "IT Support",
    "confidence_score": 0.92
}
```

## 🏗️ Project Structure

```
support-ticket-classifier/
├── dataset/                  # Dataset directory
├── model/                    # Saved model files
├── data_pipeline.py         # Data preprocessing pipeline
├── trainer.py               # Model training script
├── predict.py               # Prediction service
├── main.py                  # FastAPI application
├── helpdesk_analysis.ipynb  # Data analysis on CST dataset
├── Dockerfile               # Multi-stage Docker build
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

Key configurations can be modified in the respective files:

### Model Training (`trainer.py`)
```python
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    ...
)
```

### API Server (`main.py`)
```python
app = FastAPI()
# Configure CORS, rate limits, etc. here
```






