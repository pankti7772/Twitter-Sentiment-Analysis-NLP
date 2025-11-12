# Twitter Sentiment Analysis & Toxicity Detection

A reproducible machine learning pipeline for Twitter sentiment classification and toxicity detection using NLP techniques.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline for analyzing Twitter data to classify sentiment (Positive, Negative, Neutral) and detect toxic content. The pipeline includes data ingestion, transformation, model training, and prediction capabilities.

## 🏗️ Project Structure

```
sentiment_analysis/
├── src/
│   ├── app.py                    # Main application entry point
│   ├── components/
│   │   ├── data_ingestion.py    # Data loading and preprocessing
│   │   ├── data_transformation.py # Feature engineering and transformations
│   │   └── model_training.py    # Model training and evaluation
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Training pipeline orchestration
│   │   └── prediction_pipeline.py # Prediction pipeline
│   ├── requirements.txt          # Python dependencies
│   └── setup.py                 # Package setup configuration
├── notebook/
│   └── data/
│       └── twitter_training_cleaned.csv  # Training dataset
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pankti7772/Sentiment_Analysis_Using_NLP.git
cd sentiment_analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r src/requirements.txt
```

Or install the package in development mode:
```bash
pip install -e .
```

## 📊 Dataset

The project uses a cleaned Twitter dataset (`twitter_training_cleaned.csv`) containing:
- `id`: Unique identifier
- `entity`: Entity/topic mentioned
- `sentiment`: Sentiment label (Positive, Negative, Neutral)
- `tweet`: Tweet text content

## 🔧 Usage

### Training the Model

Run the training pipeline to train the sentiment analysis model:

```bash
python src/pipeline/train_pipeline.py
```

### Making Predictions

Use the prediction pipeline to analyze new text:

```bash
python src/pipeline/prediction_pipeline.py
```

### Running the Application

```bash
python src/app.py
```

## 🛠️ Development

### Project Components

- **Data Ingestion**: Handles loading and initial preprocessing of Twitter data
- **Data Transformation**: Applies text preprocessing, feature engineering, and vectorization
- **Model Training**: Trains and evaluates machine learning models for sentiment classification
- **Pipelines**: Orchestrates the end-to-end workflow for training and prediction

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Pankti Singh**
- Email: panktisingh16@gmail.com
- GitHub: [@pankti7772](https://github.com/pankti7772)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/pankti7772/Sentiment_Analysis_Using_NLP/issues).

## 📚 References

- Twitter Sentiment Analysis Dataset
- Natural Language Processing techniques
- Machine Learning best practices

---

**Note**: This is a work in progress. Some components may be under development.

