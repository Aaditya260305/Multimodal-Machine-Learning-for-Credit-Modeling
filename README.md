# Multimodal Machine Learning for Credit Modeling

This repository implements a multimodal machine learning approach for credit rating prediction, combining traditional financial data with textual analysis of SEC filings. The project replicates and extends the methodology from the paper on multimodal machine learning for credit modeling.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Training & Evaluation](#model-training--evaluation)
- [Web Dashboard](#web-dashboard)
- [Results](#results)
- [Configuration](#configuration)

## 🎯 Overview

Credit ratings are traditionally generated using models that rely on financial statement data and market data. This project demonstrates how incorporating long-form text from SEC filings using multimodal machine learning can generate more accurate rating predictions through stack ensembling and bagging techniques.

**Base Paper**: [Multimodal Machine Learning for Credit Modeling](https://assets.amazon.science/25/fa/882cb69c4f8983f6c4b287da7a6f/multimodal-machine-learning-for-credit-modeling.pdf)

## 📁 Project Structure

```
├── # Dataset files
│   ├── corporate_rating.csv        # Initial corporate ratings
│   ├── final_dataset.csv          # Final processed dataset
│   ├── binaryclass_with_ticker.csv # Binary classification data
│   ├── multiclass_with_ticker.csv  # Multi-class classification data
│   └── Loughran-McDonald_MasterDictionary_1993-2021.csv
│
├── # Jupyter notebooks
│   ├── Data_extraction_sec_code.ipynb    # SEC data extraction
│   ├── sentiment_analysis.ipynb          # Sentiment analysis using FinBERT
│   ├── nlp_scores.ipynb                  # NLP feature extraction
│   ├── binaryclass_classfication.ipynb   # Binary classification models
│   ├── multiclass_classfication.ipynb    # Multi-class classification
│   ├── *_deep_learning.ipynb             # Deep learning approaches
│   └── binary_cf.ipynb                   # Confusion matrix analysis
│
├── # Source code
│   └── app.py                    # Streamlit dashboard
│
├── # Model outputs
│   ├── decision_plot.png         # Decision boundary plots
│   └── comparative_analysis.csv  # Model comparison results
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for deep learning models)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aaditya260305/Multimodal-Machine-Learning-for-Credit-Modeling
cd Multimodal-Machine-Learning-for-Credit-Modeling
```

2. **Create virtual environment**
```bash
python -m venv credit_modeling_env
source credit_modeling_env/bin/activate  # On Windows: credit_modeling_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Key Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
tensorflow>=2.10.0
torch>=1.12.0
transformers>=4.20.0
catboost>=1.0.6
xgboost>=1.6.0
lightgbm>=3.3.0
streamlit>=1.15.0
plotly>=5.10.0
seaborn>=0.11.0
matplotlib>=3.5.0
sec-edgar-downloader>=5.0.0
textstat>=0.7.0
beautifulsoup4>=4.11.0
```

## 📊 Data Preparation

### Step 1: SEC Data Extraction
Extract Management Discussion & Analysis (MD&A) sections from SEC filings:

```bash
jupyter notebook Data_extraction_sec_code.ipynb
```

**Configuration:**
- Set your email for SEC API access
- Modify date ranges as needed
- Companies are identified by ticker symbols

### Step 2: Sentiment Analysis
Generate sentiment scores using FinBERT and Loughran-McDonald dictionary:

```bash
# FinBERT sentiment analysis
jupyter notebook sentiment_analysis.ipynb

# Traditional NLP features
jupyter notebook nlp_scores.ipynb
```

**Fixed Seeds:**
```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

## 🤖 Model Training & Evaluation

### Binary Classification
Train models for investment grade vs non-investment grade classification:

```bash
jupyter notebook binaryclass_classfication.ipynb
```

**Available Models:**
- Logistic Regression
- Random Forest (Gini & Entropy)
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- K-Nearest Neighbors
- Stack Ensemble

### Multi-Class Classification
Train models for detailed rating prediction:

```bash
jupyter notebook multiclass_classfication.ipynb
```

### Deep Learning Approaches
Advanced neural network models:

```bash
jupyter notebook binaryclass_classfication_deep_learning.ipynb
jupyter notebook multiclass_classfication_deep_learning.ipynb
```

**Neural Architectures:**
- Artificial Neural Networks (ANN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Convolutional Neural Networks (CNN)
- Bidirectional LSTM

### Model Configuration
```python
# Standard model parameters
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'max_iter': 1000
}

# Deep learning parameters
DL_CONFIG = {
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 0.001,
    'validation_split': 0.2
}
```

## 🚀 Running the Complete Pipeline

### Full Pipeline Execution
```bash
# 1. Extract SEC data
jupyter notebook Data_extraction_sec_code.ipynb

# 2. Generate sentiment features
jupyter notebook sentiment_analysis.ipynb
jupyter notebook nlp_scores.ipynb

# 3. Train binary classification models
jupyter notebook binaryclass_classfication.ipynb

# 4. Train multi-class models
jupyter notebook multiclass_classfication.ipynb

# 5. Deep learning experiments
jupyter notebook binaryclass_classfication_deep_learning.ipynb

# 6. Launch dashboard
streamlit run app.py
```

## 📈 Web Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

**Dashboard Features:**
- Dataset overview and statistics
- Financial ratios exploration
- Text sentiment analysis visualization
- Model performance comparison
- Interactive plotting with Plotly



## 🔬 Results & Analysis

### Model Performance Comparison
The project evaluates multiple approaches:

1. **Traditional Models:** Logistic Regression, Random Forest, SVM
2. **Gradient Boosting:** XGBoost, LightGBM, CatBoost
3. **Ensemble Methods:** Stack Ensemble, Voting Classifiers
4. **Deep Learning:** ANN, LSTM, CNN, Bidirectional LSTM

### Key Findings
- Multimodal approaches outperform traditional financial-only models
- Stack ensemble methods provide the best performance
- Text features add significant predictive power
- Deep learning models show competitive performance

## ⚙️ Configuration

### Random Seeds
All experiments use fixed seeds for reproducibility:
```python
RANDOM_SEED = 42
```

### Data Processing Configuration
```python
DATA_CONFIG = {
    'standardize_features': True,
    'handle_missing': 'drop',
    'min_text_length': 60,
    'rating_mapping': {
        'binary': {'IG': 1, 'NIG': 0},
        'multiclass': {'AAA': 5, 'AA': 4, 'A': 3, 'BBB': 2, 'BB': 1, 'B': 0}
    }
}
```

### Model Training Configuration
```python
TRAINING_CONFIG = {
    'cross_validation': 5,
    'test_size': 0.2,
    'stratify': True,
    'shuffle': True,
    'random_state': 42
}
```


## 📚 References

- [Multimodal Machine Learning for Credit Modeling - Amazon Science](https://assets.amazon.science/25/fa/882cb69c4f8983f6c4b287da7a6f/multimodal-machine-learning-for-credit-modeling.pdf)
- [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

## 🆘 Troubleshooting

### Common Issues

1. **SEC API Rate Limits**
   - Add delays between requests
   - Use proper headers with email identification

2. **Memory Issues with Deep Learning**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

3. **Missing Dependencies**
   - Ensure CUDA is installed for GPU support
   - Update pip and setuptools before installation

4. **Data Loading Errors**
   - Check file paths in notebooks
   - Ensure all CSV files are present
   - Verify data format consistency
