# 🎬 Hybrid Movie Recommendation System

A comprehensive movie recommendation system that combines **Collaborative Filtering (70%)** and **Content-Based Filtering (30%)** to provide personalized movie recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a hybrid recommendation system that leverages both collaborative filtering and content-based filtering to provide accurate movie recommendations. The system uses:

- **70% Collaborative Filtering**: Uses SVD (Singular Value Decomposition) to find user similarities
- **30% Content-Based Filtering**: Uses movie titles, years, and TF-IDF features

## ✨ Features

- 🔄 **Hybrid Recommendation Engine** (70% CF + 30% CB)
- 📊 **Comprehensive Data Preprocessing** pipeline
- 🎯 **Personalized Recommendations** for each user
- 📈 **Model Evaluation** with RMSE and MAE metrics
- 💾 **Model Persistence** for trained models
- 🚀 **Memory-Efficient** processing for large datasets
- 📋 **Modular Architecture** for easy extension

## 📁 Project Structure

```
Recomendation system/
├── data/
│   ├── raw/
│   │   └── archive/
│   │       ├── Dataset.csv          # User ratings (100,004 ratings)
│   │       └── Movie_Id_Titles.csv  # Movie metadata (1,683 movies)
│   └── processed/
│       ├── merged_with_features.csv      # Complete dataset with features
│       ├── user_item_matrix.csv          # User-item interaction matrix
│       ├── content_similarity_matrix.csv # Content similarity matrix
│       └── movies_with_features.csv      # Movies with extracted features
├── models/
│   ├── saved/                          # Directory for saved models
│   ├── hybrid.py                       # Hybrid recommendation system
│   ├── example_usage.py                # Full usage examples
│   └── simple_test.py                  # Simple test script
├── src/
│   ├── data/
│   │   └── preprocess.py              # Data preprocessing pipeline
│   ├── evaluation/                    # Model evaluation tools
│   └── models/                        # Model implementations
├── notebooks/
│   └── 03_model_experiments.ipynb    # Jupyter notebook for experiments
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Recomendation system"
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional dependencies:**
   ```bash
   pip install "numpy<2"  # For compatibility with surprise library
   ```

## 🚀 Usage

### 1. Data Preprocessing

Run the preprocessing pipeline to prepare your data:

```bash
python3 src/data/preprocess.py
```

This will:
- Load and merge ratings and movie data
- Create user-item interaction matrix
- Extract content features (TF-IDF, year, decade)
- Save processed data to `data/processed/`

### 2. Train Hybrid Model

Train the hybrid recommendation system:

```bash
python3 src/models/hybrid.py
```

This will:
- Train collaborative filtering model (SVD)
- Initialize content-based filtering
- Generate sample recommendations
- Display model performance metrics

### 3. Using the Model

```python
from src.models.hybrid import HybridRecommender, load_processed_data

# Load processed data
ratings_df, movies_df, user_item_matrix, content_similarity_matrix = load_processed_data()

# Initialize and train hybrid recommender
hybrid_recommender = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
hybrid_recommender.fit(ratings_df, movies_df, user_item_matrix, content_similarity_matrix)

# Get recommendations for a user
recommendations = hybrid_recommender.get_recommendations(user_id=0, n_recommendations=5)
```

## 📊 Dataset

The system uses the MovieLens dataset with:

- **100,004 ratings** from 943 users
- **1,683 movies** with titles and years
- **5-point rating scale** (1-5)
- **Sparse matrix** (95.88% sparsity)

### Data Format

**Dataset.csv:**
```
user_id,item_id,rating,timestamp
0,50,5,881250949
0,172,5,881250949
...
```

**Movie_Id_Titles.csv:**
```
item_id,title
1,Toy Story (1995)
2,GoldenEye (1995)
...
```

## 🏗️ Architecture

### Data Preprocessing Pipeline

```
Raw Data → Merge → Feature Extraction → User-Item Matrix → Content Similarity Matrix
```

### Hybrid Recommendation System

```
User Input → CF Model (70%) + CB Model (30%) → Weighted Combination → Recommendations
```

### Components

1. **CollaborativeFiltering Class**
   - SVD algorithm for matrix factorization
   - Hyperparameter tuning with GridSearchCV
   - User-based recommendations

2. **ContentBasedFiltering Class**
   - TF-IDF features from movie titles
   - Year and decade features
   - Content similarity matrix

3. **HybridRecommender Class**
   - Combines CF and CB predictions
   - Configurable weights (70/30 by default)
   - Comprehensive evaluation

## 📈 Model Performance

### Current Performance Metrics

- **Collaborative Filtering RMSE**: 0.9667
- **Matrix Sparsity**: 95.88%
- **Content Features**: 108 features (TF-IDF + decade)

### Sample Recommendations

For User 0:
1. Shawshank Redemption, The (1994)
2. Close Shave, A (1995)
3. Schindler's List (1993)
4. Sling Blade (1996)
5. Raging Bull (1980)

## 🔧 API Reference

### DataPreprocessor Class

```python
preprocessor = DataPreprocessor(raw_data_dir='data/raw/archive')
results = preprocessor.run_preprocessing(ratings_sample_size=50000)
```

### HybridRecommender Class

```python
# Initialize
hybrid_recommender = HybridRecommender(cf_weight=0.7, cb_weight=0.3)

# Train
hybrid_recommender.fit(ratings_df, movies_df, user_item_matrix, content_similarity_matrix)

# Get recommendations
recommendations = hybrid_recommender.get_recommendations(user_id, n_recommendations=10)

# Predict rating
predicted_rating = hybrid_recommender.predict_rating(user_id, movie_id)

# Evaluate
evaluation_results = hybrid_recommender.evaluate(test_ratings)
```

## 🧪 Testing

### Simple Test (No Dependencies)

Test basic functionality without the surprise library:

```bash
python3 models/simple_test.py
```

This tests:
- Data loading
- Content-based filtering
- User recommendations
- Model persistence

### Full Test (With Dependencies)

Test the complete system with all dependencies:

```bash
python3 models/example_usage.py
```

**Note**: This requires NumPy < 2.0 for compatibility with the surprise library.

## 🔄 Model Persistence

Trained models can be saved to the `models/saved/` directory:

```python
import pickle

# Save model
with open('models/saved/hybrid_model.pkl', 'wb') as f:
    pickle.dump(hybrid_recommender, f)

# Load model
with open('models/saved/hybrid_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

## 🧪 Experiments

Use the Jupyter notebook for experiments:

```bash
jupyter notebook notebooks/03_model_experiments.ipynb
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MovieLens dataset for providing the movie ratings data
- Surprise library for collaborative filtering algorithms
- Scikit-learn for machine learning utilities

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Recommending! 🎬✨** 