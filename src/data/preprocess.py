import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataPreprocessor:
    def __init__(self, raw_data_dir='data/raw/archive', processed_data_dir='data/processed'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.ratings_df = None
        self.movies_df = None
        self.merged_df = None
        self.user_item_matrix = None
        self.content_similarity_matrix = None
        
    def load_data(self, ratings_sample_size=None):
        """Load ratings and movies data from the new dataset structure"""
        print("Loading data from new dataset structure...")
        
        ratings_path = os.path.join(self.raw_data_dir, 'Dataset.csv')
        if ratings_sample_size:
            self.ratings_df = pd.read_csv(ratings_path, nrows=ratings_sample_size)
        else:
            self.ratings_df = pd.read_csv(ratings_path)
            
        movies_path = os.path.join(self.raw_data_dir, 'Movie_Id_Titles.csv')
        self.movies_df = pd.read_csv(movies_path)
        
        self.ratings_df = self.ratings_df.rename(columns={
            'user_id': 'userId',
            'item_id': 'movieId'
        })
        
        self.movies_df = self.movies_df.rename(columns={
            'item_id': 'movieId'
        })
        
        print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies")
        print(f"Ratings columns: {list(self.ratings_df.columns)}")
        print(f"Movies columns: {list(self.movies_df.columns)}")
        
    def merge_data(self):
        """Merge ratings and movies data"""
        print("Merging datasets...")
        self.merged_df = pd.merge(self.ratings_df, self.movies_df, on='movieId', how='inner')
        print(f"Merged dataset shape: {self.merged_df.shape}")
        
    def create_user_item_matrix(self):
        """Create user-item interaction matrix for collaborative filtering"""
        print("Creating user-item matrix...")
        
        ratings_agg = self.ratings_df.groupby(['userId', 'movieId'])['rating'].mean().reset_index()
        
        self.user_item_matrix = ratings_agg.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        )
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        sparsity = (1 - self.user_item_matrix.notna().sum().sum() / 
                   (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100
        print(f"Matrix sparsity: {sparsity:.2f}%")
        
    def extract_content_features(self):
        """Extract content-based features from titles"""
        print("Extracting content features...")
        
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)')
        self.movies_df['clean_title'] = self.movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True)
        self.movies_df['clean_title'] = self.movies_df['clean_title'].str.replace(
            r'[^a-zA-Z0-9\s]', '', regex=True
        ).str.strip()
        
        tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        title_tfidf = tfidf_vectorizer.fit_transform(self.movies_df['clean_title'].fillna(''))
        title_features = pd.DataFrame(
            title_tfidf.toarray(),
            columns=[f'title_{col}' for col in tfidf_vectorizer.get_feature_names_out()],
            index=self.movies_df.index
        )
        
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
        self.movies_df['decade'] = (self.movies_df['year'] // 10) * 10
        decade_features = pd.get_dummies(self.movies_df['decade'], prefix='decade')
        
        content_features = pd.concat([
            title_features,
            decade_features
        ], axis=1)
        
        content_matrix = content_features.select_dtypes(include=[np.number])
        self.content_similarity_matrix = cosine_similarity(content_matrix)
        
        print(f"Content features shape: {content_features.shape}")
        print(f"Content similarity matrix shape: {self.content_similarity_matrix.shape}")
        
        return content_features
        
    def save_processed_data(self):
        """Save all processed data"""
        print("Saving processed data...")
        
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        content_features = self.extract_content_features()
        merged_with_features = pd.concat([
            self.merged_df.reset_index(drop=True),
            content_features.reset_index(drop=True)
        ], axis=1)
        
        merged_with_features.to_csv(
            os.path.join(self.processed_data_dir, 'merged_with_features.csv'),
            index=False
        )
        
        self.user_item_matrix.to_csv(
            os.path.join(self.processed_data_dir, 'user_item_matrix.csv')
        )
        
        content_sim_df = pd.DataFrame(
            self.content_similarity_matrix,
            index=self.movies_df['movieId'],
            columns=self.movies_df['movieId']
        )
        content_sim_df.to_csv(
            os.path.join(self.processed_data_dir, 'content_similarity_matrix.csv')
        )
        
        movies_with_features = pd.concat([
            self.movies_df,
            content_features
        ], axis=1)
        movies_with_features.to_csv(
            os.path.join(self.processed_data_dir, 'movies_with_features.csv'),
            index=False
        )
        
        print("All processed data saved successfully!")
        
    def run_preprocessing(self, ratings_sample_size=None):
        """Run complete preprocessing pipeline"""
        print("Starting preprocessing pipeline for new dataset...")
        
        self.load_data(ratings_sample_size)
        self.merge_data()
        self.create_user_item_matrix()
        self.save_processed_data()
        
        print("Preprocessing completed!")
        return {
            'merged_data': self.merged_df,
            'user_item_matrix': self.user_item_matrix,
            'content_similarity': self.content_similarity_matrix
        }

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    results = preprocessor.run_preprocessing(ratings_sample_size=50000)