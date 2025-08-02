import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

import pickle
import os

class CollaborativeFiltering:
    """Collaborative Filtering component using SVD"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.model = None
        self.trainset = None
        
    def prepare_data(self, ratings_df):
        """Prepare data for SVD model"""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        
        trainset, temp_set = train_test_split(data, test_size=0.3, random_state=42)
        valset_size = int(len(temp_set) * 0.5)
        valset = temp_set[:valset_size]
        testset = temp_set[valset_size:]
        
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        
        return trainset, valset, testset
        
    def train_model(self, ratings_df, tune_hyperparameters=True):
        """Train SVD model with optional hyperparameter tuning"""
        print("Training Collaborative Filtering model...")
        
        trainset, valset, testset = self.prepare_data(ratings_df)
        
        if tune_hyperparameters:
            param_grid = {
                'n_factors': [10, 20, 50],
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.01, 0.02, 0.05]
            }
            
            gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
            gs.fit(Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], Reader(rating_scale=(1, 5))))
            
            self.model = gs.best_estimator['rmse']
            print(f"Best parameters: {gs.best_params['rmse']}")
        else:
            self.model = SVD(n_factors=20, lr_all=0.005, reg_all=0.02)
            
        self.model.fit(trainset)
        
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        print(f"CF Model RMSE: {rmse:.4f}")
        
        return rmse
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
        
    def get_user_recommendations(self, user_id, movies_df, n_recommendations=10):
        """Get top recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        all_movies = movies_df['movieId'].unique()
        
        predictions = []
        for movie_id in all_movies:
            pred_rating = self.predict_rating(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = [movie_id for movie_id, _ in predictions[:n_recommendations]]
        
        return top_movies

class ContentBasedFiltering:
    """Content-Based Filtering component using title and year features"""
    
    def __init__(self, content_similarity_matrix, movies_df):
        self.content_similarity_matrix = content_similarity_matrix
        self.movies_df = movies_df
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}
        
    def get_similar_movies(self, movie_id, n_recommendations=10):
        """Get similar movies based on content features"""
        if movie_id not in self.movie_id_to_idx:
            return []
            
        movie_idx = self.movie_id_to_idx[movie_id]
        similar_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
        
        similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        similar_movies = [self.movies_df.iloc[idx]['movieId'] for idx, _ in similar_scores]
        return similar_movies
        
    def get_user_content_recommendations(self, user_ratings, n_recommendations=10):
        """Get content-based recommendations for a user based on their rated movies"""
        if len(user_ratings) == 0:
            return []
            
        movie_scores = {}
        
        for movie_id in self.movies_df['movieId']:
            if movie_id in user_ratings:
                continue
                
            total_similarity = 0
            count = 0
            
            for rated_movie_id, rating in user_ratings.items():
                if rated_movie_id in self.movie_id_to_idx and movie_id in self.movie_id_to_idx:
                    rated_idx = self.movie_id_to_idx[rated_movie_id]
                    movie_idx = self.movie_id_to_idx[movie_id]
                    
                    similarity = self.content_similarity_matrix[rated_idx][movie_idx]
                    total_similarity += similarity * rating
                    count += 1
                    
            if count > 0:
                movie_scores[movie_id] = total_similarity / count
                
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_movies[:n_recommendations]]

class HybridRecommender:
    """Hybrid Recommendation System combining CF (70%) and Content-Based (30%)"""
    
    def __init__(self, cf_weight=0.7, cb_weight=0.3):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = None
        self.cb_model = None
        self.movies_df = None
        self.user_item_matrix = None
        
    def fit(self, ratings_df, movies_df, user_item_matrix, content_similarity_matrix):
        """Train both CF and Content-Based models"""
        print("Training Hybrid Recommendation System...")
        
        self.movies_df = movies_df
        self.user_item_matrix = user_item_matrix
        
        self.cf_model = CollaborativeFiltering(user_item_matrix)
        cf_rmse = self.cf_model.train_model(ratings_df, tune_hyperparameters=False)
        
        self.cb_model = ContentBasedFiltering(content_similarity_matrix, movies_df)
        
        print(f"Hybrid model trained successfully!")
        print(f"CF Weight: {self.cf_weight}, CB Weight: {self.cb_weight}")
        
    def save_model(self, filepath='models/saved/hybrid_model.pkl'):
        """Save the trained hybrid model"""
        if self.cf_model is None or self.cb_model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath='models/saved/hybrid_model.pkl'):
        """Load a trained hybrid model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

    def get_recommendations(self, user_id, n_recommendations=10):
        """Get hybrid recommendations for a user"""
        if self.cf_model is None or self.cb_model is None:
            raise ValueError("Models not trained. Call fit() first.")
            
        cf_recommendations = self.cf_model.get_user_recommendations(
            user_id, self.movies_df, n_recommendations
        )
        
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id].dropna().to_dict()
        else:
            user_ratings = {}
            
        cb_recommendations = self.cb_model.get_user_content_recommendations(
            user_ratings, n_recommendations
        )
        
        movie_scores = {}
        
        for i, movie_id in enumerate(cf_recommendations):
            score = self.cf_weight * (1.0 - i / len(cf_recommendations))
            movie_scores[movie_id] = movie_scores.get(movie_id, 0) + score
            
        for i, movie_id in enumerate(cb_recommendations):
            score = self.cb_weight * (1.0 - i / len(cb_recommendations))
            movie_scores[movie_id] = movie_scores.get(movie_id, 0) + score
            
        sorted_recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        final_recommendations = [movie_id for movie_id, _ in sorted_recommendations[:n_recommendations]]
        
        return final_recommendations
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating using hybrid approach"""
        if self.cf_model is None:
            raise ValueError("Models not trained. Call fit() first.")
            
        cf_prediction = self.cf_model.predict_rating(user_id, movie_id)
        
        cb_adjustment = 0
        
        hybrid_prediction = (self.cf_weight * cf_prediction + 
                           self.cb_weight * (cf_prediction + cb_adjustment))
        
        return hybrid_prediction
        
    def evaluate(self, test_ratings):
        """Evaluate hybrid model performance"""
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            try:
                predicted_rating = self.predict_rating(user_id, movie_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except:
                continue
                
        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            return {'RMSE': rmse, 'MAE': mae}
        else:
            return {'RMSE': None, 'MAE': None}

def load_processed_data(processed_data_dir='data/processed'):
    """Load all processed data"""
    print("Loading processed data...")
    
    ratings_df = pd.read_csv(f'{processed_data_dir}/merged_with_features.csv')
    movies_df = pd.read_csv(f'{processed_data_dir}/movies_with_features.csv')
    user_item_matrix = pd.read_csv(f'{processed_data_dir}/user_item_matrix.csv', index_col=0)
    content_similarity_matrix = pd.read_csv(
        f'{processed_data_dir}/content_similarity_matrix.csv', 
        index_col=0
    ).values
    
    return ratings_df, movies_df, user_item_matrix, content_similarity_matrix

if __name__ == "__main__":
    ratings_df, movies_df, user_item_matrix, content_similarity_matrix = load_processed_data()
    
    hybrid_recommender = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
    hybrid_recommender.fit(ratings_df, movies_df, user_item_matrix, content_similarity_matrix)
    
    hybrid_recommender.save_model()
    
    test_user = user_item_matrix.index[0]
    recommendations = hybrid_recommender.get_recommendations(test_user, n_recommendations=5)
    
    print(f"\nTop 5 hybrid recommendations for user {test_user}:")
    for i, movie_id in enumerate(recommendations, 1):
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            print(f"{i}. {title}")
            
    print(f"\nDemonstrating model loading...")
    loaded_model = HybridRecommender.load_model()
    loaded_recommendations = loaded_model.get_recommendations(test_user, n_recommendations=3)
    
    print(f"Top 3 recommendations from loaded model for user {test_user}:")
    for i, movie_id in enumerate(loaded_recommendations, 1):
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            print(f"{i}. {title}")