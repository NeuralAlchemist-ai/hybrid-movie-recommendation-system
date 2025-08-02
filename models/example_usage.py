#!/usr/bin/env python3
"""
Example usage of the Hybrid Movie Recommendation System

This script demonstrates how to:
1. Load a trained hybrid model
2. Get recommendations for users
3. Predict ratings for user-movie pairs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.hybrid import HybridRecommender, CollaborativeFiltering, ContentBasedFiltering, load_processed_data

def main():
    """Main function demonstrating model usage"""
    
    print("🎬 Hybrid Movie Recommendation System - Example Usage")
    print("=" * 60)
    
    try:
        print("Loading trained hybrid model...")
        hybrid_model = HybridRecommender.load_model('models/saved/hybrid_model.pkl')
        
        print("Loading processed data...")
        ratings_df, movies_df, user_item_matrix, content_similarity_matrix = load_processed_data()
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Dataset info: {len(ratings_df)} ratings, {len(movies_df)} movies")
        print()
        
        test_users = [0, 1, 2, 5, 10]
        
        for user_id in test_users:
            if user_id in user_item_matrix.index:
                print(f"🎯 Recommendations for User {user_id}:")
                recommendations = hybrid_model.get_recommendations(user_id, n_recommendations=5)
                
                for i, movie_id in enumerate(recommendations, 1):
                    movie_info = movies_df[movies_df['movieId'] == movie_id]
                    if not movie_info.empty:
                        title = movie_info.iloc[0]['title']
                        print(f"   {i}. {title}")
                print()
        
        print("📊 Rating Predictions:")
        test_pairs = [
            (0, 1),
            (0, 50),
            (1, 100),
        ]
        
        for user_id, movie_id in test_pairs:
            try:
                predicted_rating = hybrid_model.predict_rating(user_id, movie_id)
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                title = movie_info.iloc[0]['title'] if not movie_info.empty else f"Movie {movie_id}"
                print(f"   User {user_id} → {title}: {predicted_rating:.2f}/5.0")
            except Exception as e:
                print(f"   User {user_id} → Movie {movie_id}: Error - {e}")
        print()
        
        print("🔍 Similar Movies:")
        test_movies = [1, 50, 100]
        
        for movie_id in test_movies:
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                similar_movies = hybrid_model.cb_model.get_similar_movies(movie_id, n_recommendations=3)
                
                print(f"   Movies similar to '{title}':")
                for i, similar_movie_id in enumerate(similar_movies, 1):
                    similar_movie_info = movies_df[movies_df['movieId'] == similar_movie_id]
                    if not similar_movie_info.empty:
                        similar_title = similar_movie_info.iloc[0]['title']
                        print(f"     {i}. {similar_title}")
                print()
        
        print("✅ Example usage completed successfully!")
        
    except FileNotFoundError:
        print("❌ Model file not found. Please run the training script first:")
        print("   python src/models/hybrid.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure the model is trained and data is processed.")

if __name__ == "__main__":
    main() 