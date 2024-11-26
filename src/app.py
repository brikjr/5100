import streamlit as st
import pandas as pd
import torch
from model import BookRecommender, Config
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from typing import List, Dict

class BookRecommenderApp:
    def __init__(self, model_path: str = 'models/book_recommender'):
        self.model_path = model_path
        self.recommender = None
        
    def load_model(self):
        if self.recommender is None:
            with st.spinner('Loading model... Please wait.'):
                self.recommender = BookRecommender(self.model_path)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for any text using the model."""
        encoding = self.recommender.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=Config.MAX_LENGTH,
            return_tensors='pt'
        ).to(self.recommender.device)
        
        with torch.no_grad():
            embedding = self.recommender.embedding_model(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            
        return embedding.cpu().numpy()[0]
    
    def recommend_from_user_list(
        self,
        user_books: List[str],
        mood_description: str,
        n_recommendations: int = 3
    ) -> List[Dict]:
        """Recommend books from user's list based on mood description."""
        # Get mood embedding
        mood_embedding = self.get_embedding(mood_description)
        
        # Get embeddings for each book in user's list
        book_embeddings = []
        for book in user_books:
            embedding = self.get_embedding(book)
            book_embeddings.append(embedding)
        
        # Convert to numpy array
        book_embeddings = np.array(book_embeddings)
        
        # Calculate similarities between mood and each book
        similarities = cosine_similarity(
            mood_embedding.reshape(1, -1),
            book_embeddings
        )[0]
        
        # Create recommendations list
        recommendations = []
        top_indices = similarities.argsort()[-n_recommendations:][::-1]
        
        for idx in top_indices:
            recommendations.append({
                'title': user_books[idx],
                'similarity_score': similarities[idx]
            })
            
        return recommendations
    
    def run(self):
        st.title("📚 Book Recommender System")
        
        try:
            self.load_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        st.write("Enter your book list and mood description to get recommendations.")
        
        # Text area for book list input
        books_input = st.text_area(
            "Enter your book list (one book per line):",
            help="Enter the titles of books you want recommendations from"
        )
        
        # Input for mood/preference
        user_mood = st.text_area(
            "Describe what kind of book you're looking for:",
            help="Describe your reading mood or the type of story you're interested in"
        )
        
        n_recommendations = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=5,
            value=3,
            help="How many books do you want recommended?"
        )
        
        if st.button("Get Recommendations"):
            if not (books_input and user_mood):
                st.warning("Please enter both your book list and mood description.")
                return
                
            with st.spinner('Analyzing your books...'):
                try:
                    # Process book list
                    book_list = [book.strip() for book in books_input.split('\n') if book.strip()]
                    
                    if len(book_list) < n_recommendations:
                        st.warning(f"Please enter at least {n_recommendations} books to get {n_recommendations} recommendations.")
                        return
                    
                    # Get recommendations from user's list
                    recommendations = self.recommend_from_user_list(
                        book_list,
                        user_mood,
                        n_recommendations
                    )
                    
                    # Display input books
                    st.subheader("Your Input Books:")
                    for book in book_list:
                        st.write(f"- {book}")
                    
                    st.subheader(f"Top {n_recommendations} Recommended Books From Your List:")
                    st.write("Based on your mood:", user_mood)
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec['title']}")
                        st.write(f"   Similarity to your mood: {rec['similarity_score']:.4f}")
                        
                    # Add explanation
                    st.info("""
                    These recommendations are chosen from your provided book list based on how well 
                    each book matches your described mood or preferences. The similarity score shows 
                    how closely each book aligns with your described mood.
                    """)
                
                except Exception as e:
                    st.error(f"Error processing recommendations: {str(e)}")
                    st.error(f"Traceback: {traceback.format_exc()}")

def main():
    st.set_page_config(
        page_title="Book Recommender",
        page_icon="📚",
        layout="centered"
    )
    
    # Add CSS
    st.markdown("""
        <style>
        .stTextArea textarea {
            font-size: 16px;
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stExpander {
            background-color: #2D2D2D;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .stButton button {
            width: 100%;
            margin-top: 20px;
            margin-bottom: 20px;
            background-color: #FF4B4B;
            color: white;
        }
        .stInfo {
            background-color: #1E2F45;
            color: #9DC8FF;
        }
        </style>
    """, unsafe_allow_html=True)
    
    app = BookRecommenderApp()
    app.run()

if __name__ == "__main__":
    main()
