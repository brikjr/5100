import streamlit as st
import pandas as pd
from model import BookRecommender, Config
import torch
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import traceback

def calculate_similarity(a: str, b: str) -> float:
    """Calculate string similarity between two titles."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_matching_books(book_title: str, df: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
    """Find books with similar titles in the database."""
    matches = []
    for _, row in df.iterrows():
        similarity = calculate_similarity(book_title, row['Book-Title'])
        if similarity >= threshold:
            matches.append({
                'title': row['Book-Title'],
                'description': row['description'],
                'similarity': similarity
            })
    return matches

class BookRecommenderApp:
    def __init__(self, model_path: str = 'models/book_recommender'):
        self.model_path = model_path
        self.recommender = None
        
    def load_model(self):
        if self.recommender is None:
            with st.spinner('Loading model... Please wait.'):
                self.recommender = BookRecommender(self.model_path)
    
    def find_best_matches(self, book_list: List[str]) -> List[Dict]:
        """Find best matching books for the provided list."""
        matched_books = []
        not_found_books = []
        
        for book in book_list:
            matches = find_matching_books(book, self.recommender.books_df)
            if matches:
                # Sort matches by similarity and take the best match
                best_match = sorted(matches, key=lambda x: x['similarity'], reverse=True)[0]
                matched_books.append(best_match)
            else:
                not_found_books.append(book)
        
        return matched_books, not_found_books
    
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
            help="Enter the titles of books, one per line"
        )
        
        # Input for mood/preference
        user_mood = st.text_area(
            "Describe what kind of book you're looking for:",
            help="Describe your reading mood or the type of story you're interested in"
        )
        
        # Add similarity threshold slider
        similarity_threshold = st.slider(
            "Title matching threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Lower value will match more books but might be less accurate"
        )
        
        if st.button("Get Recommendations"):
            if not (books_input and user_mood):
                st.warning("Please enter both your book list and mood description.")
                return
                
            with st.spinner('Analyzing your preferences...'):
                try:
                    # Process book list
                    book_list = [book.strip() for book in books_input.split('\n') if book.strip()]
                    
                    # Find matching books
                    matched_books, not_found_books = self.find_best_matches(book_list)
                    
                    if not matched_books:
                        st.warning("None of the provided books were found in our database.")
                        if not_found_books:
                            st.write("Books not found:")
                            for book in not_found_books:
                                st.write(f"- {book}")
                        return
                    
                    # Display matched books
                    st.subheader("Found These Books in Our Database:")
                    for i, book in enumerate(matched_books, 1):
                        st.write(f"{i}. {book['title']} (Match: {book['similarity']:.2%})")
                        with st.expander("Show Description"):
                            st.write(book['description'])
                    
                    # Show not found books if any
                    if not_found_books:
                        st.warning("Some books were not found:")
                        for book in not_found_books:
                            st.write(f"- {book}")
                    
                    # Get recommendations based on user's mood
                    recommendations = self.recommender.recommend_books(
                        user_mood,
                        n_recommendations=3
                    )
                    
                    st.subheader("Based on Your Mood, Here Are Your Recommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec['title']} (Similarity: {rec['similarity_score']:.4f})")
                        with st.expander("Show Description"):
                            st.write(rec['description'])
                
                except Exception as e:
                    st.error(f"Error processing recommendations: {str(e)}")
                    st.error(f"Traceback: {traceback.format_exc()}")

def main():
    st.set_page_config(
        page_title="Book Recommender",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Add CSS for dark theme and better styling
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
        .stWarning {
            background-color: #423D00;
            color: #FFD700;
        }
        </style>
    """, unsafe_allow_html=True)
    
    app = BookRecommenderApp()
    app.run()

if __name__ == "__main__":
    main()
