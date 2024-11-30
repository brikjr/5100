import streamlit as st
import pandas as pd
import numpy as np
from model import BookRecommender, Config
import torch
from pathlib import Path
import os
from typing import List, Dict
from model_image_analyzer import *

class BookRecommenderApp:
    def __init__(self, model_path: str = 'models/book_recommender'):
        self.model_path = model_path
        self.recommender = None
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'book_list' not in st.session_state:
            st.session_state.book_list = []
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = False
            
    def load_model(self):
        """Load the model and embeddings."""
        if self.recommender is None:
            with st.spinner('Loading model and embeddings... Please wait.'):
                self.recommender = BookRecommender(self.model_path)
    
    def process_image(self, uploaded_file):
        """Process uploaded image and extract book titles."""
        if uploaded_file is not None:
            with st.spinner('Processing Image, Please Wait...'):
                try:
                    book_list = processImageGetList(uploaded_file)  # Using the imported function
                    if book_list:
                        st.session_state.book_list = book_list
                        st.session_state.processed_image = True
                        return book_list
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        return []
    
    def get_book_recommendations(self, books: List[str], mood: str, n_recommendations: int) -> List[Dict]:
        """Get book recommendations based on mood."""
        try:
            # Generate mood embedding and get recommendations
            recommendations = self.recommender.recommend_books(
                mood_description=mood,
                n_recommendations=len(books)  # Get scores for all books
            )
            
            # Filter and match with user's books
            matched_recommendations = []
            for user_book in books:
                user_book_lower = user_book.lower()
                # Find matching recommendation
                for rec in recommendations:
                    if user_book_lower in rec['title'].lower():
                        matched_recommendations.append(rec)
                        break
            
            # Sort by similarity score and return top N
            matched_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            return matched_recommendations[:n_recommendations]
            
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def run(self):
        st.title("📚 Book Recommender System")
        
        try:
            self.load_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        # Image Upload Section
        st.header("1. Add Your Books")
        uploaded_file = st.file_uploader(
            "Upload an image of your bookshelf",
            ['jpg', 'jpeg', 'heic', 'png']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image")
                
        with col2:
            if st.button("Process Image", type="primary"):
                book_list = self.process_image(uploaded_file)
                if book_list:
                    st.success(f"Found {len(book_list)} books!")
                else:
                    st.warning("No books detected. Try manual entry.")
        
        # Manual Entry/Edit Section
        st.header("2. Edit Book List")
        default_text = "\n".join(st.session_state.book_list) if st.session_state.book_list else ""
        books_input = st.text_area(
            "Edit or add books (one per line):",
            value=default_text,
            height=150
        )
        
        if books_input:
            st.session_state.book_list = [
                book.strip() for book in books_input.split("\n")
                if book.strip()
            ]
        
        # Display current book list
        if st.session_state.book_list:
            st.write("Current book list:")
            for book in st.session_state.book_list:
                st.write(f"- {book}")
        
        # Mood Input Section
        st.header("3. Describe Your Reading Mood")
        mood = st.text_area(
            "What kind of book are you in the mood for?",
            help="Describe the type of story or experience you're looking for"
        )
        
        n_recommendations = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=min(5, len(st.session_state.book_list)) if st.session_state.book_list else 5,
            value=2
        )
        
        # Get Recommendations
        if st.button("Get Recommendations"):
            if not st.session_state.book_list or not mood:
                st.warning("Please provide both books and your mood description.")
                return
                
            recommendations = self.get_book_recommendations(
                st.session_state.book_list,
                mood,
                n_recommendations
            )
            
            if recommendations:
                st.header("📚 Recommended Books From Your List")
                st.write("Based on your mood:", mood)
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['title']} (Match: {rec['similarity_score']:.2f})"):
                        st.write("Description:", rec['description'])
            else:
                st.warning("""
                No matching books found. This could be because:
                1. The book titles weren't found in our database
                2. Try adjusting your mood description
                3. Add more books to your list
                """)

def main():
    st.set_page_config(
        page_title="Book Recommender",
        page_icon="📚",
        layout="wide"
    )
    
    # Add CSS
    st.markdown("""
        <style>
        .stTextArea textarea {
            font-size: 16px;
        }
        .stButton button {
            width: 100%;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .stExpander {
            background-color: #f0f2f6;
            border-radius: 4px;
            margin: 8px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    app = BookRecommenderApp()
    app.run()

if __name__ == "__main__":
    main()
