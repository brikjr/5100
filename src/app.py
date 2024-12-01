import streamlit as st
import pandas as pd
import numpy as np
from model import BookRecommender, Config
import torch
import traceback
from pathlib import Path
import os
from typing import List, Dict
from model_image_analyzer import *
from sklearn.metrics.pairwise import cosine_similarity

'''
Class: BookRecommenderApp
    This class defines the streamlit application that implements the bookshelf analyzer and receommender
    model. This class defines methods for generating the UI, imlpementing the saved model, reading the
    bookshelf, and creating recommendations.
Parameters:
    model_path: Optional parameter to specify a custom trained model
    dataset_path: Optional parameter to specify a CSV dataset with book titles and descriptions.
Output: Generates a stremalit UI
'''
class BookRecommenderApp:
    # Initialize the class and accept a model path and a dataset path (both optional)
    def __init__(self, model_path: str = 'models/book_recommender', dataset_path: str = None):
        self.model_path = model_path
        self.recommender = None
        self.dataset = self.load_dataset_for_descriptions(dataset_path)
        self.initialize_session_state()
    
    # Initialize the session state for storing local variables and images
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'book_list' not in st.session_state:
            st.session_state.book_list = []
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = False
    
    # Load the model and the embeddings
    def load_model(self):
        """Load the model and embeddings."""
        if self.recommender is None:
            with st.spinner('Loading model and embeddings... Please wait.'):
                self.recommender = BookRecommender(self.model_path)
    
    # Process the image to extract the book titles in a list format
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
    
    '''
    Get the best book recommendations. This function disregards the provided bookshelf image and just recommends the books
    that match the mood the best from the dataset of books.
    '''
    def get_best_book_recommendations(self, books: List[str], mood: str, n_recommendations: int) -> List[Dict]:
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
    
    # This function generates embeddings for each book title and recommends the best title from the image based on mood
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
    
    # Load the dataset for generating descriptions only
    def load_dataset_for_descriptions(self, dataset):
        if dataset is not None:
            print("Loading book dataset...")
            df = pd.read_csv('../data/books_with_valid_descriptions.csv')  # Replace with your dataset path
            df = df[['Book-Title', 'description']].dropna()
            print(f"Loaded {len(df)} books with valid descriptions.")
            return df
        else: 
            return None
    
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
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image")
            
        if st.button("Process Image", type="primary"):
            st.session_state.book_list = self.process_image(uploaded_file)
            if st.session_state.book_list:
                st.success(f"Found {len(st.session_state.book_list)} books!")
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

        if not st.session_state.book_list or len(st.session_state.book_list) < n_recommendations:
            st.warning(f"Please enter at least {n_recommendations} books to get {n_recommendations} recommendations.")
            return
        
        # Get Recommendations
        if st.button("Get Recommendations"):
            try:
                if not st.session_state.book_list or not mood:
                    st.warning("Please provide both books and your mood description.")
                    return
                
                # Get recommendations from user's list
                recommendations = self.recommend_from_user_list(
                    st.session_state.book_list,
                    mood,
                    n_recommendations
                )

                # recommendations = self.get_book_recommendations(
                #     st.session_state.book_list,
                #     mood,
                #     n_recommendations
                # )
                    
                # Display input books
                st.subheader("Your Input Books:")
                for book in st.session_state.book_list:
                    st.write(f"- {book}")
                
                st.subheader(f"📚 Top {n_recommendations} Recommended Books From Your List:")
                st.write("Based on your mood:", mood)
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['title']} (Match: {rec['similarity_score']:.2f})"):
                        if(self.dataset is not None):
                            row = self.dataset.loc[self.dataset['Book-Title'].str.strip().str.lower() == rec['title'].strip().lower()]
                            if not row.empty:
                                st.write("Description:", row.iloc[0]['description'])
                            else:
                                st.write("Description:", "Sorry this desription is not available because this book is not our database.")

                        else: st.write("Description:", "Sorry this desription is not available because this book is not our database.")

                    
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
        }
        .stButton button {
            width: 100%;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .stExpander {
            background-color: #f0f2f6;
            color: #000000;
            font-weight: bold;
            border-radius: 4px;
            margin: 8px 0;
        }
        .stExpander .stMarkdown {
            color: #000000;           
        }
        .stExpander:hover {
            background-color: #f5f5f5;
        }

        </style>
    """, unsafe_allow_html=True)
    
    app = BookRecommenderApp(datasetPath="../data/books_with_valid_descriptions.csv")
    app.run()

if __name__ == "__main__":
    main()
