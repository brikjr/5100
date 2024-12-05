import os
import pandas as pd
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Tuple

# class config to store all the hyperparameters
class Config:
    MODEL_NAME = 'bert-base-uncased'  # Pretrained model name
    MAX_LENGTH = 256  # Maximum sequence length for the model
    BATCH_SIZE = 32  # Training batch size for data loader
    SAVE_PATH = 'models/book_recommender'  # Path to save the trained models
    N_RECOMMENDATIONS = 10  # Number of book recommendations to return
    EMBEDDING_DIM = 768  #Dimensionality of BERT embeddings
    HIDDEN_DIM = 512   # Hidden dimension for the recommender model
    DROPOUT = 0.3   # Dropout rate for regularization
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Custom dataset class to load the book data
class BookRecommenderDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = Config.MAX_LENGTH):
        self.titles = df['Book-Title'].tolist()
        self.descriptions = df['description'].tolist()
        self.tokenizer = tokenizer   
        self.max_length = max_length
        
    def __len__(self):
        return len(self.titles)  # Return the number of books
        
    def __getitem__(self, idx):
        # Combine the title and description for tokenization
        book_text = f"Title: {self.titles[idx]} Description: {self.descriptions[idx]}"
        
        # Tokenize the text
        encoding = self.tokenizer(
            book_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'title': self.titles[idx]
        }

# Model to generate book embeddings usign transformer model
class BookEmbeddingModel(nn.Module):
    def __init__(self, model_name: str = Config.MODEL_NAME):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)  # Load the transformer model
        self.embedding_dim = self.transformer.config.hidden_size  # Get the embedding dimension
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

# Feed forward neural network to recommend books based on mood
class RecommenderModel(nn.Module):
    def __init__(self, embedding_dim: int = Config.EMBEDDING_DIM):
        super().__init__()
        
        # Define a neural network to encode mood embeddings
        self.mood_encoder = nn.Sequential(
            nn.Linear(embedding_dim, Config.HIDDEN_DIM),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Dropout(Config.DROPOUT),  # Dropout layer
            nn.Linear(Config.HIDDEN_DIM, embedding_dim)  # Linear layer
        )
        
    def forward(self, mood_embedding, book_embeddings):
        # Encode the mood embedding and compute cosine similarity with book embeddings
        mood_encoded = self.mood_encoder(mood_embedding)
        return cosine_similarity(
            mood_encoded.detach().cpu().numpy(),
            book_embeddings.detach().cpu().numpy()
        )

# Main class to manage training, saving, and recommending books
class BookRecommender:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)  # Load the tokenizer
        self.embedding_model = BookEmbeddingModel().to(self.device)  # Load the embedding model
        self.recommender = None
        self.book_embeddings = None
        self.books_df = None
        
        # Load existing models if a path is provided
        if model_path and os.path.exists(model_path):
            self.load_models(model_path)
            print(f"Loaded existing models from {model_path}")
    
    def train(self, df: pd.DataFrame, save_path: str = Config.SAVE_PATH):
        print("\nInitializing training process...")
        self.books_df = df
        
        # Create a dataset and dataloader
        print(f"Creating dataset with {len(df)} books...")
        dataset = BookRecommenderDataset(df, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Generate book embeddings
        print("\nGenerating book embeddings...")
        self.book_embeddings = self._generate_embeddings(dataloader)
        
        # Initialize the recommender model
        print("\nInitializing recommender model...")
        self.recommender = RecommenderModel(self.embedding_model.transformer.config.hidden_size).to(self.device)
        
        # save the recommender model
        print("\nSaving models and embeddings...")
        self.save_models(save_path)
        
    def _generate_embeddings(self, dataloader: DataLoader) -> np.ndarray:
        self.embedding_model.eval()  # Set the model to evaluation mode
        embeddings = []
        
        # Use tqdm for progress bar
        progress_bar = tqdm(
            dataloader,
            desc="Generating embeddings",
            total=len(dataloader),
            unit="batch"
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                embedding = self.embedding_model(input_ids, attention_mask)
                embeddings.extend(embedding.cpu().numpy())   # Append the embeddings to the list
                
                progress_bar.set_postfix({
                    'batch_size': input_ids.shape[0],  # current batch size
                    'embeddings': len(embeddings)   # total embeddings generated
                })
        
        return np.array(embeddings)  # return as numpy array
    
    def recommend_books(
        self, 
        mood_description: str, 
        n_recommendations: int = Config.N_RECOMMENDATIONS
    ) -> List[Dict]:
        print(f"\nProcessing mood: '{mood_description}'")
        
        # tokenize the mood description
        mood_encoding = self.tokenizer(
            mood_description,
            truncation=True,
            padding=True,
            max_length=Config.MAX_LENGTH,
            return_tensors='pt'  # Return PyTorch tensors
        ).to(self.device)
        
        with torch.no_grad():
            # Get the mood embedding
            mood_embedding = self.embedding_model(
                mood_encoding['input_ids'],
                mood_encoding['attention_mask']
            )
            
            # Get the similarity scores with book embeddings
            similarity_scores = self.recommender(
                mood_embedding,
                torch.tensor(self.book_embeddings).to(self.device)
            )[0]
        
        # Get the top N book recommendations
        top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:  # Iterate over the top indices
            book = self.books_df.iloc[idx]
            recommendations.append({
                'title': book['Book-Title'],  # Get the book title
                'description': book['description'],  # Get the book description
                'similarity_score': similarity_scores[idx]  # Get the similarity score
            })
        
        return recommendations
    
    def save_models(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
        print("Saving models...")
        torch.save({
            'embedding_model': self.embedding_model.state_dict(),  # Save the embedding model
            'recommender': self.recommender.state_dict() if self.recommender else None
        }, os.path.join(save_path, 'models.pth'))
        np.save(os.path.join(save_path, 'book_embeddings.npy'), self.book_embeddings)  # Save the book embeddings
        self.books_df.to_csv(os.path.join(save_path, 'books.csv'), index=False, encoding='utf-8')  # Save the book data as CSV
    
    def load_models(self, load_path: str):
        print("Loading saved models...")
        # Load the saved models
        checkpoint = torch.load(os.path.join(load_path, 'models.pth'))  # Load the model states
        self.embedding_model.load_state_dict(checkpoint['embedding_model'])  # Load the embedding model
        if checkpoint['recommender']:
            # Load the recommender model if available
            self.recommender = RecommenderModel(self.embedding_model.transformer.config.hidden_size).to(self.device)
            self.recommender.load_state_dict(checkpoint['recommender'])
        self.book_embeddings = np.load(os.path.join(load_path, 'book_embeddings.npy'))  # Load the book embeddings
        self.books_df = pd.read_csv(os.path.join(load_path, 'books.csv'), encoding='utf-8')  # Load the book data

def main():
    try:
        # Ensure the save directory exists
        os.makedirs(Config.SAVE_PATH, exist_ok=True)
        
        # Load your dataset
        print("Loading book dataset...")
        df = pd.read_csv('../data/books_with_valid_descriptions.csv')  # Replace with your dataset path
        df = df[['Book-Title', 'description']].dropna()
        print(f"Loaded {len(df)} books with valid descriptions.")
        
        # Initialize the recommender system
        recommender = BookRecommender()
        
        # Train the model
        recommender.train(df, save_path=Config.SAVE_PATH)
        
        # Test recommendations
        test_moods = [
            "I want to read something magical and adventurous with dragons.",
            "Looking for a romantic story set in modern times.",
            "Need a thrilling mystery that will keep me guessing.",
            "Interested in historical fiction about World War II.",
            "Want to read about science and space exploration."
        ]
        
        print("\nTesting recommender system...")
        for i, mood in enumerate(test_moods, 1):
            print(f"\nTest {i}/{len(test_moods)}")
            recommendations = recommender.recommend_books(mood)
            
            print(f"\nFor mood: {mood}")
            print("\nRecommended Books:")
            for j, book in enumerate(recommendations, 1):
                print(f"\n{j}. {book['title']}")
                print(f"Similarity Score: {book['similarity_score']:.4f}")
                print(f"Description: {book['description'][:200]}...")
        
        print("\nRecommendation testing completed!")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")   # Handle keyboard interrupt
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")   # Handle any other errors
        raise

if __name__ == "__main__":
    main()
