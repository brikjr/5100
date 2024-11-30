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

class Config:
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    SAVE_PATH = 'models/book_recommender'
    N_RECOMMENDATIONS = 10
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 512
    DROPOUT = 0.3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class BookRecommenderDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = Config.MAX_LENGTH):
        self.titles = df['Book-Title'].tolist()
        self.descriptions = df['description'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.titles)
        
    def __getitem__(self, idx):
        book_text = f"Title: {self.titles[idx]} Description: {self.descriptions[idx]}"
        
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

class BookEmbeddingModel(nn.Module):
    def __init__(self, model_name: str = Config.MODEL_NAME):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.transformer.config.hidden_size
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

class RecommenderModel(nn.Module):
    def __init__(self, embedding_dim: int = Config.EMBEDDING_DIM):
        super().__init__()
        self.mood_encoder = nn.Sequential(
            nn.Linear(embedding_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, embedding_dim)
        )
        
    def forward(self, mood_embedding, book_embeddings):
        mood_encoded = self.mood_encoder(mood_embedding)
        return cosine_similarity(
            mood_encoded.detach().cpu().numpy(),
            book_embeddings.detach().cpu().numpy()
        )

class BookRecommender:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.embedding_model = BookEmbeddingModel().to(self.device)
        self.recommender = None
        self.book_embeddings = None
        self.books_df = None
        
        if model_path and os.path.exists(model_path):
            self.load_models(model_path)
            print(f"Loaded existing models from {model_path}")
    
    def train(self, df: pd.DataFrame, save_path: str = Config.SAVE_PATH):
        print("\nInitializing training process...")
        self.books_df = df
        
        print(f"Creating dataset with {len(df)} books...")
        dataset = BookRecommenderDataset(df, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print("\nGenerating book embeddings...")
        self.book_embeddings = self._generate_embeddings(dataloader)
        
        print("\nInitializing recommender model...")
        self.recommender = RecommenderModel(self.embedding_model.transformer.config.hidden_size).to(self.device)
        
        print("\nSaving models and embeddings...")
        self.save_models(save_path)
        
    def _generate_embeddings(self, dataloader: DataLoader) -> np.ndarray:
        self.embedding_model.eval()
        embeddings = []
        
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
                embeddings.extend(embedding.cpu().numpy())
                
                progress_bar.set_postfix({
                    'batch_size': input_ids.shape[0],
                    'embeddings': len(embeddings)
                })
        
        return np.array(embeddings)
    
    def recommend_books(
        self, 
        mood_description: str, 
        n_recommendations: int = Config.N_RECOMMENDATIONS
    ) -> List[Dict]:
        print(f"\nProcessing mood: '{mood_description}'")
        
        mood_encoding = self.tokenizer(
            mood_description,
            truncation=True,
            padding=True,
            max_length=Config.MAX_LENGTH,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            mood_embedding = self.embedding_model(
                mood_encoding['input_ids'],
                mood_encoding['attention_mask']
            )
            
            similarity_scores = self.recommender(
                mood_embedding,
                torch.tensor(self.book_embeddings).to(self.device)
            )[0]
        
        top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            book = self.books_df.iloc[idx]
            recommendations.append({
                'title': book['Book-Title'],
                'description': book['description'],
                'similarity_score': similarity_scores[idx]
            })
        
        return recommendations
    
    def save_models(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        print("Saving models...")
        torch.save({
            'embedding_model': self.embedding_model.state_dict(),
            'recommender': self.recommender.state_dict() if self.recommender else None
        }, os.path.join(save_path, 'models.pth'))
        np.save(os.path.join(save_path, 'book_embeddings.npy'), self.book_embeddings)
        self.books_df.to_csv(os.path.join(save_path, 'books.csv'), index=False, encoding='utf-8')
    
    def load_models(self, load_path: str):
        print("Loading saved models...")
        checkpoint = torch.load(os.path.join(load_path, 'models.pth'))
        self.embedding_model.load_state_dict(checkpoint['embedding_model'])
        if checkpoint['recommender']:
            self.recommender = RecommenderModel(self.embedding_model.transformer.config.hidden_size).to(self.device)
            self.recommender.load_state_dict(checkpoint['recommender'])
        self.book_embeddings = np.load(os.path.join(load_path, 'book_embeddings.npy'))
        self.books_df = pd.read_csv(os.path.join(load_path, 'books.csv'), encoding='utf-8')

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
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
