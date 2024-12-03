import pandas as pd
import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
from tqdm.auto import tqdm
import os

class Config:
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    SAVE_PATH = 'models/book_recommender'
    N_RECOMMENDATIONS = 5
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 512
    DROPOUT = 0.3
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class BookRecommenderDataset(Dataset):
    def _init_(self, df: pd.DataFrame, tokenizer, max_length: int = Config.MAX_LENGTH):
        self.titles = df['Book-Title'].tolist()
        self.descriptions = df['description'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def _len_(self):
        return len(self.titles)
        
    def _getitem_(self, idx):
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
    def _init_(self, model_name: str = Config.MODEL_NAME):
        super()._init_()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.transformer.config.hidden_size
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

class RecommenderModel(nn.Module):
    def _init_(self, embedding_dim: int = Config.EMBEDDING_DIM):
        super()._init_()
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
    def _init_(self, model_path: str = None):
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
    
    def verify_saved_data(self, save_path: str):
        """Verify that the data was saved correctly."""
        try:
            model_path = os.path.join(save_path, 'models.pth')
            embedding_path = os.path.join(save_path, 'book_embeddings.npy')
            csv_path = os.path.join(save_path, 'books.csv')
            
            files_exist = all(os.path.exists(p) for p in [model_path, embedding_path, csv_path])
            
            if files_exist:
                print("All files saved successfully!")
                print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
                print(f"Embeddings size: {os.path.getsize(embedding_path) / 1024 / 1024:.2f} MB")
                print(f"CSV size: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")
                return True
            else:
                print("Some files are missing!")
                return False
                
        except Exception as e:
            print(f"Error verifying saved data: {str(e)}")
            return False

    def train(self, df: pd.DataFrame, save_path: str = 'models/book_recommender', 
            hidden_dim: int = Config.HIDDEN_DIM, 
            dropout: float = Config.DROPOUT, 
            batch_size: int = Config.BATCH_SIZE, 
            learning_rate: float = Config.LEARNING_RATE):
        """
        Train the Book Recommender model with specified hyperparameters.
        Args:
            df (pd.DataFrame): The training dataset.
            save_path (str): Path to save the trained model.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
            batch_size (int): Training batch size.
            learning_rate (float): Learning rate for the optimizer.
        """
        print("\nInitializing training process...")
        self.books_df = df

        # Dynamically update Config
        Config.HIDDEN_DIM = hidden_dim
        Config.DROPOUT = dropout
        Config.BATCH_SIZE = batch_size
        Config.LEARNING_RATE = learning_rate

        print(f"Creating dataset with {len(df)} books...")
        dataset = BookRecommenderDataset(df, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE,  # Use updated batch size
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print("\nGenerating book embeddings...")
        self.book_embeddings = self._generate_embeddings(dataloader)
        
        print("\nInitializing recommender model...")
        self.recommender = RecommenderModel(self.embedding_model.embedding_dim).to(self.device)
        
        # Optimizer with updated learning rate
        optimizer = torch.optim.Adam(self.recommender.parameters(), lr=Config.LEARNING_RATE)

        print("\nSaving models and embeddings...")
        self.save_models(save_path)
        
        # Verify the save was successful
        if self.verify_saved_data(save_path):
            print("Training and saving completed successfully!")
        else:
            print("Warning: There might be issues with saved data!")

    
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
        try:
            os.makedirs(save_path, exist_ok=True)
            
            print("Saving models...")
            model_path = os.path.join(save_path, 'models.pth')
            torch.save({
                'embedding_model': self.embedding_model.state_dict(),
                'recommender': self.recommender.state_dict() if self.recommender else None
            }, model_path)
            
            print("Saving embeddings...")
            embedding_path = os.path.join(save_path, 'book_embeddings.npy')
            np.save(embedding_path, self.book_embeddings)
            
            print("Saving book data...")
            csv_path = os.path.join(save_path, 'books.csv')
            self.books_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"All data saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            raise
    
    def load_models(self, load_path: str):
        try:
            print("Loading saved models...")
            model_path = os.path.join(load_path, 'models.pth')
            checkpoint = torch.load(model_path)
            self.embedding_model.load_state_dict(checkpoint['embedding_model'])
            
            if checkpoint['recommender']:
                self.recommender = RecommenderModel(self.embedding_model.embedding_dim).to(self.device)
                self.recommender.load_state_dict(checkpoint['recommender'])
            
            print("Loading embeddings and book data...")
            embedding_path = os.path.join(load_path, 'book_embeddings.npy')
            self.book_embeddings = np.load(embedding_path)
            
            csv_path = os.path.join(load_path, 'books.csv')
            self.books_df = pd.read_csv(csv_path, encoding='utf-8')
            
            print("Model loading completed!")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

def evaluate_recommendations(recommender, test_moods):
    """
    Evaluate the model's performance based on recommendation similarity scores.
    Args:
        recommender (BookRecommender): The trained recommender system.
        test_moods (list): A list of mood descriptions to test.

    Returns:
        float: The average similarity score across all test moods.
    """
    all_scores = []
    for mood in test_moods:
        recommendations = recommender.recommend_books(mood)
        
        # Calculate average similarity score for the recommendations
        similarity_scores = [rec['similarity_score'] for rec in recommendations]
        avg_score = np.mean(similarity_scores)
        all_scores.append(avg_score)
    
    # Return the overall average score
    return np.mean(all_scores)


def main():
    try:
        os.makedirs('models/book_recommender', exist_ok=True)
        
        print("Loading book dataset...")
        df = pd.read_csv('../data/books_with_valid_descriptions.csv')
        df = df[['Book-Title', 'description']].dropna()
        print(f"Loaded {len(df)} books with valid descriptions")
        
        # Define hyperparameter grid
        hidden_dims = [512, 1024, 2048]  
        dropouts = [0.1, 0.3, 0.5]      
        batch_sizes = [32, 64]           
        learning_rates = [1e-3, 1e-4]    
        # Define test moods
        test_moods = [
            "I want to read something magical and adventurous with dragons",
            "Looking for a romantic story set in modern times",
            "Need a thrilling mystery that will keep me guessing",
            "Interested in historical fiction about World War II",
            "Want to read about science and space exploration"
        ]

        # Store results
        results = []

        # Loop through all hyperparameter combinations
        for hidden_dim in hidden_dims:
            for dropout in dropouts:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        print(f"\nTesting HIDDEN_DIM={hidden_dim}, DROPOUT={dropout}, BATCH_SIZE={batch_size}, LEARNING_RATE={learning_rate}")

                        # Initialize and train the model
                        recommender = BookRecommender()
                        recommender.train(
                            df,
                            save_path=f'models/h{hidden_dim}_d{dropout}_b{batch_size}_lr{learning_rate}',
                            hidden_dim=hidden_dim,
                            dropout=dropout,
                            batch_size=batch_size,
                            learning_rate=learning_rate
                        )
                        
                        # Evaluate the model using test moods
                        avg_similarity_score = evaluate_recommendations(recommender, test_moods)
                        print(f"Average Recommendation Similarity: {avg_similarity_score:.4f}")
                        
                        # Log the results
                        results.append({
                            'hidden_dim': hidden_dim,
                            'dropout': dropout,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'avg_similarity_score': avg_similarity_score
                        })

        # Save the results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv('hyperparameter_results.csv', index=False)
        print("\nResults saved to 'hyperparameter_results.csv'")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if _name_ == "_main_":
    main()