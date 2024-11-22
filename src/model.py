import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

class BookDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
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
        }

class BookEmbeddingModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, 'book_embeddings_model.pth'))
        
        config = {
            'model_type': 'BookEmbeddingModel',
            'base_model': 'bert-base-uncased'
        }
        with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_model(cls, load_path):
        with open(os.path.join(load_path, 'model_config.json'), 'r') as f:
            config = json.load(f)
        model = cls(model_name=config['base_model'])
        model.load_state_dict(torch.load(os.path.join(load_path, 'book_embeddings_model.pth')))
        return model

def save_embeddings(embeddings, book_info, save_path):
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
    with open(os.path.join(save_path, 'book_info.json'), 'w') as f:
        json.dump(book_info, f)

def load_embeddings(load_path):
    embeddings = np.load(os.path.join(load_path, 'embeddings.npy'))
    with open(os.path.join(load_path, 'book_info.json'), 'r') as f:
        book_info = json.load(f)
    return embeddings, book_info

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            embeddings = model(input_ids, attention_mask)
            augmented = embeddings + torch.randn_like(embeddings) * 0.01
            labels = torch.ones(embeddings.size(0)).to(device)
            
            loss = criterion(embeddings, augmented, labels)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_and_save_model(df, save_dir='book_model', batch_size=16, epochs=3, lr=2e-5, val_size=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=42)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BookEmbeddingModel().to(device)
    
    train_dataset = BookDataset(train_df, tokenizer)
    val_dataset = BookDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    best_val_loss = float('inf')
    early_stopping_patience = 3
    no_improvement = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            embeddings = model(input_ids, attention_mask)
            augmented = embeddings + torch.randn_like(embeddings) * 0.01
            labels = torch.ones(embeddings.size(0)).to(device)
            
            loss = criterion(embeddings, augmented, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            model.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print("Early stopping triggered!")
                break
    
    # Generate embeddings using full dataset
    full_dataset = BookDataset(df, tokenizer)
    full_loader = DataLoader(full_dataset, batch_size=batch_size)
    
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in full_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            embedding = model(input_ids, attention_mask)
            embeddings.extend(embedding.cpu().numpy())
    
    embeddings = np.array(embeddings)
    
    # Save embeddings and book info (only title and description)
    book_info = {i: {
        'title': row['Book-Title'],
        'description': row['description']
    } for i, row in df.iterrows()}
    
    save_embeddings(embeddings, book_info, save_dir)
    
    return model, tokenizer, embeddings

def main():
    # Load data
    df = pd.read_csv('../data/books_with_valid_descriptions.csv')
    df = df[['Book-Title', 'description']].dropna()  # Only keep needed columns
    
    print("Training and saving model...")
    model, tokenizer, embeddings = train_and_save_model(
        df, 
        save_dir='book_model',
        epochs=10
    )
    
    # Test the model
    query = "I'm looking for a mystery thriller with supernatural elements"
    device = next(model.parameters()).device
    
    query_encoding = tokenizer(
        query,
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        query_embedding = model(
            query_encoding['input_ids'],
            query_encoding['attention_mask']
        ).cpu().numpy()
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    print("\nTop matches:")
    book_info = load_embeddings(save_dir='book_model')[1]
    for idx in top_indices:
        info = book_info[str(idx)]
        print(f"\nTitle: {info['title']}")
        print(f"Similarity: {similarities[idx]:.2f}")
        print(f"Description: {info['description'][:200]}...")

if __name__ == "__main__":
    main()