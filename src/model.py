import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json

class BookDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.titles = df['Book-Title'].tolist()
        self.descriptions = df['description'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.titles)
        
    def __getitem__(self, idx):
        # Combine title and description only
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

def save_embeddings(embeddings, book_ids, save_path):
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
    with open(os.path.join(save_path, 'book_ids.json'), 'w') as f:
        json.dump(book_ids, f)

def load_embeddings(load_path):
    embeddings = np.load(os.path.join(load_path, 'embeddings.npy'))
    with open(os.path.join(load_path, 'book_ids.json'), 'r') as f:
        book_ids = json.load(f)
    return embeddings, book_ids

def generate_book_embeddings(df, model, tokenizer, device):
    model.eval()
    embeddings = []
    batch_size = 32
    
    dataset = BookDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            embedding = model(input_ids, attention_mask)
            embeddings.extend(embedding.cpu().numpy())
    
    return np.array(embeddings)

def train_and_save_model(df, save_dir='book_model', batch_size=16, epochs=3, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BookEmbeddingModel().to(device)
    
    dataset = BookDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            embeddings = model(input_ids, attention_mask)
            augmented = embeddings + torch.randn_like(embeddings) * 0.01
            labels = torch.ones(embeddings.size(0)).to(device)
            
            loss = criterion(embeddings, augmented, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    model.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    embeddings = generate_book_embeddings(df, model, tokenizer, device)
    book_ids = {i: {'isbn': row['ISBN'], 'title': row['Book-Title']} 
                for i, row in df.iterrows()}
    save_embeddings(embeddings, book_ids, save_dir)
    
    return model, tokenizer, embeddings

def load_saved_model_and_embeddings(load_dir='book_model'):
    model = BookEmbeddingModel.load_model(load_dir)
    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    embeddings, book_ids = load_embeddings(load_dir)
    return model, tokenizer, embeddings, book_ids

def main():
    df = pd.read_csv('../data/books_with_valid_descriptions.csv')
    df = df.dropna(subset=['description'])
    
    print("Training and saving model...")
    model, tokenizer, embeddings = train_and_save_model(df, save_dir='book_model', epochs=3)
    
    print("\nLoading saved model and embeddings...")
    model, tokenizer, embeddings, book_ids = load_saved_model_and_embeddings('book_model')
    
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
    for idx in top_indices:
        book_info = book_ids[str(idx)]
        print(f"\nTitle: {book_info['title']}")
        print(f"ISBN: {book_info['isbn']}")
        print(f"Similarity: {similarities[idx]:.2f}")

if __name__ == "__main__":
    main()
