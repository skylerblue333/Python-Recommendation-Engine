import numpy as np
from typing import List, Dict

class CollaborativeFilter:
    def __init__(self, num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        # Initialize random embeddings
        self.user_embeddings = np.random.normal(0, 0.1, (num_users, 10))
        self.item_embeddings = np.random.normal(0, 0.1, (num_items, 10))
        
    def train(self, interactions: List[tuple], epochs: int = 10, lr: float = 0.01):
        for epoch in range(epochs):
            total_loss = 0
            for u, i, r in interactions:
                # Predict
                pred = np.dot(self.user_embeddings[u], self.item_embeddings[i])
                err = r - pred
                total_loss += err**2
                
                # Update (Gradient Descent)
                u_grad = -2 * err * self.item_embeddings[i]
                i_grad = -2 * err * self.user_embeddings[u]
                
                self.user_embeddings[u] -= lr * u_grad
                self.item_embeddings[i] -= lr * i_grad
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(interactions):.4f}")

    def recommend(self, user_id: int, top_k: int = 5) -> List[int]:
        scores = np.dot(self.item_embeddings, self.user_embeddings[user_id])
        return np.argsort(scores)[::-1][:top_k].tolist()
