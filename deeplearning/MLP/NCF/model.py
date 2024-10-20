import torch
import torch.nn as nn
import torch.optim as optim


class NCF(nn.Module):
    def __init__(self,num_user,num_item):
        super(NCF,self).__init__()
        
       
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = 16
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # user과 아이템 임베딩
        self.embedding_user = nn.Embedding(num_embeddings=self.num_user,embedding_dim=self.embedding_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_item, embedding_dim=self.embedding_dim)

        self.fc1 = nn.Linear(in_features=32, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=8)
        self.fc3 = nn.Linear(in_features=8,out_features=1)

    def forward(self,user_indices,item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding,item_embedding],dim=-1) 

        x = self.fc1(vector)
        x = self.relu(x)
        x = self.fc3(x)
        out = self.sigmoid(x)

        return out.squeeze()
    
