import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리
rating_data = pd.read_csv('../.././data/ratings.csv')
movie_data = pd.read_csv('../.././data/movies.csv')
rating_data.drop('timestamp',axis=1, inplace=True)
movie_data.drop('genres', axis=1, inplace=True)

# user_movie_data로 병합
user_movie_data = pd.merge(rating_data, movie_data,on= 'movieId')

# userId와 MovieId에 인덱스 부여
user_movie_data['userId'] = user_movie_data['userId'].astype('category').cat.codes.values
user_movie_data['movieId'] = user_movie_data['movieId'].astype('category').cat.codes.values

# 훈련 데이터로 나누기 (train/test split)
train_data, test_data = train_test_split(user_movie_data,test_size=0.2,random_state=42)

# pytorch 텐서로 변환
train_user_ids = torch.tensor(train_data['userId'].values, dtype=torch.long)
train_movie_ids = torch.tensor(train_data['movieId'].values, dtype=torch.long)
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float)

test_user_ids = torch.tensor(test_data['userId'].values, dtype=torch.long)
test_movie_ids = torch.tensor(test_data['movieId'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float)

# NNMF 모델 정의
class NNMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NNMF,self).__init__()

        # 사용자와 아이템 임베딩
        self.user_embedding = nn.Embedding(num_users,embedding_dim)
        self.item_embedding = nn.Embedding(num_items,embedding_dim)

        # Fully Connected Layers (MLP)
        self.fc1 = nn.Linear(embedding_dim*2,128) #사용자와 아이템 임베딩을 결합해서 입력
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,1) # 최종 평점 예측 출력

        # 활성화 함수
        self.relu = nn.ReLU()
    def forward(self, user_id, item_id) :

        # 사용자와 아이템 임베딩을 각각 구함
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)

        # 두 임베딩을 하나로 연결 (concat)
        x = torch.cat([user_emb, item_emb], dim=-1)

        # Fully Connected Layers를 통과
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        rating_pred = self.fc3(x) # 예측된 평점

        return rating_pred
    

# 모델 초기화 (사용자 수, 영화 수, 임베딩 차원 설정)
num_users = user_movie_data['userId'].nunique()
num_items = user_movie_data['movieId'].nunique()
embedding_dim = 50
model = NNMF(num_users, num_items, embedding_dim)

# 손실 함수 및 최적화 설정
criterion = nn.MSELoss() # 평점 예측이므로 MSE 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 학습 과정
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    # 예측 값 계산
    rating_preds = model(train_user_ids, train_movie_ids).squeeze()

    # 손실 계산
    loss = criterion(rating_preds,train_ratings)

    # 역전파 및 최적화

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')


# 테스트 데이터 평가
model.eval()
with torch.no_grad():
    test_preds = model(test_user_ids, test_movie_ids).squeeze()
    test_loss = criterion(test_preds, test_ratings)
    print(f'Test Loss: {test_loss.item():.4f}')