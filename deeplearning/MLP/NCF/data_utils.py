import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data
import config

def load_all(test_numm=100):


    # NCF는 user와 item간의 상호작용 데이터만 사용하기 때문에 train_rating.txt파일에서 user,item 
    # 열만 불러옴
    train_data = pd.read_csv(
        config.train_rating,
        sep='\t', header=None, names=['user','item'],
        usecols=[0,1], dtype={0: np.int32, 1:np.int32})
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()
    
    # user-item interaction 희소행렬을 key-value 딕셔너리 형태로 정의
    train_mat = sp.dok_matrix((user_num,item_num),dtype=np.float32)