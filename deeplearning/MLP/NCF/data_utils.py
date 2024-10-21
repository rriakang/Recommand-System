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
    for x in train_data :
        train_mat[x[0],x[1]] = 1.0 # 평가가 존재함을 나타냄
    test_data = []

    # with open 사용: 테스트 파일은 복잡한 형식을 가질 수 있기 때문에, 일반적인 CSV 형식으로 처리하는 것이 적합하지 않을 수 있음
    # 따라서 with open과 같은 파일 입출력 방법으로 파일을 한 줄씩 읽어와서 수동으로 처리
 
    with open(config.test_negative,'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')

            # eval()의 역할: 이 함수는 문자열을 실행 가능한 Python 코드로 평가. 이 경우, 예를 들어 (1, 10) 같은 형식이 있다면 이를 튜플로 변환함.
            # eval(arr[0])[0]: 첫 번째 값을 가져와 사용자 ID u로 사용. 예를 들어, (1, 10)에서 1이 사용자 ID로 저장된다.
            u = eval(arr[0])[0] #사용자 ID저장
            test_data.append([u,eval(arr[0])[1]]) # eval(arr[0])[1] : 아이템 , 형태로 사용자 ID와 평가된 아이템 ID를 하나의 리스트로 묶어 test_data 리스트에 추가
            for i in arr[1:]:
                test_data.append([u,int(i)]) # arr의 첫 번째 항목을 제외한 나머지 값들 ex) (1,10) 다음으로 20,30,40과 같은 값들을 test_data리스트에 추가함
            line = fd.readline() # 파일에서 다음 줄을 읽기  (다음줄이 없을때까지)
    return train_data,test_data,user_num,item_num,train_mat

    

   