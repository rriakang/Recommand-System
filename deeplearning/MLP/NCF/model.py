import torch
import torch.nn as nn
import torch.optim as optim




class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model, GMF_model = None, MLP_model = None):
        super(NCF,self).__init__()
        
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        # factor_num은 NCF모델에서 사용자의 특성과 아이템의 특성을 임베딩하기 위한 차원 수를 정의
        self.embed_user_GMF = nn.Embedding(user_num,factor_num)
        self.embed_item_GMF = nn.Embedding(item_num,factor_num)
        self.embed_user_MLP = nn.Embedding(user_num,factor_num * (2 ** (num_layers -1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers -1 )))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size,input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size,1)
        self._init_weight_()

    # 최적의 학습을 위해 _init_weight() 함수로 각 가중치를 초기화함
    # 초기화 과정들은 신경망이 더 안정적으로 학습, 수렴 속도를 높이는데 중요한 역할을 함
    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)


        
        for m in self.MLP_layers:
            if isinstance(m,nn.Linear):
                # Xavier initialization
                # xavier 초기화는 주로 sigmoid나 Tanh와 같은 활성화 함수를 사용하는 신경망에서 가중치를 초기화할때 사용
                nn.init.xavier_uniform_(m.weight)
                # Kaimin Initialization
                # kaiming 초기화는 주로 Relu 활성화 함수와 함께 사용됨
                nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                a=1, nonlinearity='sigmoid')        
        
        for m in self.MLP_layers:
            if isinstance(m,nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self,user,item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)

        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_GMF(item)
        # 벡터 결합 : interaction은 사용자 임베딩과 아이템 임베딩을 연결(concatenate)하여 두 벡터를 하나로 만듬
        interaction = torch.cat((embed_user_MLP,embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF,output_MLP),-1)

        prediction = self.predict_layer(concat)
        
        return prediction.view(-1)



        






