import time
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical

# 원핫인코딩 적용하기
# 1. 데이터
docs = [
    '너무 재미있다', '참 최고예요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로예요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    'ㅇㅇㅇ 바보', 'ㅁㅁㅁ 잘생겼다', 'ㄴㄴㄴ 또 거짓말한다'
]

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])

token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, # 패딩할 대상
                         padding='pre',
                         maxlen = 5,        # 최대 문장 길이
                         )

print(padded_x.shape)   # (15, 5)

# 2. 모델 구성
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

model = Sequential()
############### 임베딩 1 ###############
model.add(Embedding(input_dim=30, output_dim=10, input_length=5))   # input_dim - 단어사전의 갯수, output_dim - 차원
model.add(SimpleRNN(10))
model.add(Dense(1))

############### 임베딩 2 ###############
model.add(Embedding(input_dim=30, output_dim=10))       # input_length를 적지 않아도 embedding layer에서는 알아서 맞춰줌
model.add(SimpleRNN(10))
model.add(Dense(1))

############### 임베딩 3 ###############
model.add(Embedding(30, 10))                          # 필드명을 명시하지 않고도 사용 가능 **** 텐서플로우에서 유일하게 input_dim이 먼저 오는 레이어
# model.add(Embedding(30, 10, 5))                       # input_length는 생략 불가 Embeding((30, 10, input_length=5)) 형태로 명시해줘야 함
model.add(SimpleRNN(10))
model.add(Dense(1))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 5, 10)             300
 simple_rnn (SimpleRNN)      (None, 10)                210
 dense (Dense)               (None, 1)                 11
=================================================================
Total params: 521
Trainable params: 521
Non-trainable params: 0

input_dim * outpput_dim = 30 * 10 = 300
'''

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],)

model.fit(padded_x, labels, epochs=3,)