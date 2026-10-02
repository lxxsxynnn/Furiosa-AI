from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.metrics import accuracy_score

# 임베딩 실습 - 로이터 뉴스 기사 주제 분류
# 1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,     # 단어사전의 개수, 빈도수가 높은 단어 순으로 1000개 뽑기
    # maxlen=1000,          # maxlen 이상인 기사는 제외 (자르는 게 아님). 남은 기사도 길이가 달라 패딩 필요
    test_split=0.2,
)

print(x_train)
'''[list([1, 2, ... , 15, 17, 12])
 list([1, 2, ... , 505, 17, 12])
 ... 
 list([1, 227, ..., 113, 17, 12])]
 '''
print(x_train.shape, y_train.shape) # (8982,) (8982,)
print(x_test.shape, y_test.shape)   # (2246,) (2246,)
print(y_train)                      # [3 4 3 ... 25 3 25]
print(np.unique(y_train))           # [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train))                # <class 'numpy.ndarray'> - 기사(리스트)를 원소로 담은 1차원 배열
print(type(x_train[0]))             # <class 'list'> - pad_sequences로 길이를 맞추면 2차원 np.ndarray 배열이 됨
print(len(x_train[0]))              # 87

maxlen = max(len(i) for i in x_train)
minlen = min(len(i) for i in x_train)
avglen = sum(map(len, x_train))/len(x_train)
avglen2 = sum(map(len, x_test))/len(x_test)


print("뉴스 기사의 최대 길이: ", maxlen)    # 뉴스 기사의 최대 길이:  2376
print("뉴스 기사의 최소 길이: ", minlen)    # 뉴스 기사의 최소 길이:  13
print("뉴스 기사의 평균 길이: ", avglen)    # 뉴스 기사의 평균 길이:  145.5398574927633
print("뉴스 기사 테스트 set의 평균 길이: ", avglen2)    # 147.66117542297417s   

# 전처리(패드 시퀀스)
padded_x_train = pad_sequences(x_train, padding='pre',
                               maxlen=200,
                               truncating='pre',
                               )

padded_x_test = pad_sequences(x_test, padding='pre',
                              maxlen=200,
                              truncating='pre',
                              )

print(padded_x_train.shape)   # (8982, 200)
print(padded_x_test.shape)    # (2246, 200)

# y 원핫 인코딩
encoded_y_train = to_categorical(y_train)
encoded_y_test = to_categorical(y_test)

print(encoded_y_train.shape)  # (8982, 46)
print(encoded_y_test.shape)   # (2246, 46)

# 2. 모델 구성
model = Sequential()
model.add(Embedding(1000, 100, input_length=200))
model.add(LSTM(200))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(46, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'],
              )
model.fit(padded_x_train, encoded_y_train,
          epochs=1000,
          batch_size=256,
          )

# acc 0.67 이상
# 4. 평가, 예측
result = model.evaluate(padded_x_test, encoded_y_test)
print('loss : ', result[0]) # loss :  3.4982187747955322
print('acc : ', result[1])  # acc :  0.739091694355011

y_predict = model.predict(padded_x_test)
y_predict = np.argmax(y_predict, axis=1)
print('acc_score : ', accuracy_score(y_test, y_predict))    # acc_score :  0.7390917186108638