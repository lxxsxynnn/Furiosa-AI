from tensorflow.keras.preprocessing.text import Tokenizer

# 문장을 토큰화해보기
text = 'Hello. Today is Oct 1st. It\'s a bit chilly today. It\'s sunny. It\'s 9:34 AM.'

token = Tokenizer()
# 1. Tokenizer : 클래스. 케라스 안에 class Tokenizer:로 이미 정의돼 있음
# 2. () : 그 클래스를 호출해서 인스턴스를 새로 만듦 (인스턴스화)
# 3. token : 그렇게 만든 인스턴스를 담은 변수

token.fit_on_texts([text])
print(token.word_index)
# {"it's": 1, 'today': 2, 'hello': 3, 'is': 4, 'oct': 5, '1st': 6, 'a': 7, 'bit': 8, 'chilly': 9, 'sunny': 10, '9': 11, '34': 12, 'am': 13}
# 많이 나온 단어일수록 앞 번호, 횟수가 같으면 먼저 나온 순서. 번호는 1부터
# 소문자로 바꾸고 . : 같은 문장부호는 지움. 작은따옴표는 남아서 it's가 한 단어

print(token.word_counts)
# OrderedDict([('hello', 1), ('today', 2), ('is', 1), ('oct', 1), ('1st', 1), ("it's", 3), ('a', 1), ('bit', 1), ('chilly', 1), ('sunny', 1), ('9', 1), ('34', 1), ('am', 1)])
# 단어별 등장 횟수. 순서는 처음 나온 순서

x = token.texts_to_sequences([text])
print(x)    # [[3, 2, 4, 5, 6, 1, 7, 8, 9, 2, 1, 10, 1, 11, 12, 13]]
# 번호는 이름표라 크기에 의미가 없는데, 그대로 넣으면 모델은 3이 2보다 크다고 계산함 > 원핫 인코딩
print(len(x))       # 1
print(len(x[0]))    # 16

# 원핫 인코딩 방법 3가지
import numpy as np
x = np.array(x) # (1, 16)
x = x.reshape(-1)   # 단어 16개가 각각 샘플이 되도록 (1, 16) > (16,)
print(x.shape)  # (16,)

# 1. pandas
# import pandas as pd
# x = pd.get_dummies(x, dtype=int).values
# print(x.shape)  # (16, 13)

# 2. sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
x = x.reshape(-1, 1)      # OneHotEncoder가 2차원 데이터만 받아서 모양을 맞춰줘야 함
x = ohe.fit_transform(x)
print(x.shape)  # (16, 13)

# 3. tensorflow
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x - 1)   # 라벨이 1부터 시작하므로 인덱스를 하나씩 내려줌
# print(x.shape)  # (16, 13)