import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')

# 여러 개의 모델 한 번에 돌리기(회귀)
# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# all = all_estimators(type_filter='classifier')
# print('all models: ', all)
# print('sk version: ', sk.__version__)   # sk version:  1.9.1
# print('모델의 개수: ', len(all))         # 모델의 개수:  44

all = all_estimators(type_filter='regressor')
print('all models: ', all)
print('sk version: ', sk.__version__)   # sk version:  1.9.1
print('모델의 개수: ', len(all))         # 모델의 개수:  55

for name, model in all:
    try:
        model = model()

        # 3. 훈련
        model.fit(x_train, y_train)

        # 4. 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률: ', acc)

    except:
        print(name, '모델 오류')