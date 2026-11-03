import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')

# 여러 개의 모델 한 번에 돌리기(분류)
# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델 구성
all = all_estimators(type_filter='classifier')
print('all models: ', all)
print('sk version: ', sk.__version__)   # sk version:  1.9.1
print('모델의 개수: ', len(all))         # 모델의 개수:  44

# all = all_estimators(type_filter='regressor')
# print('all models: ', all)
# print('sk version: ', sk.__version__)   # sk version:  1.9.1
# print('모델의 개수: ', len(all))         # 모델의 개수:  55

for name, model in all:
    # print(name)
    try:
        model = model()

        # 3. 훈련, 평가
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("acc: ", scores,'\n평균 acc: ', round(np.mean(scores), 2))

    except:
        print(name, '모델 오류')