from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 중요도 그래프 : xgboost plot_importance로 컬럼별 weight(F score)·gain을 막대그래프로 그리기 (03에 그래프 추가)
# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    stratify=y,
)

# 2. 모델 구성
model = XGBClassifier(random_state=random_state)

# 3. 훈련 (sklearn은 compile 없이 fit만)
model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")

# 4. 평가, 예측
score = model.score(x_test, y_test)     # 분류에서는 acc, 회귀에서는 r2
print("acc: ", score)
print(model.feature_importances_)       # 각 컬럼의 중요도. 합은 1
# 중요도를 보고 컬럼을 빼거나 새로 만들어 성능을 올리는 게 feature engineering

gain_score = model.get_booster().get_score(importance_type='gain')  # get_booster() : sklearn 래퍼 안의 xgboost 원래 모델(Booster)
print("gain score: ", gain_score)   # gain : 그 컬럼으로 분기할 때 손실이 줄어든 양의 평균. 높을수록 그 컬럼의 분기가 손실을 많이 줄임
# importance_type : 'weight'(분기에 쓰인 횟수), 'gain', 'cover', 'total_gain', 'total_cover'
'''
================ XGBClassifier ================
acc:  0.9333333333333333
[0.02430454 0.02472077 0.7376847  0.21328993]
gain score:  {'f0': 0.09280245006084442, 'f1': 0.09439174085855484, 'f2': 2.8167145252227783, 'f3': 0.8144086599349976}
'''
# f0~f3 : numpy로 넣어서 컬럼 이름이 없어 순서대로 f0, f1 ...
# gain을 합이 1이 되게 나누면 feature_importances_와 같음 (2.8167 / 3.8183 = 0.7377). XGB의 feature_importances_ 기본 기준이 gain

import matplotlib.pyplot as plt
from xgboost import plot_importance  # Booster에서 중요도를 꺼내 큰 순서대로 가로 막대그래프로 그림. 막대 끝에 값 표시

# F-Score 그래프
# F score : 그 컬럼이 분기에 쓰인 횟수(weight). 분류 평가지표 F1 score와는 다른 값
# plot_importance(model, importance_type='weight')    # importance_type 기본값이 'weight'
# plt.title("F-score")
# plt.show()

# Gain-Score 그래프
plot_importance(model, importance_type='gain')  # 막대 끝 값이 긴 소수로 나오면 values_format='{v:.2f}'로 자릿수 조절
plt.title("Gain-score")    # 기본 제목 'Feature importance'를 덮어씀
plt.show()