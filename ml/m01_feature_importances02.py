from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# feature_importances_ 회귀 : 트리 계열 회귀 모델 4개로 diabetes를 학습하고 컬럼별 중요도 비교
# 1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (442, 10) (442,)

random_state = 253

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    shuffle=True,   # 기본값이 True. y가 연속값이라 stratify는 못 씀
)

# 2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models:
    # 3. 훈련 (sklearn은 compile 없이 fit만)
    model.fit(x_train, y_train)
    print("================", model.__class__.__name__, "================")

    # 4. 평가, 예측
    score = model.score(x_test, y_test)     # 회귀 모델의 score는 r2
    print("r2_score: ", score)
    print(model.feature_importances_)       # 각 컬럼의 중요도. 합은 1

'''
================ DecisionTreeRegressor ================
r2_score:  -0.30981757872596094
[0.04897258 0.01445977 0.21511851 0.08386488 0.07259693 0.07103889
 0.0363423  0.02235448 0.3817944  0.05345726]
================ RandomForestRegressor ================
r2_score:  0.43633746090904924
[0.06390849 0.01054082 0.23587154 0.10759207 0.05157605 0.04612085
 0.05045233 0.01686943 0.34806286 0.06900555]
================ GradientBoostingRegressor ================
r2_score:  0.41712701330464697
[0.0442415  0.01580064 0.20414342 0.13071193 0.03118715 0.04383221
 0.04147993 0.01835395 0.43209203 0.03815723]
================ XGBRegressor ================
r2_score:  0.29077414204954877
[0.0264537  0.0622987  0.12267357 0.09518689 0.06240314 0.05072496
 0.04177285 0.07289746 0.40001872 0.06557004]
'''
# DecisionTree r2가 음수 : 평균값 하나로 예측하는 것보다 못함. 트리 하나가 train에 과적합됨
# 컬럼 순서 : age, sex, bmi, bp, s1~s6(혈액 검사 수치)
# 네 모델 모두 s5(9번째)가 가장 크고 bmi(3번째)가 그다음