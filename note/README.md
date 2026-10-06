# 스터디 노트

Keras 딥러닝 학습 노트. 날짜별 노트북 한 개 + 그날 실습한 스크립트로 구성됩니다.

| 노트 | 날짜 | 다룬 내용 | 실습 코드 |
|---|---|---|---|
| [day01](day01_0831_keras.ipynb) | 08/31 | 환경 구축, AI 수학 기초, 순전파/역전파, Sequential·Dense, epoch/batch_size | `keras/keras01.py` ~ `keras06_batch.py` |
| [day02](day02_0901_keras.ipynb) | 09/01 | 파이썬 import 방식, 스칼라~텐서, shape, reshape, train/test 분리 | `keras/keras07_matrix.py` ~ `keras09_train_test1.py` |
| [day03](day03_0902_keras.ipynb) | 09/02 | 학습과 추론(NPU), train/test 분리, 시각화, 실전 데이터셋, loss 읽는 법 (작성 중) | `keras/keras09_train_test2.py` ~ `keras11_03_boston_tf.py` |
| [day04](day04_0903_keras.ipynb) | 09/03 | 회귀와 분류, MSE/MAE/RMSE, R², evaluate와 predict, pandas·CSV, 이상치·결측치, 대회 데이터 구조 | `keras/keras12_01_RS_RMSE_boston.py` ~ `keras13_ddarung01.py` |
| [day05](day05_0904_keras.ipynb) | 09/04 | 문자열·경로 다루기, 판다스 기초, 제출 파이프라인, 데이터 누수, 활성화 함수와 ReLU | `keras/keras13_ddarung02_submit.py` ~ `keras14_kaggle_bike1.py` |
| [day06](day06_0907_keras.ipynb) | 09/07 | 딕셔너리, verbose, 검증 데이터(validation), 학습 시간 측정, history 시각화, EarlyStopping | `keras/keras15_verbose.py` ~ `keras20_EarlyStopping5_kaggle_bike.py` |
| [day07](day07_0908_keras.ipynb) | 09/08 | 속성·키 접근, local/global minima, 회귀에서 분류로, stratify, sigmoid, binary_crossentropy와 metrics | `keras/keras21_sigmoid_metrics_cancer.py` |
| [day08](day08_0909_keras.ipynb) | 09/09 | 클래스 인스턴스화·메서드 체이닝, 다중분류, 원핫 인코딩, softmax, argmax | `keras/keras23_softmax1_OneHot_iris.py` ~ `keras23_softmax4_digits.py` |
| [day09](day09_0910_keras.ipynb) | 09/10 | 판다스→넘파이, 이진을 다중으로 풀기, `summary()`와 파라미터 개수, `input_shape`, 스케일링과 그 순서 | `keras/keras24_kaggle_santander.py` ~ `keras28_Scaler10_digits.py` |
| [day10](day10_0911_keras.ipynb) | 09/11 | 스케일링의 목적, 스케일러 4종(MinMax·Standard·MaxAbs·Robust)과 이상치, 모델 저장·불러오기 | `keras/keras29_1_save_model.py` ~ `keras29_4_load_model2.py` |
| [day11](day11_0914_keras.ipynb) | 09/14 | 파일명 문자열 만들기, 가중치 저장·불러오기, ModelCheckpoint, 체크포인트 이력 남기기, Dropout, 함수형 모델 | `keras/keras29_5_save_weights.py` ~ `keras34_function00.py` |
| [day12](day12_0915_keras.ipynb) | 09/15 | GPU 환경 구축(TF 2.9 버전 조합), CPU·GPU 실행 시간 비교, 모델 종류와 데이터 차원, CNN(이미지 수치화, `Conv2D`, MNIST, 이미지 스케일링) | `keras/keras35_gpu_test00.py` ~ `keras36_cnn3_mnist.py` |
| [day13](day13_0916_keras.ipynb) | 09/16 | `Flatten`, `Conv2D`의 filters 인자, padding, strides, MaxPooling | `keras/keras36_cnn1.py` ~ `keras39_MaxPooling_0.py` |
| [day14](day14_0917_keras.ipynb) | 09/17 | Conv2D 파라미터 개수 계산, Global Average Pooling, DNN으로 이미지 처리, 2차원 데이터를 CNN으로 처리 | `keras/keras39_MaxPooling0.py` ~ `keras42_cnn10_digits.py` |
| [day15](day15_0918_keras.ipynb) | 09/18 | ImageDataGenerator와 증폭, fill_mode, 폴더 구조로 x·y 만들기, batch_size와 Iterator, 제너레이터를 모델에 먹이는 법 | `keras/keras44_ImageDataGenerator1.py` ~ `keras44_ImageDataGenerator3_CatDog.py` |
| [day16](day16_0921_keras.ipynb) | 09/21 | train/test 분리가 없는 데이터, `class_mode`, 출력층과 loss의 짝, 타입별 인덱싱, 이미지 한 장으로 예측, `flow`로 증폭 | `keras/keras46_01_save_npy_horse.py` ~ `keras50_flow1.py` |
| [day17](day17_0922_keras.ipynb) | 09/22 | 증폭으로 훈련셋 늘리기, `randint`와 `choice`, 검증셋을 증폭 전에 나누는 이유, 데이터셋마다 다른 x·y 형태 | `keras/keras50_flow2_next.py` ~ `keras51_augment4_cifar100.py` |
| [day18](day18_0923_keras.ipynb) | 09/23 | `learning_rate` 직접 지정, `ReduceLROnPlateau`, RNN 도입(순환·timesteps·입력 차원) | `keras/keras52_optimizer01_california.py` ~ `keras53_ReduceLR15_man_woman.py` |
| [day19](day19_0928_keras.ipynb) | 09/28 | `split_x`로 수열 자르기, 콤마 인덱싱 | `keras/keras55_LSTM2_scale.py` ~ `keras56_split1.py` |
| [day20](day20_0929_keras.ipynb) | 09/29 | 모델별 입력·출력 차원, `return_sequences`와 RNN 쌓기, 가중치 초기화 난수, `drop`, 예측 시점과 데이터 누수 | `keras/keras56_split3.py` ~ `keras58_kaggle_jena1.py` |
| [day21](day21_0930_keras.ipynb) | 09/30 | `Bidirectional` | `keras/keras58_kaggle_jena2.py` ~ `keras59_Bidirectional3_jena.py` |
| [day21 RAG](day21_0930_rag.ipynb) | 09/30 | 채팅 모델과 `AIMessage`, `PromptTemplate`, LCEL 체인, 출력 파서 | `RAG/rag01_add_api_keys.py` ~ `rag09_output_parser02.py` |
| [day22](day22_1001_keras.ipynb) | 10/01 | 임베딩(토큰화 → 정수 인코딩 → 임베딩), 원핫의 한계, `Embedding` 층과 파라미터, `Tokenizer`, 원핫 전 모양, 패딩 | `keras2/keras60_Tokenizer1.py` ~ `keras61_Embedding04_important.py` |
| [day22 RAG](day22_1001_rag.ipynb) | 10/01 | 임베딩 모델(`OpenAIEmbeddings`, `embed_query`, 차원), 코사인 유사도 | `RAG/rag10_Embedding01.py` ~ `rag10_Embedding02.py` |
| [day23](day23_1002_keras.ipynb) | 10/02 | `reuters`·`imdb` 텍스트 데이터셋, 패딩 길이 정하기, 텍스트 분류 모델, 모델별 차원(Embedding 포함), 텐서플로와 넘파이, `sparse_categorical_crossentropy`, DNN·CNN을 RNN으로, `Reshape` 층 | `keras2/keras62_1_reuters.py` ~ `keras65_Reshape2.py` |
| [day23 RAG](day23_1002_rag.ipynb) | 10/02 | 의미 검색(키워드 검색과 비교), 벡터 저장소 Chroma·FAISS | |
| [day24](day24_1006_keras.ipynb) | 10/06 | RNN 데이터를 `Reshape`로 접어 `Conv2D`에, `Conv1D`, RNN 모델을 Conv1D로(`GlobalAveragePooling1D`) | `keras2/keras66_jena_CNN.py` ~ `keras67_Conv1D_3_jena.py` |
| [day24 RAG](day24_1006_rag.ipynb) | 10/06 | 문서 분할(`TextLoader`, `RecursiveCharacterTextSplitter`), 여러 파일 불러오기, 청크 임베딩(`embed_query`·`embed_documents`), Chroma 저장·불러오기·`get`, 유사도 검색, Retriever | `RAG/rag11_Chroma01_save.py` ~ `rag12_Chroma03_save.py` |

## 주제별 찾아보기

분류 → 키워드 묶음 순으로 찾습니다. 묶음 안은 날짜순이고, 참조를 누르면 그날 노트가 열립니다

[데이터 준비](#데이터-준비) · [모양·차원](#모양차원) · [모델 층](#모델-층) · [컴파일·훈련](#컴파일훈련) · [평가·예측](#평가예측) · [원리](#원리) · [파이썬·넘파이·판다스](#파이썬넘파이판다스) · [환경·저장](#환경저장) · [LangChain](#langchain)

### 데이터 준비

#### 불러오기
- 실전 데이터셋 불러오기 (`load_` vs `fetch_`, Bunch 객체) — [day03 §4](day03_0902_keras.ipynb)
- sklearn과 케라스의 반환 순서 차이 — [day03 §4](day03_0902_keras.ipynb)
- CSV를 뷰어에서 고치면 안 되는 이유 — [day04 §5](day04_0903_keras.ipynb)
- `pd.read_csv`와 `index_col=0` — [day04 §5](day04_0903_keras.ipynb)
- `mnist.load_data()` — train/test로 나눠 반환, `(60000, 28, 28)`, `plt.imshow`로 확인 — [day12 §4](day12_0915_keras.ipynb)

#### 라벨 확인
- 분류 데이터를 받으면 먼저 라벨 확인 — [day07 §3](day07_0908_keras.ipynb)
- 라벨별 개수 세기 (`np.unique` / `value_counts`) — [day07 §3](day07_0908_keras.ipynb)
- 클래스 불균형을 확인해야 하는 이유 — [day07 §3](day07_0908_keras.ipynb)
- `pd.value_counts`는 pandas 3.0에서 삭제 → `pd.Series(y).value_counts()` — [day12 §4](day12_0915_keras.ipynb)

#### 결측치·이상치
- 이상치(outlier)의 기준은 상대적 — [day04 §6](day04_0903_keras.ipynb)
- 결측치 처리 3가지 (`dropna` / `fillna` / 앞뒤 채우기) — [day04 §6](day04_0903_keras.ipynb)
- train은 `dropna()`, test는 `fillna()`인 이유 — [day04 §6](day04_0903_keras.ipynb)
- `df.isna()` / `df.isnull()`은 같은 함수 — [day04 §6](day04_0903_keras.ipynb)
- `dropna()`가 행 단위로 지운다는 것 — [day04 §6 cf)](day04_0903_keras.ipynb)

#### x·y 나누기
- x / y 분리 (`drop(axis=1)`, `df['count']`) — [day04 §7](day04_0903_keras.ipynb)
- test에 없는 컬럼은 x에 둘 수 없다 — [day05 §2](day05_0904_keras.ipynb)
- 데이터 누수(leakage) — `casual` + `registered` = `count` — [day05 §2 cf)](day05_0904_keras.ipynb)

#### train·val·test 나누기
- train / test 분리, shuffle이 필요한 이유 — [day02 §4](day02_0901_keras.ipynb)
- 나누는 방법 3가지 (직접/슬라이싱/`train_test_split`) — [day03 §2](day03_0902_keras.ipynb)
- `random_state`를 고정하는 이유 — [day03 §2](day03_0902_keras.ipynb)
- 시계열은 섞으면 안 되는 이유 — [day03 §2](day03_0902_keras.ipynb)
- validation / x_test / test.csv 3단계 — [day04 §7](day04_0903_keras.ipynb)
- 검증 데이터(validation)가 필요한 이유 — [day06 §2](day06_0907_keras.ipynb)
- val과 test를 나누는 이유 — [day06 §2](day06_0907_keras.ipynb)
- 검증셋 만드는 4가지 방법 — [day06 §3](day06_0907_keras.ipynb)
- `validation_split` vs `validation_data` — [day06 §3](day06_0907_keras.ipynb)
- `stratify=y`로 라벨 비율 유지하며 분할 — [day07 §3](day07_0908_keras.ipynb)
- `stratify`에 원핫(2차원)을 넘겨도 되는가 — [day08 §6 cf)](day08_0909_keras.ipynb)

#### 스케일링
- 특성 스케일이 제각각이면 생기는 문제 — [day03 §5](day03_0902_keras.ipynb)
- 스케일링이 필요한 이유, 큰 값이 loss를 지배함 — [day09 §4](day09_0910_keras.ipynb)
- `scaler.fit()` / `scaler.transform()`의 역할 구분 — [day09 §4](day09_0910_keras.ipynb)
- 부동소수점 오차 `1.0000000000000002` — [day09 §4 cf)](day09_0910_keras.ipynb)
- 스케일링은 분리 뒤에, `fit`은 train에만 — [day09 §5](day09_0910_keras.ipynb)
- 스케일러를 전체에 `fit`하면 생기는 누수 — [day09 §5](day09_0910_keras.ipynb)
- x_test가 0~1을 벗어나는 것은 정상 — [day09 §5](day09_0910_keras.ipynb)
- `validation_split`에 남아 있는 미세한 누수 — [day09 §5 cf)](day09_0910_keras.ipynb)
- 스케일링의 목적 — 크기만으로 우위를 갖지 못하게 — [day10 §1](day10_0911_keras.ipynb)
- 스케일링은 범위를 맞추는 것이지 분포를 맞추는 게 아님 — [day10 §1](day10_0911_keras.ipynb)
- 이상치 처리가 먼저, 스케일링이 나중 — [day10 §1](day10_0911_keras.ipynb)
- 이미지 스케일링 `/255.`(0~1)과 `(x-127.5)/127.5`(-1~1) — [day12 §4](day12_0915_keras.ipynb)
- 스케일링은 창을 만들기 전 2차원일 때 — [day20 §4 cf)](day20_0929_keras.ipynb)

#### 스케일러 종류
- MinMaxScaler 공식 `(x-MIN)/(MAX-MIN)` — [day09 §4](day09_0910_keras.ipynb)
- MinMaxScaler는 이상치에 가장 약하다 — [day10 §1](day10_0911_keras.ipynb)
- StandardScaler — 평균 0, 표준편차 1 — [day10 §2](day10_0911_keras.ipynb)
- MinMaxScaler와 StandardScaler의 보완 관계 — [day10 §2](day10_0911_keras.ipynb)
- 이상치가 있으면 StandardScaler 쪽 — [day10 §2](day10_0911_keras.ipynb)
- StandardScaler는 정규분포로 만들어주지 않는다 — [day10 §2 cf)](day10_0911_keras.ipynb)
- MaxAbsScaler — 최대 절댓값으로 나누기, 0과 부호 유지 — [day10 §3](day10_0911_keras.ipynb)
- 0에서 먼 컬럼은 MaxAbs가 범위를 거의 못 씀 — [day10 §3](day10_0911_keras.ipynb)
- RobustScaler — 중앙값과 IQR로 나누기, 이상치에 강함 — [day10 §4](day10_0911_keras.ipynb)
- Robust도 이상치를 없애지는 않는다 — [day10 §4](day10_0911_keras.ipynb)
- 결과가 좋다 ≠ 스케일러가 좋다 (알게 되는 건 내 데이터) — [day10 §4 cf)](day10_0911_keras.ipynb)

#### 원핫 인코딩
- 원핫 인코딩이 필요한 이유, 라벨 간 거리 — [day08 §2](day08_0909_keras.ipynb)
- 원핫과 임베딩의 차이 — [day08 §2 cf)](day08_0909_keras.ipynb)
- 원핫 만드는 3가지 방법과 각각의 주의점 — [day08 §3](day08_0909_keras.ipynb)
- `to_categorical`은 라벨이 0부터여야 함 — [day08 §3](day08_0909_keras.ipynb)
- `LabelEncoder`는 원핫이 아니라 그 전 단계 — [day08 §3 cf)](day08_0909_keras.ipynb)
- 원핫 인코딩 위치 — 증폭본을 붙인 뒤 한 번만, test는 다시 학습시키지 않음 — [day17 §3-2](day17_0922_keras.ipynb)
- 원핫 전 모양 — 단어를 샘플 축으로, 도구마다 받는 차원 (`get_dummies` 1차원, `OneHotEncoder` 2차원) — [day22 §2-1](day22_1001_keras.ipynb)

#### 이미지 폴더·제너레이터
- `ImageDataGenerator` — 이미지를 수치화하고 변형해서 늘리는(증폭) 도구 — [day15 §1](day15_0918_keras.ipynb)
- 폴더 구조로 x와 y 만들기 — 하위 폴더 이름이 라벨이 됨 — [day15 §1-4](day15_0918_keras.ipynb)
- `batch_size`와 Iterator — 인덱스는 이미지 번호가 아니라 배치 번호 — [day15 §1-5](day15_0918_keras.ipynb)
- 제너레이터를 모델에 먹이는 두 가지 방법 (통째로 꺼내기 / 그대로 넘기기) — [day15 §1-6](day15_0918_keras.ipynb)
- 제너레이터에서 검증 데이터 나누기 — 비율 지정이 통하지 않는 이유 — [day15 §1-7](day15_0918_keras.ipynb)
- 이름이 같은 `batch_size` 두 개 — 꺼내는 단위 vs 갱신 단위 — [day15 §1 cf)](day15_0918_keras.ipynb)
- 평가용 데이터는 섞지 않음 — 예측 순서와 정답 순서 — [day15 §1 cf)](day15_0918_keras.ipynb)
- train/test 분리가 없는 데이터 — 상위 폴더를 넘겨 통째로 읽고 `train_test_split` — [day16 §1](day16_0921_keras.ipynb)
- `batch_size`가 작으면 정렬된 상태에서 잘려 뒤쪽 클래스가 통째로 빠짐 — [day16 §1-1 cf)](day16_0921_keras.ipynb)
- `class_mode` 3종 (`binary`/`categorical`/`sparse`)과 y의 형태 — [day16 §1-3](day16_0921_keras.ipynb)
- `target_size` 정하기 — 디테일·원본 크기·메모리 — [day16 §1-7](day16_0921_keras.ipynb)
- `flow`와 `flow_from_directory` — 배열을 받느냐 폴더를 받느냐 — [day16 §4-1](day16_0921_keras.ipynb)

#### 데이터 증폭
- 증폭 옵션 8가지 (`rescale`, flip, shift, `rotation_range`, `zoom_range`, `shear_range`, `fill_mode`) — [day15 §1-1](day15_0918_keras.ipynb)
- `fill_mode` 4종 — 변형으로 생긴 빈 공간을 채우는 방법 — [day15 §1-2](day15_0918_keras.ipynb)
- 훈련 데이터만 증폭하고 테스트는 스케일링만 하는 이유 — [day15 §1-3](day15_0918_keras.ipynb)
- 증폭 이터레이터는 꺼낼 때마다 새 이미지 — [day16 §4 cf)](day16_0921_keras.ipynb)
- 데이터 증폭 — 원본에서 뽑아 변형한 뒤 원본에 이어붙임 — [day17 §1](day17_0922_keras.ipynb)
- 증폭 절차 5단계 — 번호 뽑기 → 꺼내기 → 변형 → 이어붙이기 — [day17 §1-1](day17_0922_keras.ipynb)
- 증폭 옵션 — 데이터에 있을 법한 변형만. 위아래 반전은 방향 없는 데이터에만 — [day17 §1-3](day17_0922_keras.ipynb)
- 검증셋은 증폭 전에 나눔 — 검증 비율 지정은 뒷부분을 떼어가므로 전부 증폭본이 됨 — [day17 §1 cf)](day17_0922_keras.ipynb)

#### 시계열 자르기
- 시계열 데이터는 y가 없음 — 앞 몇 개로 다음 하나를 맞히도록 잘라서 만듦 — [day18 §4](day18_0923_keras.ipynb)
- 자르면 행이 `전체 길이 - timesteps`로 줄어듦 — [day18 §4-1](day18_0923_keras.ipynb)
- `timesteps` 선택 — 며칠치를 볼지 정하는 순간 문제가 정의됨 — [day18 §4-2](day18_0923_keras.ipynb)
- `split_x` — 수열을 `size` 크기로 잘라 2차원 배열로 만드는 함수 — [day19 §1](day19_0928_keras.ipynb)
- 자르기만 하면 행이 `len - size + 1`, y까지 떼면 `len - timesteps` — [day19 §1-2](day19_0928_keras.ipynb)
- 예측 시점을 정하면 x와 y를 그만큼 어긋나게 자름 — [day20 §4](day20_0929_keras.ipynb)
- jena 실습 순서 — 정답 빼두기 → 어긋나게 자르기 → 스케일링 → 창 — [day20 §4-1](day20_0929_keras.ipynb)
- 정답 구간의 앞 구간도 빼야 함 — 그 행들의 정답이 정답 구간 안에 있음 — [day20 §4-2](day20_0929_keras.ipynb)
- 창 개수에서 필요한 행 수 거꾸로 구하기 — `창 길이 + 창 개수 - 1` — [day20 §4-3](day20_0929_keras.ipynb)
- 창을 만들면 데이터가 창 길이배로 늘어남 — 5.86 GB — [day20 §4 cf)](day20_0929_keras.ipynb)

#### 텍스트 (토큰화·패딩)
- 임베딩 — 토큰화 → 정수 인코딩 → 임베딩, 수치화는 정수 인코딩에서 이미 됨 — [day22 §1](day22_1001_keras.ipynb)
- 원핫의 한계 — 칸 대부분이 0, 단어 수만큼 칸이 늘어남 > 임베딩. 패딩의 0은 남음 — [day22 §1-1](day22_1001_keras.ipynb)
- `Tokenizer` — `fit_on_texts`·`word_index`·`texts_to_sequences`, 많이 나온 단어가 앞 번호 — [day22 §2](day22_1001_keras.ipynb)
- 패딩 `pad_sequences` — 앞에 채우는 이유(RNN은 마지막 칸 결과), `maxlen`, `truncating` — [day22 §3](day22_1001_keras.ipynb)
- `reuters`·`imdb` — 정수 인코딩된 텍스트 데이터셋, `num_words`, 특수 번호 0·1·2, `test_split`은 reuters만 — [day23 §1](day23_1002_keras.ipynb)
- 패딩 길이 정하기 — 최대·평균 길이를 보고 `maxlen` — [day23 §1-1](day23_1002_keras.ipynb)
- `load_data(maxlen=)`은 빼고 `pad_sequences(maxlen=)`은 자름 — [day23 §1 cf)](day23_1002_keras.ipynb)

### 모양·차원

#### 배열 모양
- 스칼라 / 벡터 / 행렬 / 텐서 — [day02 §1](day02_0901_keras.ipynb)
- shape 읽는 법 — [day02 §2](day02_0901_keras.ipynb)
- reshape와 `-1` — [day02 §3](day02_0901_keras.ipynb)
- reshape vs 전치(`.T`) 차이 — [day02 §3 cf)](day02_0901_keras.ipynb)
- 이미지 수치화 `(장수, 세로, 가로, 채널)` — [day12 §4](day12_0915_keras.ipynb)
- 데이터셋마다 다른 x·y 형태 — 흑백은 3차원·`(N,)`, 컬러는 4차원·`(N, 1)` — [day17 §3](day17_0922_keras.ipynb)

#### 모델별 입력 차원
- `input_dim`은 열(특성) 개수 — [day02 §3](day02_0901_keras.ipynb)
- `input_dim` vs `input_shape`, data shape → input shape — [day09 §3](day09_0910_keras.ipynb)
- DNN·RNN·CNN과 입력 데이터 차원 (2·3·4차원) — [day12 §3](day12_0915_keras.ipynb)
- 모델별 입력 차원 — DNN 2차원 / RNN 3차원 / CNN 4차원 — [day18 §3-3](day18_0923_keras.ipynb)
- `input_shape=(3, 1)`과 `input_length=3, input_dim=1`은 같은 뜻 — [day18 §3-3](day18_0923_keras.ipynb)
- 모델별 입력·출력 차원 한눈에 — `input_shape`은 데이터 구조보다 하나 적음 — [day20 §1](day20_0929_keras.ipynb)
- 모델별 차원에 Embedding 추가 — x 2 → output 3, Embedding만 차원이 늘어남 — [day23 §2-1](day23_1002_keras.ipynb)

#### 모델에 맞게 바꾸기
- DNN으로 이미지 처리 — `reshape(-1, 28 * 28)`, 컬러는 `32 * 32 * 3` — [day14 §3](day14_0917_keras.ipynb)
- 표 데이터를 reshape해서 `Conv2D`에 넣기 — 가능한 조건과 성능을 장담 못하는 이유 — [day14 §4](day14_0917_keras.ipynb)
- 표 데이터를 CNN으로 바꿀 때 자주 틀리는 곳 (reshape 곱, input_shape, 작은 입력, 출력층 활성화) — [day14 §4 cf)](day14_0917_keras.ipynb)
- 3·4차원 출력에 바로 붙인 `Dense` — 마지막 축에만 적용돼 차원이 남고 `fit`에서 에러 — [day23 §2 cf)](day23_1002_keras.ipynb)
- DNN·CNN을 RNN으로 — `(N, 특성)` → `(N, 특성, 1)`, `(N, 세로, 가로, 채널)` → `(N, 세로, 가로 × 채널)` — [day23 §5](day23_1002_keras.ipynb)
- `Reshape` 층 — 모델 안에서 모양 변경, 파라미터 0, 원소 개수 유지, `LSTM` 앞에서는 timesteps 축을 정함 — [day23 §6](day23_1002_keras.ipynb)
- RNN 데이터를 `Conv2D`로 — `Reshape((12, 12, 13))`로 144칸을 접음, 세로 이웃이 12칸 떨어진 시점이라 순서가 섞임 — [day24 §1](day24_1006_keras.ipynb)

### 모델 층

#### 모델 기본
- 코딩 4단계 (데이터 → 모델 → compile/fit → evaluate/predict) — [day01 §5](day01_0831_keras.ipynb)
- Sequential / Layer / Node(Unit) — [day01 §6](day01_0831_keras.ipynb)
- Dense(완전연결층) — [day01 §7](day01_0831_keras.ipynb)
- 파라미터(w·b) 개수 세는 공식 — [day09 §2](day09_0910_keras.ipynb)
- `model.summary()` 읽는 법, `Output Shape`의 `None` — [day09 §2, §3](day09_0910_keras.ipynb)
- 함수형 모델 `Input` / `Model` — `Sequential`과 연결 방식 비교 — [day11 §4](day11_0914_keras.ipynb)

#### 활성화 함수
- 활성화 함수란, 기본값이 `linear`라는 것 — [day05 §3](day05_0904_keras.ipynb)
- ReLU가 비선형성을 주는 이유, 기울기 소실 방지 — [day05 §3](day05_0904_keras.ipynb)
- `activation='relu'` 넣는 위치 — [day05 §3](day05_0904_keras.ipynb)

#### Dropout·BatchNormalization
- `Dropout` — 배치마다 랜덤하게 노드를 꺼서 과적합 줄이기 — [day11 §3](day11_0914_keras.ipynb)
- `BatchNormalization`과 `Dropout` 순서 — 꺼진 상태의 통계를 배우면 예측 때와 안 맞음 — [day17 §3 cf)](day17_0922_keras.ipynb)

#### CNN
- `Conv2D` — `kernel_size`(자르는 크기)와 `filters`(필터 개수 = 출력 채널 수) — [day12 §4](day12_0915_keras.ipynb)
- `Conv2D` 층별 Output Shape·Param # 계산 — [day12 §4](day12_0915_keras.ipynb)
- `Flatten` — Conv 출력을 한 줄로 펴서 `Dense`에 연결, 파라미터 0 — [day13 §1](day13_0916_keras.ipynb)
- `padding` — `valid`/`same`, 가장자리 정보와 크기 유지 — [day13 §3](day13_0916_keras.ipynb)
- `strides` — 필터 이동 칸수, 출력 크기 축소와 정보 손실 — [day13 §4](day13_0916_keras.ipynb)
- `MaxPooling2D` — 구역별 최댓값만 남겨 크기를 절반으로, 파라미터 0 — [day13 §5](day13_0916_keras.ipynb)
- Conv2D 파라미터 개수 `f × (n × n × c + b)`, c는 앞 층의 filters — [day14 §1](day14_0917_keras.ipynb)
- `GlobalAveragePooling2D` — 특징맵마다 평균 하나, Flatten 대신 써서 Dense 파라미터 감소 — [day14 §2](day14_0917_keras.ipynb)
- `AveragePooling2D`와 `GlobalAveragePooling2D`의 차이 — [day14 §2 cf)](day14_0917_keras.ipynb)
- `Conv1D` — 커널이 시간 축으로만 이동, `kernel_size`는 `n`, 특성이 채널 자리, 파라미터 `filters × (n × 특성 + 1)` — [day24 §2](day24_1006_keras.ipynb)
- RNN 모델을 `Conv1D`로 — 같은 `input_shape`, 3차원 출력을 `Flatten`·`GlobalAveragePooling1D`로 편 뒤 `Dense` — [day24 §2-1](day24_1006_keras.ipynb)

#### RNN·LSTM·GRU
- RNN — 한 칸씩 넣으며 앞 결과를 다음 칸에 함께 넣는 순환 구조 — [day18 §3](day18_0923_keras.ipynb)
- `timesteps` — 시퀀스를 몇 칸으로 끊을지. `timestamp`와 다른 말 — [day18 §3-2](day18_0923_keras.ipynb)
- RNN 출력은 2차원이라 `Flatten` 없이 `Dense`에 바로 연결 — [day18 §3-3](day18_0923_keras.ipynb)
- RNN의 가중치 두 벌 — 입력 가중치와 순환 가중치 — [day18 §3-4](day18_0923_keras.ipynb)
- RNN 파라미터 개수 `units × (feature + units + 1)` — timesteps는 들어가지 않음 — [day18 §3-4](day18_0923_keras.ipynb)
- 순환 가중치는 앞 칸 h에 곱해짐 — `(units, units)`인 이유 — [day18 §3-4 계산식](day18_0923_keras.ipynb)
- 옛날 입력일수록 순환 가중치가 여러 번 곱해짐 — 긴 시퀀스에 약한 이유 — [day18 §3-4](day18_0923_keras.ipynb)
- LSTM — cell state라는 기억 전용 통로를 따로 둬서 앞부분을 덜 잊음 — [day18 §3-5](day18_0923_keras.ipynb)
- LSTM 파라미터는 SimpleRNN의 4배, GRU는 3배 — [day18 §3-5](day18_0923_keras.ipynb)
- RNN 계열 층은 activation이 이미 들어 있음 — 따로 넣지 않음 — [day19 §2 cf)](day19_0928_keras.ipynb)
- `return_sequences` — 마지막 칸만 내보낼지 칸마다 내보낼지 — [day20 §2](day20_0929_keras.ipynb)
- RNN을 쌓으려면 앞 층에 `return_sequences=True`, 마지막 층은 끔 — [day20 §2](day20_0929_keras.ipynb)
- 쌓으면 앞 층의 `units`가 다음 층의 `feature`가 됨 — [day20 §2-1](day20_0929_keras.ipynb)
- 쌓았을 때 영향 범위는 삼각형 — 같은 칸 안은 Dense처럼 전부, 칸 사이는 앞에서 뒤로만 — [day20 §2-2](day20_0929_keras.ipynb)
- `Bidirectional` — RNN을 순방향·역방향 두 벌로 돌려 이어 붙이는 래퍼 레이어 — [day21 §1](day21_0930_keras.ipynb)
- `Bidirectional` 파라미터·출력은 한 방향의 2배 — [day21 §1-1](day21_0930_keras.ipynb)
- `input_shape`는 안쪽 RNN이 아니라 `Bidirectional`에 — [day21 §1 cf)](day21_0930_keras.ipynb)

#### Embedding
- `Embedding` 층 — `(N, 문장 길이)` > `(N, 문장 길이, output_dim)`, 파라미터 `input_dim × output_dim` — [day22 §1-2](day22_1001_keras.ipynb)
- 텍스트 분류 모델 — `Embedding` 뒤에 LSTM·GRU·Bidirectional·Flatten으로 2차원 — [day23 §2](day23_1002_keras.ipynb)

### 컴파일·훈련

#### 출력층·loss 짝
- sigmoid 함수와 확률로 읽기 — [day07 §4](day07_0908_keras.ipynb)
- 출력층 `activation='sigmoid'` (이진분류) — [day07 §4](day07_0908_keras.ipynb)
- `loss='binary_crossentropy'` — [day07 §5](day07_0908_keras.ipynb)
- `metrics`와 loss의 차이, 정확도를 loss로 못 쓰는 이유 — [day07 §5](day07_0908_keras.ipynb)
- softmax의 합이 항상 1인 이유 — [day08 §4](day08_0909_keras.ipynb)
- 출력층 `activation='softmax'` (다중분류) — [day08 §4](day08_0909_keras.ipynb)
- 출력층 노드 수·활성화·loss·`class_mode`의 짝 — [day16 §1-4](day16_0921_keras.ipynb)
- 노드 1개에 softmax를 못 쓰는 이유 — 항상 1.0이라 기울기가 0 — [day16 §1-5](day16_0921_keras.ipynb)
- `sparse_categorical_crossentropy` — y 원핫 없이 라벨 그대로, 정답에 argmax 하지 않음 — [day23 §4](day23_1002_keras.ipynb)

#### fit 설정
- epoch vs batch_size vs iteration — [day01 §10](day01_0831_keras.ipynb)
- `verbose` — 출력 옵션이고 결과에 영향 없음 — [day06 §1](day06_0907_keras.ipynb)
- 학습 시간 재기 (`time.time()`) — [day06 §5](day06_0907_keras.ipynb)
- `fit`이 반환하는 History 객체와 `hist.history` — [day06 §6](day06_0907_keras.ipynb)

#### learning_rate
- `learning_rate` — 보폭. 작으면 느리고 크면 바닥에서 진동 — [day17 §4-3](day17_0922_keras.ipynb)
- `learning_rate` 직접 지정 — 문자열은 기본값, 객체는 값을 정할 수 있음 — [day18 §1-1](day18_0923_keras.ipynb)
- `lr`이 너무 크면 찍기 수준 — 다중분류 loss가 `ln(클래스 수)` 근처면 학습 안 된 것 — [day18 §1 cf)](day18_0923_keras.ipynb)

#### 콜백
- 콜백(callback)과 `EarlyStopping` — [day06 §7](day06_0907_keras.ipynb)
- 과적합 지점에서 자동으로 멈추기 — [day06 §7](day06_0907_keras.ipynb)
- `restore_best_weights`가 필요한 이유 — [day06 §7](day06_0907_keras.ipynb)
- `patience`는 epochs의 10~20% 선 — [day06 §7](day06_0907_keras.ipynb)
- Early Stopping이 최고 모델을 보장하지 않는 이유 — [day07 §1](day07_0908_keras.ipynb)
- 체크포인트 파일명에 `{epoch}`·`{val_loss}` 넣기 — [day11 §0](day11_0914_keras.ipynb)
- `ModelCheckpoint` — 훈련 중 최고 지점을 파일로 저장 — [day11 §2](day11_0914_keras.ipynb)
- `restore_best_weights`(메모리) vs 체크포인트(파일) — [day11 §2](day11_0914_keras.ipynb)
- `ReduceLROnPlateau` — `val_loss`가 안 좋아지면 `lr`을 줄여주는 콜백 — [day18 §2](day18_0923_keras.ipynb)
- `ReduceLROnPlateau`와 `EarlyStopping`의 `patience` — 줄여볼 기회를 주려면 차이를 둠 — [day18 §2-2](day18_0923_keras.ipynb)

#### 튜닝
- 하이퍼파라미터란 — [day01 §8, §9](day01_0831_keras.ipynb)
- 층·epoch만 키운다고 좋아지지 않는다 (튜닝 기록) — [day05 §4](day05_0904_keras.ipynb)
- 튜닝 우선순위 — 데이터 > 짝 > 구조 > 규제 > 학습 설정 — [day17 §4-7](day17_0922_keras.ipynb)

### 평가·예측

#### evaluate
- matplotlib으로 회귀선 그려 확인하기 — [day03 §3](day03_0902_keras.ipynb)
- `fit`의 loss와 `evaluate`의 loss 차이 — [day03 §5](day03_0902_keras.ipynb)
- loss 값을 데이터셋끼리 비교할 수 없는 이유 — [day03 §5](day03_0902_keras.ipynb)
- `evaluate` 안에 `predict`가 포함된다 — [day04 §4](day04_0903_keras.ipynb)
- MSE를 구하는 두 경로 (케라스 / sklearn) — [day04 §4](day04_0903_keras.ipynb)
- `metrics`를 주면 `evaluate`가 리스트를 반환 — [day07 §5](day07_0908_keras.ipynb)
- `evaluate`에 y를 안 주면 나는 에러 — [day08 §5 cf)](day08_0909_keras.ipynb)

#### 회귀 지표
- MSE가 "제곱"인 이유, 경사 하강법 갱신식 — [day03 cf)](day03_0902_keras.ipynb)
- 케라스에 RMSE가 없어 직접 함수로 만드는 것 — [day04 §0, §2](day04_0903_keras.ipynb)
- MAE 대신 MSE를 쓰는 이유 (기울기·미분 가능성) — [day04 §2](day04_0903_keras.ipynb)
- RMSE는 단위만 되돌린다 (이상치 취약성은 남음) — [day04 §2](day04_0903_keras.ipynb)
- R²(결정계수) 읽는 법, 0이 기준선인 이유 — [day04 §3](day04_0903_keras.ipynb)
- R²가 지나치게 높을 때 의심할 것 — [day04 §3](day04_0903_keras.ipynb)

#### 분류 지표
- 분류에 R²·RMSE를 쓰면 안 되는 이유 — [day07 §5 cf)](day07_0908_keras.ipynb)
- 불균형 데이터에서 accuracy가 쓸모없는 이유 — [day09 §1](day09_0910_keras.ipynb)

#### 예측
- `argmax`로 확률을 라벨로 되돌리기 — [day08 §5](day08_0909_keras.ipynb)
- 예측값을 라벨로 되돌리기 (`round` / `argmax`) — [day16 §1-6](day16_0921_keras.ipynb)
- 이미지 한 장으로 예측 — `load_img` → `img_to_array` → 차원 증가 — [day16 §3](day16_0921_keras.ipynb)
- 예측할 이미지도 학습 때와 같은 전처리 — 스케일이 다르면 에러 없이 예측값만 달라짐 — [day16 §3 cf)](day16_0921_keras.ipynb)
- 예측할 입력도 훈련 x와 같은 3차원으로 — [day18 §4-3](day18_0923_keras.ipynb)

#### 대회 제출
- `x_test`(내 채점용)와 `test_csv`(제출용)의 차이 — [day04 §7](day04_0903_keras.ipynb)
- `to_csv()`로 제출 파일 저장 — [day05 §1](day05_0904_keras.ipynb)
- 대회 제출 파이프라인 5단계 — [day05 §1](day05_0904_keras.ipynb)
- 예측값을 답안지 컬럼에 대입하기 — [day05 §1](day05_0904_keras.ipynb)
- softmax 결과에서 제출할 확률 열 고르기 `[:, 1]` — [day09 §1](day09_0910_keras.ipynb)

### 원리

#### 학습
- 학습이 "반복"인 이유, 경사 하강법 — [day01 §3](day01_0831_keras.ipynb)
- 순전파(forward) / 역전파(backward) — [day01 §3-3](day01_0831_keras.ipynb)
- AI > ML > DL > LLM 포함 관계 — [day01 §4](day01_0831_keras.ipynb)
- 학습(training) vs 추론(inference), NPU — [day03 §1](day03_0902_keras.ipynb)
- 로스 = 에러 = 오차 = 코스트 — [day04 §1](day04_0903_keras.ipynb)

#### 경사하강법
- local minima와 global minima — [day07 §1](day07_0908_keras.ipynb)
- loss 0이 목표가 아닌 이유 — [day07 §1](day07_0908_keras.ipynb)
- 경사하강법 — loss가 낮아지는 쪽으로 조금씩 내려가는 방법 — [day17 §4](day17_0922_keras.ipynb)
- w와 loss의 관계 — U자의 바닥(꼭짓점)이 최적의 w. 변곡점과 다름 — [day17 §4-1](day17_0922_keras.ipynb)
- 갱신식 — 미분은 방향만 알려주고 거리는 알려주지 않음 — [day17 §4-2](day17_0922_keras.ipynb)
- 지역 최저점 — 출발 위치에 따라 다른 곳에 도착. 튜닝이 필요한 이유 — [day17 §4-4](day17_0922_keras.ipynb)
- 자동 미분 — 역전파를 손으로 쓰지 않는 이유, BPTT — [day17 §4-5](day17_0922_keras.ipynb)

#### 회귀와 분류
- 회귀와 분류의 차이 — [day04 §1](day04_0903_keras.ipynb)
- 이진분류 vs 다중분류 — [day07 §2](day07_0908_keras.ipynb)
- 다중분류 (라벨 3개 이상) — [day08 §1](day08_0909_keras.ipynb)
- 이진분류를 다중분류로 풀 수 있다 — [day09 §1](day09_0910_keras.ipynb)

#### 과적합
- 과적합(overfitting)이 생기는 이유 — [day02 §4](day02_0901_keras.ipynb)
- loss/val_loss 그래프로 과적합 읽기 — [day06 §6](day06_0907_keras.ipynb)

### 파이썬·넘파이·판다스

#### 파이썬 기본
- REPL vs 스크립트 실행 — [day01 §0](day01_0831_keras.ipynb)
- `from A import B`를 쓰는 이유 — [day02 §0](day02_0901_keras.ipynb)
- 함수 vs 클래스, PEP8 이름 규칙 — [day02 §0](day02_0901_keras.ipynb)
- 튜플 언패킹 `(a, b), (c, d) = ...` — [day03 §0](day03_0902_keras.ipynb)
- `def` / `return` 기본 문법 — [day04 §0](day04_0903_keras.ipynb)
- 딕셔너리(dict) — 키로 값 꺼내기, `.keys()` / `.get()` — [day06 §0](day06_0907_keras.ipynb)
- 속성 접근 `.` vs 키 접근 `[]`, `Bunch`가 둘 다 되는 이유 — [day07 §0](day07_0908_keras.ipynb)
- `type()`으로 자료형 확인하기 — [day07 §0 cf)](day07_0908_keras.ipynb)
- 클래스 인스턴스화 `Class()` vs 클래스 자체 — [day08 §0](day08_0909_keras.ipynb)
- 메서드 체이닝 `a.b().c()` — [day08 §0](day08_0909_keras.ipynb)
- 튜플 언패킹 — 묶여 나온 값을 변수에 나눠 담기 — [day16 §1-2](day16_0921_keras.ipynb)
- 메서드 체이닝 — `.flow(...).next()`는 두 줄을 한 줄로 붙인 것 — [day17 §2-2](day17_0922_keras.ipynb)
- 제너레이터 — `( ... for ...)`는 print하면 `<generator object>`, `list()`로 꺼내야 값이 보임 — [day24 RAG §1-2](day24_1006_rag.ipynb)

#### 문자열·경로
- 윈도우 경로에서 `\\`와 `/` — [day04 §5](day04_0903_keras.ipynb)
- 문자열 연결로 경로 만들기, 이스케이프 `\\` — [day05 §0](day05_0904_keras.ipynb)
- `datetime.strftime()` 포맷 코드 — [day05 §0](day05_0904_keras.ipynb)
- 여러 줄 문자열 `"""..."""` — [day05 §0](day05_0904_keras.ipynb)
- raw string은 백슬래시로 끝날 수 없다 — [day05 §0 cf)](day05_0904_keras.ipynb)
- `datetime.datetime` — 모듈과 클래스 이름이 같음 — [day11 §0](day11_0914_keras.ipynb)
- `''.join([...])`로 문자열 이어붙이기 — [day11 §0](day11_0914_keras.ipynb)
- 포맷 지정자 `{:04d}` / `{:.4f}`, `4f`와 `.4f`의 차이 — [day11 §0](day11_0914_keras.ipynb)

#### 인덱싱·슬라이싱
- 슬라이싱 `[시작:끝]`, 끝 인덱스는 미포함 — [day03 §0](day03_0902_keras.ipynb)
- 타입별 인덱싱 — 문자열·생성기·튜플·배열에서 `[0]`이 꺼내는 것이 각각 다름 — [day16 §2](day16_0921_keras.ipynb)
- 팬시 인덱싱 — 대괄호에 번호 목록을 넣어 여러 개를 한 번에 꺼내기 — [day17 §2-1](day17_0922_keras.ipynb)
- 콤마 인덱싱 — `arr[묶음, 줄, 특성]`으로 축을 한 번에 고름 — [day19 §2](day19_0928_keras.ipynb)
- `arr[:, -1]`과 `arr[:][-1]`은 다름 — 뒤는 0번 축에 두 번 접근한 것 — [day19 §2-1](day19_0928_keras.ipynb)

#### 넘파이
- `randint`와 `choice` — 중복을 끌 수 있느냐의 차이 — [day17 §1-2](day17_0922_keras.ipynb)
- `np.concatenate`의 괄호가 두 겹인 이유 — 재료를 묶음으로 받음 — [day17 §2-3](day17_0922_keras.ipynb)
- 리스트는 줄 길이가 달라도 되지만 배열은 안 됨 — `inhomogeneous shape` — [day19 §2-2](day19_0928_keras.ipynb)
- 텐서플로와 넘파이 — 전처리는 넘파이, 학습은 텐서 — [day23 §3](day23_1002_keras.ipynb)

#### 판다스
- 판다스를 넘파이로 바꾸는 3가지 — [day09 §0](day09_0910_keras.ipynb)
- 속성 `.values` vs 메서드 `.to_numpy()` — [day09 §0](day09_0910_keras.ipynb)
- `drop`은 라벨만 받음 — 위치로 자르려면 `iloc` — [day20 §3-1](day20_0929_keras.ipynb)
- `index`는 행 이름, `columns`는 열 이름 — 지우려는 축에서 뽑음 — [day20 §3-2](day20_0929_keras.ipynb)
- `drop`은 원본을 바꾸지 않음 — `inplace`와 재대입의 차이 — [day20 §3 cf)](day20_0929_keras.ipynb)

### 환경·저장

#### GPU 환경
- GPU 환경 구축 — 드라이버·CUDA·cuDNN·numpy 버전 조합, `nvidia-smi`·`nvcc -V` — [day12 §1](day12_0915_keras.ipynb)
- CPU·GPU 실행 시간 비교 — 작은 Dense 모델은 CPU가 빠른 경우가 많음 — [day12 §2](day12_0915_keras.ipynb)

#### 모델·가중치 저장
- `model.save()` / `load_model()` — `.keras` 포맷 — [day10 §5](day10_0911_keras.ipynb)
- 저장 시점(`fit` 전/후)에 따라 달라지는 것 — [day10 §5](day10_0911_keras.ipynb)
- 불러온 모델에 `compile`이 필요한 경우 — [day10 §5](day10_0911_keras.ipynb)
- `save_weights` / `load_weights` — `.weights.h5`, 구조·compile 없음 — [day11 §1](day11_0914_keras.ipynb)

### LangChain

#### 채팅 모델·체인
- 채팅 모델 클래스 — 회사가 달라도 `invoke`로 같게 부름, `base_url`로 호환 서버 연결 — [day21 RAG §1](day21_0930_rag.ipynb)
- `invoke` 한 번은 독립된 요청 — 이전 질문을 기억하지 않음 — [day21 RAG §1](day21_0930_rag.ipynb)
- `AIMessage` — `content`·`.text`·`response_metadata`·`usage_metadata` — [day21 RAG §1-1](day21_0930_rag.ipynb)
- `PromptTemplate` — `{변수}` 자리를 dict로 채움, 여러 줄로 역할·형식 지정 — [day21 RAG §2](day21_0930_rag.ipynb)
- LCEL `prompt | model | output_parser` — 단계별 입출력 — [day21 RAG §3](day21_0930_rag.ipynb)
- `StrOutputParser` — `AIMessage`에서 텍스트만 문자열로 — [day21 RAG §4](day21_0930_rag.ipynb)

#### 문서 분할
- 문서 분할 — 긴 문서를 검색 단위인 청크로, `TextLoader` → `Document` → `load_and_split` — [day24 RAG §1](day24_1006_rag.ipynb)
- `RecursiveCharacterTextSplitter` — `separators` 순서대로 자르고 `chunk_size`를 넘는 조각만 다시 자름, 짧은 조각은 합침, `chunk_size`는 상한이라 길이가 제각각 — [day24 RAG §1-1](day24_1006_rag.ipynb)
- `chunk_overlap` — 앞 청크 끝부분을 다음 청크 앞에 겹침, 끝의 100자 이하 조각만 넘어감 — [day24 RAG §1-1](day24_1006_rag.ipynb)
- 여러 파일 불러오기 — `glob` → `data += loader.load()` → `split_documents`, `append`면 리스트가 겹침 — [day24 RAG §1-2](day24_1006_rag.ipynb)
- 합치기와 겹침 — 합치기는 짧은 조각을 청크 하나로 묶음(중복 없음), 겹침은 앞 청크 끝을 다음 청크에 한 번 더 넣음 — [day24 RAG §1 cf)](day24_1006_rag.ipynb)

#### 임베딩
- 임베딩 모델 `OpenAIEmbeddings` — 문장 하나를 벡터 하나로, small 1536·large 3072, `dimensions`로 줄이기 — [day22 RAG §1](day22_1001_rag.ipynb)
- keras `Embedding`과 비교 — 단어마다 학습 vs 문장 전체를 학습된 모델이 — [day22 RAG §1-1](day22_1001_rag.ipynb)
- 코사인 유사도 — `(A · B) / (|A| × |B|)`, 방향이 같으면 1·수직 0·정반대 -1, 길이는 보지 않음 — [day22 RAG §1-2](day22_1001_rag.ipynb)
- 청크 임베딩 — 청크 하나가 벡터 하나, 청크 N개 → `(N, dimensions)` — [day24 RAG §2](day24_1006_rag.ipynb)
- `embed_query`와 `embed_documents` — 문자열 하나 → 벡터 하나 / 문자열 리스트 → 벡터 리스트 — [day24 RAG §2](day24_1006_rag.ipynb)

#### 벡터 DB·검색
- 의미 검색 — 단어가 달라도 뜻이 비슷한 문서를 찾음, 키워드 검색과 비교 — [day23 RAG §1](day23_1002_rag.ipynb)
- 벡터 저장소 — Chroma(컬렉션, 벡터 + 원문 + 메타데이터), FAISS(인덱스, 대규모 검색 속도), LangChain에서는 둘 다 `VectorStore` — [day23 RAG §1-1](day23_1002_rag.ipynb)
- Chroma 저장·불러오기 — `from_documents`는 임베딩해서 추가, `Chroma()`는 연결만, 같은 모델·`collection_name` — [day24 RAG §3](day24_1006_rag.ipynb)
- `db.get()` — `ids`·`documents`·`metadatas`, 벡터는 `include=['embeddings']` — [day24 RAG §3-1](day24_1006_rag.ipynb)
- `_collection.count()` — 컬렉션에 들어 있는 전체 청크 수, 실행할 때마다 누적 — [day24 RAG §3-1](day24_1006_rag.ipynb)
- 저장을 다시 실행하면 같은 청크가 새 id로 쌓임 — 검색 결과에 중복 — [day24 RAG §3 cf)](day24_1006_rag.ipynb)
- `similarity_search` — 질문과 가까운 청크 `k`개(기본 4)를 `Document` 리스트로 — [day24 RAG §4](day24_1006_rag.ipynb)
- Retriever — `as_retriever(search_kwargs={"k": 2})`로 만들고 `invoke`로 검색, LCEL 체인에 연결, 답이 없어도 `k`개를 돌려줌 — [day24 RAG §5](day24_1006_rag.ipynb)

## 작성 규칙

- 노트북 제목은 `#`, 대단원은 `##`, 소단원은 `###`
- 요약이 아니라 **개념·설명 위주**로 씁니다. 섹션 안의 순서는 개념 한 줄 → 필요한 이유 → **종류·규칙**(번호) → 설명·예제
  - 필요한 이유는 `**필요한 이유**` 라벨 없이 문장으로 바로 씁니다
  - 종류·규칙은 `1. 이름 : 설명` 형태로 번호를 매기고, 세부 설명은 `-` 하위 목록으로
  - 실습 결과표에는 **결과 해석**(왜 그런 결과인지)을 붙입니다
- 섹션 번호: 대단원 `## N. 개념명`, 소단원 `### N-M. 개념명` (예: `### 3-3. 순전파(Forward)와 역전파(Backward)`)
  - 곁가지·함정·예외는 모두 `cf)`를 개념명 앞에 붙입니다. 번호는 붙이지 않고 `####`로 대단원 맨 뒤에 둡니다
  - 제목은 명사형·동명사형으로 끝냅니다. 서술형 문장은 `cf)`를 붙일 때만 씁니다
  - README의 `§N` 참조가 깨지지 않도록 대단원 번호는 바꾸면 README도 같이 고칩니다
- 주제별 찾아보기는 `- 개념 — 설명 — [dayNN §N](파일)` 한 줄로, 알맞은 분류·키워드 묶음 안에 날짜순으로 넣습니다
- 실행 결과는 저장하지 않고, 기대값을 코드 주석으로 적습니다
- 그림은 `assets/`에 두고 이미지로 참조. 직접 그린 다이어그램은 SVG, 캡처·사진은 PNG (노트북에 인라인 HTML/SVG를 넣지 않음)
- 줄바꿈은 `<br/>` 대신 빈 줄(문단 구분)로
