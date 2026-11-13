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
| [day24](day24_1006_keras.ipynb) | 10/06 | RNN 데이터를 `Reshape`로 접어 `Conv2D`에, `Conv1D`, RNN 모델을 Conv1D로(`GlobalAveragePooling1D`), 층별 차원(`Conv1D` 추가) | `keras2/keras66_jena_CNN.py` ~ `keras68_Conv1D_15_man_woman.py` |
| [day24 RAG](day24_1006_rag.ipynb) | 10/06 | 문서 분할(`TextLoader`, `RecursiveCharacterTextSplitter`), 여러 파일 불러오기, 청크 임베딩(`embed_query`·`embed_documents`), Chroma 저장·불러오기·`get`, 유사도 검색, Retriever | `RAG/rag11_Chroma01_save.py` ~ `rag12_Chroma03_save.py` |
| [day25 RAG](day25_1007_rag.ipynb) | 10/07 | RAG(검색 → 붙이기 → 생성), 문서에 없는 질문, RAG 체인(`create_stuff_documents_chain`·`create_retrieval_chain`), 체인 결과 dict, Gradio 챗봇(`ChatInterface`), FAISS 저장·불러오기·저장 구조 | `RAG/rag12_Chroma04_load.py` ~ `rag18_FAISS_gradio.py` |
| [day26](day26_1008_keras.ipynb) | 10/08 | 다중 입력 모델(`Concatenate`), 여러 배열 `train_test_split`, 앙상블과의 차이, 다중 출력 모델, 출력별 loss, 출력에 이어지지 않은 층 | `keras2/keras69_ensemble1.py` ~ `keras69_ensemble3.py` |
| [day26 RAG](day26_1008_rag.ipynb) | 10/08 | PDF 불러오기(`PyPDFLoader`), 쪽별 `metadata`, PDF 분할(`split_documents`), 허깅페이스 임베딩(`HuggingFaceEmbeddings`), 질문 지시문, 임베딩 모델 교체, 요약·결론 질문 | `RAG/rag19_PyPDF_1.py` ~ `rag20_embedding5_chatbot.py` |
| [day27 RAG](day27_1012_rag.ipynb) | 10/12 | 대화 기록(`MessagesPlaceholder`·`HumanMessage`·`AIMessage`), Gradio `history` 형식, 검색은 이번 질문만 | `RAG/rag21_Gradio_history.py` |
| [day28](day28_1013_keras.ipynb) | 10/13 | 가중치 확인(`weights`·`trainable_weights`), 직접 계산, 랜덤 고정(seed), 동결(`trainable = False`), VGG16, `include_top` | `keras2/keras71_trainable_weights1.py` ~ `keras72_1_vgg16_include_top.py` |
| [day28 RAG](day28_1013_rag.ipynb) | 10/13 | 검색 문서 확인(`response['context']`), 유사도 기준값(`similarity_score_threshold`), 프롬프트 규칙 | `RAG/rag22_Gradio_1_doc.py` ~ `rag22_Gradio_3_prompt.py` |
| [day29](day29_1014_keras.ipynb) | 10/14 | 전이학습, 사전 학습 모델, Feature Extraction(동결), Fine Tuning(작은 `learning_rate`), 층별 동결(`layers[i].trainable`), `keras.applications` 모델들 | `keras2/keras73_vgg16_13_cifar10.py` ~ `keras76_all_T_01_cifar10.py` |
| [day30 RAG](day30_1015_rag.ipynb) | 10/15 | 키워드 검색(BM25), 벡터 검색과 비교, 하이브리드 검색(`EnsembleRetriever`·`weights`) | `RAG/rag23_01_bm25.py` ~ `rag23_03_hybrid.py` |
| [day32 LangGraph](day32_1019_langgraph.ipynb) | 10/19 | LangGraph(State·Node·Edge), `TypedDict`, 리듀서 `add_messages`, 그래프 만들기(`StateGraph`·`add_node`·`add_edge`·`compile`·`invoke`), 노드 여러 개 연결 | `LG/lg01_TypedDict.py` ~ `lg04_add_node_edge.py` |
| [day33 LangGraph](day33_1020_langgraph.ipynb) | 10/20 | Tool Calling(`bind_tools`·`tool_calls`), 검색 도구(`TavilySearch`·`DuckDuckGoSearchRun`), `ToolNode`·조건부 엣지(`tools_condition`), 그래프 그림(`draw_mermaid_png`) | `LG/lg05_ToolNode.py` ~ `lg07_ToolNode_graph01.py` |
| [day34 LangGraph](day34_1021_langgraph.ipynb) | 10/21 | 그래프 그림 출력(`BytesIO`·`plt.imshow`), 메모리(`InMemorySaver`·`thread_id`), fan-out / fan-in | `LG/lg07_ToolNode_graph02.py` ~ `lg09_fanout_fanin2_concat.py` |
| [day35 LangGraph·ML](day35_1022_langgraph_ml.ipynb) | 10/22 | 결과 통합 노드(LLM으로 fan-in 결과 정리), 트리 계열 모델(`DecisionTree`·`RandomForest`·`GradientBoosting`·`XGBoost`), `feature_importances_`, XGBoost 중요도 기준(weight·gain·`plot_importance`) | `LG/lg09_fanout_fanin3.py`, `ml/m01_feature_importances01.py` ~ `m01_feature_importances04_graph.py` |
| [day41 NPU](day41_1030_npu.ipynb) | 10/30 | WSL 배포판(`wsl --install --name`·`wsl -l -v`), 윈도우 파일 접근(`/mnt/c`·`explorer.exe`), SSH 키 접속(`ssh -i -p`·`chmod 600`), NPU 상태 확인(`furiosa-smi`), `watch`, 서버 사양 확인(`lscpu`·`free -h`), Miniconda 설치(`curl -O`·`conda init`·`conda create`), 윈도우에서 바로 접속(`.ssh/config`·`icacls`) | — |
| [day42 NPU·LLM](day42_1102_npu.ipynb) | 11/02 | WSL 배포판 삭제·전환(`wsl --unregister`·`wsl -d`), 파일/권한 트러블슈팅(`wslpath`·`chown`·`chmod`), SSH 키 복사 및 원격 접속(`StrictHostKeyChecking=accept-new`), Hugging Face 캐시 관리(`hf cache ls`·`hf cache rm`), Furiosa-LLM NPU 모델 서빙(`Qwen3-8B`·`EXAONE-32B`·`Llama-3.1-8B`), 모델 파라미터(B)와 양자화(FP8·메모리 계산), `fxb build` 사전 컴파일, WSL vs SSH 환경 가이드 | `furiosa/furiosa1/test0001.py` |
| [day42 ML](day42_1102_ml.ipynb) | 11/02 | Hold-out 분할의 한계, K-Fold 교차 검증(5-Fold 분할·데이터 손실 방지·일반화 성능), random_state 착시, K-Fold 검증 데이터(`validation_split`), Stratified K-Fold, 최종 가중치 결정(전체 재학습·앙상블) | `ml/m02_kfold_00_iris.py` ~ `m02_kfold_01_california.py` |
| [day43 NPU](day43_1103_npu.ipynb) | 11/03 | 파이썬에서 모델 올리기(`LLM`·`SamplingParams`·`apply_chat_template`·`generate`), Tool Calling(`tools` 스키마·`tool_choice`·`tool_calls`), `fxb`로 사전 컴파일(`fxb build`·`fxb cache ls`·`fxb add`·`--fxb`) | `furiosa/furiosa1/f02_tutorial.py` ~ `f04_fxb_custom.py` |
| [day43 ML](day43_1103_ml.ipynb) | 11/03 | `train_test_split`과 K-Fold 함께 쓰기(스케일링 위치), `cross_val_predict`, `all_estimators`로 모델 전체 비교 | `ml/m02_kfold_05_kaggle_bike.py` ~ `m05_all_estimators2_kfold.py` |
| [day44 양자화](day44_1104_quantization.ipynb) | 11/04 | 양자화 파이프라인(BF16 → GGUF → Q4_K_M), Ollama(`ollama run`·`ollama list`), llama.cpp 빌드(`cmake -B`·`-j`), 파이썬 가상환경(`venv`·`source`), GGUF 변환(`convert_hf_to_gguf.py`), 양자화 실행(`llama-quantize`) | — |
| [day45 NPU](day45_1105_npu.ipynb) | 11/05 | OpenAI 호환 API로 NPU 서버 추론(`OpenAI`·`models.list()`), LangChain 연결(`ChatOpenAI`·`invoke`), 처리량 측정(`time.perf_counter`·tokens/s), 스트리밍(`stream=True`·`delta.content`·`llm.stream`), 공통 설정 모듈(`get_llm`·`import`) | `furiosa/furiosa1/common.py`, `f01_furiosa_test.py`, `f05_langchain.py` ~ `f07_stream02_lc.py` |
| [day45 파이썬](day45_1105_python.ipynb) | 11/05 | 모듈(`import`·`from … import`), `__name__`과 `__main__`, `if __name__=='__main__':`, 모듈별 이름공간(전역 변수가 모듈마다 따로), 패키지(`machine.car`·`from machine import car, tv`) | `python_pkg/p01_car.py` ~ `p51_from_module.py` |
| [day45 ML](day45_1105_ml.ipynb) | 11/05 | 데이터셋 리스트로 묶어 돌리기(`enumerate`·`np.argmax`), for문 파라미터 튜닝(조합마다 `best_score` 갱신), `GridSearchCV`(`parameters` 딕셔너리 목록·`best_params_`·`best_estimator_`·`best_score_`) | `ml/m06_all_estimators1_datasets.py` ~ `m08_GridSearchCV_02_diabetes.py` |
| [day46 ML](day46_1106_ml.ipynb) | 11/06 | `RandomizedSearchCV`(`n_iter`로 조합 일부만 뽑기·`random_state`), `refit`(최종 재학습과 `best_estimator_`) | `ml/m09_RandomSearchCV_00.py` ~ `m09_RandomSearchCV_02_diabetes.py` |
| [day46](day46_1106_keras.ipynb) | 11/06 | 케라스 모델 하이퍼파라미터 튜닝(모델을 함수로·하이퍼파라미터 딕셔너리·`KerasRegressor` 래핑·`RandomizedSearchCV`) | `keras2/keras77_hyperParameter02_diabetes.py` |
| [day47 NPU](day47_1109_npu.ipynb) | 11/09 | 지연 시간 반복 측정(평균·prefill과 decode), 첫 응답 시간(`first_time`·`break`·`next(iter(...))`), `usage_metadata`(`output_tokens`·`cache_read`·TPS) | `furiosa/furiosa1/f08_pkg01_invoke.py`, `furiosa2/f11_basic.py` ~ `f18_compare.py` |
| [day47 ML](day47_1109_ml.ipynb) | 11/09 | `HalvingGridSearchCV`(적은 샘플로 조합 걸러내기·`factor`·`min_resources`·`max_resources`·iter 수) | `ml/m10_HalvingGridSearchCV_00.py` |
| [day48 NPU](day48_1110_npu.ipynb) | 11/10 | 동시 요청과 continuous batching(`ThreadPoolExecutor`·`pool.map`·동시 요청 수별 TPS와 지연), 측정 함수로 묶기(`run_once`·`usage_metadata` 없을 때 `.get`), KV Cache와 `/metrics`(prefix cache 적중률·사용률) | `furiosa/furiosa2/f21_continuous_batching1.py` ~ `f22_kv_cache2.py` |
| [day48 파이썬](day48_1110_python.ipynb) | 11/10 | `map`(함수 이름만 넘기기·`list()`로 꺼내기), `lambda`(한 줄 함수·조건식·`map`과 함께) | `python/p01_map.py` ~ `p02_lambda3.py` |
| [day49 NPU](day49_1111_npu.ipynb) | 11/11 | 요청 처리량(RPS와 TPS·순차와 동시·동시 요청 수별 비교), 요청 대기(`submit`·`futures`·`max_workers`), 적중률 전후 차이 계산, 서버 상태 확인(`/metrics` 거르기·`num_requests_running`) | `furiosa/furiosa2/f21_continuous_batching3.py`, `f23_prefix_cache1.py` ~ `f29_check_server_metric.py` |
| [day50](day50_1112_keras.ipynb) | 11/12 | 활성화 함수 그래프(sigmoid·tanh·relu·silu·mish·elu·selu·leaky_relu 식 직접 구현·`np.arange`·`plt.plot`) | `keras2/keras78_01_simoid.py` ~ `keras78_08_leaky_relu.py` |
| [day51 NPU](day51_1113_npu.ipynb) | 11/13 | 스케줄링 모드(eager와 overlap·`--enable-overlap-scheduling`·서버 재시작과 `EBUSY`) | `furiosa/furiosa2/f30_overlap_eager1.py` ~ `f30_overlap_eager2.py` |
| [day51](day51_1113_keras.ipynb) | 11/13 | softmax(합이 1인 비율·파이 차트), 경사하강법(기울기 반대 방향으로 이동·`lr * 기울기`), `learning_rate`(고정 폭으로 옮겨 보기·진동), TensorBoard(`log_dir`·`tensorboard --logdir`) | `keras2/keras78_09_softmax.py` ~ `keras80_Tensorboard.py` |

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
- 여러 배열을 한 번에 나누기 — `train_test_split(x1, x2, y)`, 모두 같은 행 번호로 나눠 넣은 순서대로 `(train, test)` 쌍 — [day26 §1](day26_1008_keras.ipynb)
- Hold-out 분할의 한계 — 테스트 데이터 손실과 평가 편향 — [day42 ML §1](day42_1102_ml.ipynb)
- K-Fold 교차 검증 — k개 묶음으로 나눠 테스트셋을 바꿔가며 k번 학습·평가, 모든 데이터 활용 — [day42 ML §2](day42_1102_ml.ipynb)
- K-Fold 검증 데이터 — fold 내부 학습셋에서 `validation_split`으로 조기종료 모니터링 — [day42 ML §3](day42_1102_ml.ipynb)
- Stratified K-Fold — 분류 라벨 비율을 fold마다 동일하게 유지 — [day42 ML §3 cf)](day42_1102_ml.ipynb)
- random_state의 착시 — 우연히 쉬운 테스트셋이 걸려 점수가 과대평가되는 현상 — [day42 ML §2 cf)](day42_1102_ml.ipynb)
- 최종 가중치 결정 — K-Fold 검증 후 전체 데이터(100%) 재학습 또는 k개 모델 앙상블 — [day42 ML §4](day42_1102_ml.ipynb)
- `train_test_split` + K-Fold — test를 먼저 떼고 train 안에서만 교차 검증, 스케일러는 train에만 `fit` — [day43 ML §1](day43_1103_ml.ipynb)
- `cross_val_predict` — 점수 대신 fold별 예측값을 모아 돌려줌, `accuracy_score`·`r2_score`로 따로 계산 — [day43 ML §2](day43_1103_ml.ipynb)
- `cross_val_predict` 점수가 더 낮은 이유 — 넘긴 데이터를 다시 k등분해 학습하므로 학습량이 줄어서 — [day43 ML §2 cf)](day43_1103_ml.ipynb)
- `all_estimators` — sklearn 모델을 `(이름, 클래스)`로 전부 받아 비교, `type_filter`, `try`/`except`로 에러 넘김 — [day43 ML §3](day43_1103_ml.ipynb)

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
- 층별 차원에 `Conv1D` 추가 — `Conv1D` 입력·출력 3차원, `Dense`는 2차원 이상, `Embedding`은 `input_dim`·`output_dim`만 정함 — [day24 §3](day24_1006_keras.ipynb)

#### 모델에 맞게 바꾸기
- DNN으로 이미지 처리 — `reshape(-1, 28 * 28)`, 컬러는 `32 * 32 * 3` — [day14 §3](day14_0917_keras.ipynb)
- 표 데이터를 reshape해서 `Conv2D`에 넣기 — 가능한 조건과 성능을 장담 못하는 이유 — [day14 §4](day14_0917_keras.ipynb)
- 표 데이터를 CNN으로 바꿀 때 자주 틀리는 곳 (reshape 곱, input_shape, 작은 입력, 출력층 활성화) — [day14 §4 cf)](day14_0917_keras.ipynb)
- 3·4차원 출력에 바로 붙인 `Dense` — 마지막 축에만 적용돼 차원이 남고 `fit`에서 에러 — [day23 §2 cf)](day23_1002_keras.ipynb)
- DNN·CNN을 RNN으로 — `(N, 특성)` → `(N, 특성, 1)`, `(N, 세로, 가로, 채널)` → `(N, 세로, 가로 × 채널)` — [day23 §5](day23_1002_keras.ipynb)
- `Reshape` 층 — 모델 안에서 모양 변경, 파라미터 0, 원소 개수 유지, `LSTM` 앞에서는 timesteps 축을 정함 — [day23 §6](day23_1002_keras.ipynb)
- RNN 데이터를 `Conv2D`로 — `Reshape((12, 12, 13))`로 144칸을 접음, 세로 이웃이 12칸 떨어진 시점이라 순서가 섞임 — [day24 §1](day24_1006_keras.ipynb)
- `Flatten`이 필요한 경우 — `Conv2D`·`Conv1D` 출력, 3차원 이상을 받은 `Dense`, RNN 없이 `Dense`로 가는 `Embedding` — [day24 §3](day24_1006_keras.ipynb)

### 모델 층

#### 모델 기본
- 코딩 4단계 (데이터 → 모델 → compile/fit → evaluate/predict) — [day01 §5](day01_0831_keras.ipynb)
- Sequential / Layer / Node(Unit) — [day01 §6](day01_0831_keras.ipynb)
- Dense(완전연결층) — [day01 §7](day01_0831_keras.ipynb)
- 파라미터(w·b) 개수 세는 공식 — [day09 §2](day09_0910_keras.ipynb)
- `model.summary()` 읽는 법, `Output Shape`의 `None` — [day09 §2, §3](day09_0910_keras.ipynb)
- 함수형 모델 `Input` / `Model` — `Sequential`과 연결 방식 비교 — [day11 §4](day11_0914_keras.ipynb)
- 다중 입력 모델 — `Input` 여러 개, 입력마다 쌓은 층을 `Concatenate`로 합침, x는 `inputs` 순서대로 리스트로 (수업에서는 앙상블) — [day26 §1](day26_1008_keras.ipynb)
- 앙상블 — 여러 모델의 예측값을 평균·투표로 합침, 모델마다 예측이 달라야 오차 일부가 상쇄됨 — [day26 §1 cf)](day26_1008_keras.ipynb)
- 다중 출력 모델 — `outputs`·y를 리스트로, loss는 출력마다 계산해 합을 줄임 — [day26 §2](day26_1008_keras.ipynb)
- 출력에 이어지지 않은 층 — 에러 없이 모델에서 빠짐, `summary()`로 확인 — [day26 §2 cf)](day26_1008_keras.ipynb)
- 가중치 확인 — `model.weights`·`trainable_weights`, `Dense`마다 `kernel`·`bias`, `kernel`은 `(입력 노드 수, 출력 노드 수)`, `bias`는 0으로 시작 — [day28 §1](day28_1013_keras.ipynb)
- 가중치로 직접 계산 — 입력마다 `kernel` 열 값을 곱해 더하고 `bias`를 더함, `predict`와 비교 — [day28 §1-1](day28_1013_keras.ipynb)
- 랜덤 고정 — `random.seed`·`np.random.seed`·`tf.random.set_seed`, 같은 층 구성이면 처음 가중치가 같음 — [day28 §1 cf)](day28_1013_keras.ipynb)

#### 활성화 함수
- 활성화 함수란, 기본값이 `linear`라는 것 — [day05 §3](day05_0904_keras.ipynb)
- ReLU가 비선형성을 주는 이유, 기울기 소실 방지 — [day05 §3](day05_0904_keras.ipynb)
- `activation='relu'` 넣는 위치 — [day05 §3](day05_0904_keras.ipynb)
- 식을 그려서 모양 보기 — `np.arange`로 x 구간, sigmoid는 0~1, tanh는 -1~1, relu는 음수 구간 0 — [day50 §1](day50_1112_keras.ipynb)
- SiLU(swish) — `x * sigmoid(x)`, 음수 쪽이 작은 음수로 남아 0에서 각지게 꺾이지 않음 — [day50 §1](day50_1112_keras.ipynb)
- softmax — `np.exp(x) / np.sum(np.exp(x))`, 묶음 전체를 합이 1인 비율로 바꿔 확률로 읽음 — [day51 §1](day51_1113_keras.ipynb)
- Mish — `x * tanh(log(1 + exp(x)))`, silu와 모양은 비슷하고 연산량은 더 많음 — [day50 §1](day50_1112_keras.ipynb)
- ELU·SELU — 음수 구간을 지수 함수로, `-alpha`에 수렴, selu는 `lmbda`를 곱한 고정 상수 버전 — [day50 §1](day50_1112_keras.ipynb)
- leaky_relu — 음수 구간에 `alpha`(0.01)를 곱해 기울기를 남김 — [day50 §1](day50_1112_keras.ipynb)
- 넘파이로 식 적기 — `np.maximum`, `np.where`, `(x > 0) * x`로 구간 나누기, `lambda`가 예약어라 `lmbda` — [day50 §1 cf)](day50_1112_keras.ipynb)

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

#### 전이학습
- 동결 — `model.trainable = False`, `trainable_weights`만 0개, `fit`해도 가중치가 안 바뀜, `compile` 전에 설정 — [day28 §2](day28_1013_keras.ipynb)
- VGG16 — ImageNet으로 학습된 CNN, `weights='imagenet'`, 3×3 `Conv2D` 블록 5개 + `Dense` 3개, 출력 1000개 클래스 — [day28 §3](day28_1013_keras.ipynb)
- `include_top` — 분류 부분(`Flatten`~출력 `Dense`)까지 가져올지, 클래스 수가 다르면 `False`로 빼고 `Dense`를 붙임 — [day28 §3-1](day28_1013_keras.ipynb)
- top을 두고 `input_shape` 바꾸기 — `include_top=True`·`weights='imagenet'`이면 `(224, 224, 3)`만 됨 — [day28 §3-1 cf)](day28_1013_keras.ipynb)
- 전이학습 — 사전 학습 모델(Pre-Trained Model)을 가져와 내 데이터로 학습, `include_top=False` + 클래스 수에 맞는 `Dense` — [day29 §1](day29_1014_keras.ipynb)
- Feature Extraction — 사전 학습 모델을 동결(`trainable = False`)해 특징 추출기로, 붙인 `Dense`만 학습 — [day29 §1-1](day29_1014_keras.ipynb)
- Fine Tuning — 사전 학습 모델까지 `trainable = True`로 학습, `learning_rate`를 작게 — [day29 §1-2](day29_1014_keras.ipynb)
- Fine Tuning에서 `learning_rate`가 크면 — loss ln(클래스 수)·acc 1 / 클래스 수에서 멈춤 — [day29 §1 cf)](day29_1014_keras.ipynb)
- 층별 동결 — `model.layers[i].trainable = False`로 층 하나만, `Sequential`에 넣은 사전 학습 모델은 `Functional` 층 하나로 보임, 안쪽 층은 그 모델의 `layers`로 — [day29 §2](day29_1014_keras.ipynb)
- `keras.applications` — VGG·ResNet·DenseNet·Inception·MobileNet·NASNet·EfficientNet, 기본 입력 크기가 모델마다 다름, 리스트를 `for`로 돌며 비교 — [day29 §3](day29_1014_keras.ipynb)
- `weights` > `trainable_weights` — `BatchNormalization`의 `moving_mean`·`moving_variance`는 역전파로 안 바뀜, VGG는 둘이 같음 — [day29 §3 cf)](day29_1014_keras.ipynb)

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
- 고정 폭으로 옮겨 보기 — `w ± lr`의 loss를 견줘 방향만 정하면 최솟값에서 `lr` 간격으로 진동 — [day51 §3](day51_1113_keras.ipynb)

#### 콜백
- 콜백(callback)과 `EarlyStopping` — [day06 §7](day06_0907_keras.ipynb)
- 과적합 지점에서 자동으로 멈추기 — [day06 §7](day06_0907_keras.ipynb)
- `restore_best_weights`가 필요한 이유 — [day06 §7](day06_0907_keras.ipynb)
- `patience`는 epochs의 10~20% 선 — [day06 §7](day06_0907_keras.ipynb)
- Early Stopping이 최고 모델을 보장하지 않는 이유 — [day07 §1](day07_0908_keras.ipynb)
- `TensorBoard` 콜백 — `log_dir`에 로그를 남기고 `tensorboard --logdir=<경로>`로 띄워 `localhost:6006`에서 봄 — [day51 §4](day51_1113_keras.ipynb)
- 체크포인트 파일명에 `{epoch}`·`{val_loss}` 넣기 — [day11 §0](day11_0914_keras.ipynb)
- `ModelCheckpoint` — 훈련 중 최고 지점을 파일로 저장 — [day11 §2](day11_0914_keras.ipynb)
- `restore_best_weights`(메모리) vs 체크포인트(파일) — [day11 §2](day11_0914_keras.ipynb)
- `ReduceLROnPlateau` — `val_loss`가 안 좋아지면 `lr`을 줄여주는 콜백 — [day18 §2](day18_0923_keras.ipynb)
- `ReduceLROnPlateau`와 `EarlyStopping`의 `patience` — 줄여볼 기회를 주려면 차이를 둠 — [day18 §2-2](day18_0923_keras.ipynb)

#### 튜닝
- 하이퍼파라미터란 — [day01 §8, §9](day01_0831_keras.ipynb)
- 층·epoch만 키운다고 좋아지지 않는다 (튜닝 기록) — [day05 §4](day05_0904_keras.ipynb)
- 튜닝 우선순위 — 데이터 > 짝 > 구조 > 규제 > 학습 설정 — [day17 §4-7](day17_0922_keras.ipynb)
- 케라스 모델 래핑 — 모델을 함수로 만들고 `KerasRegressor`로 감싸 sklearn 탐색기에 넣음 — [day46 §1](day46_1106_keras.ipynb)
- 하이퍼파라미터 딕셔너리 — 키는 모델 함수의 매개변수 이름과 같아야 하고, `batch_size`는 래퍼가 받음 — [day46 §1](day46_1106_keras.ipynb)

### 평가·예측

#### evaluate
- matplotlib으로 회귀선 그려 확인하기 — [day03 §3](day03_0902_keras.ipynb)
- `fit`의 loss와 `evaluate`의 loss 차이 — [day03 §5](day03_0902_keras.ipynb)
- loss 값을 데이터셋끼리 비교할 수 없는 이유 — [day03 §5](day03_0902_keras.ipynb)
- `evaluate` 안에 `predict`가 포함된다 — [day04 §4](day04_0903_keras.ipynb)
- MSE를 구하는 두 경로 (케라스 / sklearn) — [day04 §4](day04_0903_keras.ipynb)
- `metrics`를 주면 `evaluate`가 리스트를 반환 — [day07 §5](day07_0908_keras.ipynb)
- `evaluate`에 y를 안 주면 나는 에러 — [day08 §5 cf)](day08_0909_keras.ipynb)
- 출력이 여러 개일 때 `evaluate` — `[loss 합, 출력별 loss]`, `return_dict=True`면 이름 붙은 dict — [day26 §2](day26_1008_keras.ipynb)

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
- 입력·출력이 여러 개일 때 `predict` — x는 리스트로 넣고(묶지 않으면 두 번째 배열이 `batch_size` 자리로), 출력별 배열을 리스트로 돌려줌 — [day26 §1, §2](day26_1008_keras.ipynb)

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
- 경사하강법 직접 구현 — `x = x - lr * gradient(x)`, 기울기가 작아지면 이동 폭도 줄어 최솟값에서 멈춤 — [day51 §2](day51_1113_keras.ipynb)

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
- `map` — 함수를 값마다 한 번씩 적용, 함수는 이름만 넘기고 `list()`로 꺼냄, 한 번 꺼내면 비워짐 — [day48 파이썬 §1](day48_1110_python.ipynb)
- `lambda` — 콜론 왼쪽이 매개변수, 오른쪽 식이 반환값, `return`을 적지 않고 식 하나만 들어감 — [day48 파이썬 §2](day48_1110_python.ipynb)
- 조건식 — `값 if 조건 else 값` 순서, `lambda`나 `map` 안에 한 줄로 넣음 — [day48 파이썬 §2](day48_1110_python.ipynb)

#### 모듈·패키지
- 모듈 — 함수·변수를 모아 둔 `.py` 파일, `import <모듈>`은 이름을 붙여 쓰고 `from <모듈> import <이름>`은 바로 씀, 같은 폴더면 경로를 안 적음 — [day45 파이썬 §1](day45_1105_python.ipynb)
- `__pycache__` — 처음 불러올 때 생기는 `.pyc`, 다음부터 파일을 다시 해석하지 않고 그것을 씀 — [day45 파이썬 §1](day45_1105_python.ipynb)
- `__name__` — 직접 실행하면 `__main__`, `import`되면 모듈 이름. `import`가 그 파일을 한 번 실행하기 때문에 구분이 필요함 — [day45 파이썬 §2](day45_1105_python.ipynb)
- `if __name__=='__main__':` — 직접 실행할 때만 돌리는 블록, 다른 파일에서 쓸 `def`는 바깥에 둠 — [day45 파이썬 §3](day45_1105_python.ipynb)
- 모듈별 이름공간 — 전역 변수는 적어 둔 모듈 안에서만 전역, 함수가 보는 변수는 어디서 불렀는지가 아니라 어디에 정의됐는지로 정해짐 — [day45 파이썬 §4](day45_1105_python.ipynb)
- 패키지 — 모듈을 폴더로 묶어 `machine.car`처럼 점으로 내려감, `import machine.car as car`·`from machine import car, tv`, 폴더 안의 폴더도 점으로 이어짐 — [day45 파이썬 §5](day45_1105_python.ipynb)
- `__init__.py` — 없어도 폴더가 패키지로 인식됨(3.3부터) — [day45 파이썬 §5](day45_1105_python.ipynb)

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
- `ChatPromptTemplate` — 템플릿 전체가 `HumanMessage` 하나로 들어감, `PromptTemplate`은 채우면 문자열 — [day25 RAG §2](day25_1007_rag.ipynb)

#### 문서 분할
- 문서 분할 — 긴 문서를 검색 단위인 청크로, `TextLoader` → `Document` → `load_and_split` — [day24 RAG §1](day24_1006_rag.ipynb)
- `RecursiveCharacterTextSplitter` — `separators` 순서대로 자르고 `chunk_size`를 넘는 조각만 다시 자름, 짧은 조각은 합침, `chunk_size`는 상한이라 길이가 제각각 — [day24 RAG §1-1](day24_1006_rag.ipynb)
- `chunk_overlap` — 앞 청크 끝부분을 다음 청크 앞에 겹침, 끝의 100자 이하 조각만 넘어감 — [day24 RAG §1-1](day24_1006_rag.ipynb)
- 여러 파일 불러오기 — `glob` → `data += loader.load()` → `split_documents`, `append`면 리스트가 겹침 — [day24 RAG §1-2](day24_1006_rag.ipynb)
- 합치기와 겹침 — 합치기는 짧은 조각을 청크 하나로 묶음(중복 없음), 겹침은 앞 청크 끝을 다음 청크에 한 번 더 넣음 — [day24 RAG §1 cf)](day24_1006_rag.ipynb)
- `PyPDFLoader` — PDF를 쪽마다 `Document` 하나로, `TextLoader`로는 못 읽음, 쪽 번호·머리말도 `page_content`에 들어감 — [day26 RAG §1](day26_1008_rag.ipynb)
- PDF `metadata` — `source`·`total_pages`·`page`(0부터)·`page_label`(문자열)과 PDF 파일 정보, 쪽마다 `page`·`page_label`만 다름 — [day26 RAG §1-1](day26_1008_rag.ipynb)
- PDF 분할 — `split_documents(pdf_docs)`, 청크마다 쪽 `metadata`가 붙고 두 쪽에 걸치지 않음, `load_and_split`은 PDF를 한 번 더 읽음 — [day26 RAG §1-2](day26_1008_rag.ipynb)

#### 임베딩
- 임베딩 모델 `OpenAIEmbeddings` — 문장 하나를 벡터 하나로, small 1536·large 3072, `dimensions`로 줄이기 — [day22 RAG §1](day22_1001_rag.ipynb)
- keras `Embedding`과 비교 — 단어마다 학습 vs 문장 전체를 학습된 모델이 — [day22 RAG §1-1](day22_1001_rag.ipynb)
- 코사인 유사도 — `(A · B) / (|A| × |B|)`, 방향이 같으면 1·수직 0·정반대 -1, 길이는 보지 않음 — [day22 RAG §1-2](day22_1001_rag.ipynb)
- 청크 임베딩 — 청크 하나가 벡터 하나, 청크 N개 → `(N, dimensions)` — [day24 RAG §2](day24_1006_rag.ipynb)
- `embed_query`와 `embed_documents` — 문자열 하나 → 벡터 하나 / 문자열 리스트 → 벡터 리스트 — [day24 RAG §2](day24_1006_rag.ipynb)
- `HuggingFaceEmbeddings` — Hub 임베딩 모델을 내 컴퓨터에서, 키·크레딧 없음, 처음 실행할 때 모델 파일을 `~/.cache/huggingface/hub`에 받음, bge-m3·Qwen3-Embedding-0.6B 1024차원 — [day26 RAG §2](day26_1008_rag.ipynb)
- 질문 지시문 — 지시문을 붙여 학습된 모델(Qwen3-Embedding)은 `query_encode_kwargs={"prompt_name": "query"}`로 질문에만 붙임, `encode_kwargs`면 청크에도 붙음 — [day26 RAG §2 cf)](day26_1008_rag.ipynb)

#### 벡터 DB·검색
- 의미 검색 — 단어가 달라도 뜻이 비슷한 문서를 찾음, 키워드 검색과 비교 — [day23 RAG §1](day23_1002_rag.ipynb)
- 벡터 저장소 — Chroma(컬렉션, 벡터 + 원문 + 메타데이터), FAISS(인덱스, 대규모 검색 속도), LangChain에서는 둘 다 `VectorStore` — [day23 RAG §1-1](day23_1002_rag.ipynb)
- Chroma 저장·불러오기 — `from_documents`는 임베딩해서 추가, `Chroma()`는 연결만, 같은 모델·`collection_name` — [day24 RAG §3](day24_1006_rag.ipynb)
- `db.get()` — `ids`·`documents`·`metadatas`, 벡터는 `include=['embeddings']` — [day24 RAG §3-1](day24_1006_rag.ipynb)
- `_collection.count()` — 컬렉션에 들어 있는 전체 청크 수, 실행할 때마다 누적 — [day24 RAG §3-1](day24_1006_rag.ipynb)
- 저장을 다시 실행하면 같은 청크가 새 id로 쌓임 — 검색 결과에 중복 — [day24 RAG §3 cf)](day24_1006_rag.ipynb)
- `similarity_search` — 질문과 가까운 청크 `k`개(기본 4)를 `Document` 리스트로 — [day24 RAG §4](day24_1006_rag.ipynb)
- Retriever — `as_retriever(search_kwargs={"k": 2})`로 만들고 `invoke`로 검색, LCEL 체인에 연결, 답이 없어도 `k`개를 돌려줌 — [day24 RAG §5](day24_1006_rag.ipynb)
- FAISS 저장·불러오기 — `from_documents` → `save_local`, `load_local`, `.faiss`(벡터)·`.pkl`(문서) 두 파일, 다시 저장하면 덮어씀 — [day25 RAG §4](day25_1007_rag.ipynb)
- `allow_dangerous_deserialization` — `.pkl`을 pickle로 읽어서 `True`를 줘야 `load_local`이 열림 — [day25 RAG §4](day25_1007_rag.ipynb)
- FAISS 저장 구조 — `index`(벡터)·`docstore`(원문)·`index_to_docstore_id`(행 번호 → id), `index.d`·`index.ntotal` — [day25 RAG §4-1](day25_1007_rag.ipynb)
- 인덱스 직접 만들기 — `IndexFlatL2(차원)` + `InMemoryDocstore` + 빈 연결표, 잘 안 씀 — [day25 RAG §4 cf)](day25_1007_rag.ipynb)
- 임베딩 모델 교체 — 저장·질문 모델이 같아야 함, 차원이 다르면 검색할 때 `AssertionError`, 같으면 에러 없이 질문과 상관없는 청크 — [day26 RAG §3](day26_1008_rag.ipynb)
- 요약·결론 질문 — 비교할 내용이 없어 초록·결론 청크가 위로 오기 어려움, 내용을 넣어 묻기, 본문이 아닌 쪽 빼기, `k`를 늘려도 한참 밀린 청크는 안 잡힘 — [day26 RAG §4](day26_1008_rag.ipynb)
- 유사도 기준값 — `search_type="similarity_score_threshold"`, `score_threshold` 이상만 최대 `k`개, 없으면 경고와 빈 `context` — [day28 RAG §2](day28_1013_rag.ipynb)
- BM25 — 질문 단어가 든 문서에 점수, `BM25Retriever.from_documents`·`k` 속성, 임베딩 없이 동작 — [day30 RAG §1](day30_1015_rag.ipynb)
- BM25 띄어쓰기 토큰화 — `text.split()`이라 조사가 붙으면 다른 단어, 일치 단어가 없으면 0점 — [day30 RAG §1 cf)](day30_1015_rag.ipynb)
- 벡터 검색 — 임베딩 유사도로 찾음, 단어가 달라도 뜻이 가까우면 찾음, `from_documents` 인자는 `embedding` — [day30 RAG §2](day30_1015_rag.ipynb)
- 하이브리드 검색 — `EnsembleRetriever(retrievers, weights)`, 검색기마다 `weight / (순위 + 60)`을 더해 다시 정렬, 같은 문서는 하나로 — [day30 RAG §3](day30_1015_rag.ipynb)
- 하이브리드 결과 개수 — 검색기별 `k`의 합에서 겹친 문서를 뺀 수 — [day30 RAG §3 cf)](day30_1015_rag.ipynb)

#### RAG
- RAG — 검색한 청크를 질문과 함께 프롬프트에 넣어 답 받기, 검색 → 붙이기 → 생성 — [day25 RAG §1](day25_1007_rag.ipynb)
- 문서에 없는 질문 — "근거해 답하라"고 해도 학습한 지식으로 답함, 정보가 없으면 답할 수 없다고 지시 — [day25 RAG §1 cf)](day25_1007_rag.ipynb)
- RAG 체인 — `create_stuff_documents_chain`은 문서를 `{context}`에 채워 `prompt | model | StrOutputParser`, `create_retrieval_chain`은 검색까지 묶음, 변수 이름 `{context}`·`{input}` 고정 — [day25 RAG §2](day25_1007_rag.ipynb)
- 체인 결과 — `input`·`context`·`answer` dict, 답은 `response['answer']` — [day25 RAG §2-1](day25_1007_rag.ipynb)
- `langchain_classic` — langchain 1.x에는 `chains`가 없어 체인 함수를 여기서 가져옴 — [day25 RAG §2 cf)](day25_1007_rag.ipynb)
- Gradio 챗봇 — `ChatInterface(fn, title)` + `launch()`, `fn(message, history)`가 답 문자열을 돌려줌, `history`를 안 넘기면 질문마다 따로 검색 — [day25 RAG §3](day25_1007_rag.ipynb)
- 답이 나와도 검색이 틀렸을 수 있음 — 제목·저자만 든 청크가 넘어가도 유명한 문서는 요약이 나올 수 있음, `response['context']`로 확인 — [day26 RAG §4 cf)](day26_1008_rag.ipynb)
- 대화 기록 — `MessagesPlaceholder('chat_history')`, `history`를 `HumanMessage`·`AIMessage`로 바꿔 `invoke`에 넘김 — [day27 RAG §1](day27_1012_rag.ipynb)
- 검색은 이번 질문만 — 검색기는 `input`만 받음 — [day27 RAG §1 cf)](day27_1012_rag.ipynb)
- 검색 문서 확인 — `response['context']`의 `Document`마다 `metadata.get("source")`·`page_content` 출력 — [day28 RAG §1](day28_1013_rag.ipynb)
- 프롬프트 규칙 — system에 이전 대화·일상 대화·문서 질문·정보 없음일 때 답하는 법을 번호로, 검색 방식은 그대로 — [day28 RAG §3](day28_1013_rag.ipynb)

### LangGraph

- LangGraph — 체인 단계를 노드로, 연결을 직접 정함, LCEL 체인은 Sequential처럼 한 방향이라 분기를 넣으려고 씀, State·Node·Edge — [day32 LangGraph §1](day32_1019_langgraph.ipynb)
- `TypedDict` — dict의 키마다 값 타입, `class 이름(TypedDict)`, 결과는 일반 dict — [day32 LangGraph §2](day32_1019_langgraph.ipynb)
- `TypedDict` 타입 강제 없음 — 다른 타입을 넣어도 실행 에러 없음, 편집기 검사에서만 표시 — [day32 LangGraph §2 cf)](day32_1019_langgraph.ipynb)
- `add_messages` — `Annotated[list, add_messages]`, 리듀서는 (기존 값, 새 값)을 합침, 뒤에 붙이고 id가 같으면 바꿈, 리듀서 없는 키는 덮어씀 — [day32 LangGraph §3](day32_1019_langgraph.ipynb)
- 그래프 만들기 — `StateGraph(State)` → `add_node` → `add_edge(START, …)`·`add_edge(…, END)` → `compile()` → `invoke`, 결과는 State dict — [day32 LangGraph §4](day32_1019_langgraph.ipynb)
- 노드 연결 — `add_edge`로 `START → 노드1 → 노드2 → END`, 메시지가 실행 순서대로 쌓이고 다음 노드는 앞 노드까지 쌓인 State를 받음 — [day32 LangGraph §5](day32_1019_langgraph.ipynb)
- Tool Calling — `bind_tools`로 모델에 도구 이름·설명·인자를 알려줌, 모델은 `tool_calls`만 만들고 실행은 안 함, 도구가 여럿이면 이름·설명으로 고름 — [day33 LangGraph §1](day33_1020_langgraph.ipynb)
- 검색 도구 — `TavilySearch`(API 키, 본문까지 dict), `DuckDuckGoSearchRun`(키 없음, 요약문 문자열) — [day33 LangGraph §2](day33_1020_langgraph.ipynb)
- DuckDuckGo `max_results` — `DuckDuckGoSearchRun`에 넣으면 무시됨, `api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=…)`로 — [day33 LangGraph §2 cf)](day33_1020_langgraph.ipynb)
- `ToolNode` — `tool_calls`대로 도구를 실행해 `ToolMessage`로 붙임, `add_conditional_edges("chatbot", tools_condition)` + `add_edge("tools", "chatbot")`로 반복, 노드 이름은 `tools` — [day33 LangGraph §3](day33_1020_langgraph.ipynb)
- 그래프 그림 — `get_graph().draw_mermaid_png()`(웹 API, 인터넷 필요)·`draw_mermaid()`(텍스트), 실선은 `add_edge`·점선은 조건부 — [day33 LangGraph §4](day33_1020_langgraph.ipynb)
- 그래프 그림 출력 — `Image.open(BytesIO(png_data))` → `plt.imshow` → `plt.show()`, 파일 저장 없이 띄움 — [day34 LangGraph §1](day34_1021_langgraph.ipynb)
- 메모리 — `compile(checkpointer=InMemorySaver())`, `config`의 `thread_id`가 같으면 저장된 State에 이어서 실행, checkpointer가 없으면 `invoke`마다 새 State — [day34 LangGraph §2](day34_1021_langgraph.ipynb)
- fan-out / fan-in — 한 노드에서 엣지 여러 개로 같은 단계에 같이 실행, 여러 엣지를 한 노드로 모아 다음 단계에서 합침 — [day34 LangGraph §3](day34_1021_langgraph.ipynb)
- 노드가 돌려주는 키 — 같은 단계에서 리듀서 없는 같은 키를 바꾸면 `InvalidUpdateError`, State에 없는 키는 에러 없이 무시 — [day34 LangGraph §3 cf)](day34_1021_langgraph.ipynb)
- 결과 통합 노드 — 이어 붙이기 대신 앞 결과를 프롬프트에 넣어 LLM이 최종 답을 씀, 결과마다 이름을 붙이고 우선순위를 지시 — [day35 LangGraph·ML §1](day35_1022_langgraph_ml.ipynb)

### 머신러닝

- 트리 계열 모델 — 컬럼 값으로 데이터를 나눠(분기) 예측, `DecisionTree`(트리 하나)·`RandomForest`(배깅)·`GradientBoosting`(부스팅)·`XGBoost`, 분기는 손실이 가장 많이 줄어드는 기준, 스케일링 영향 없음, sklearn은 `compile` 없이 `fit` — [day35 LangGraph·ML §2](day35_1022_langgraph_ml.ipynb)
- `feature_importances_` — 트리 계열 모델이 `fit` 후 컬럼마다 매긴 중요도, 합은 1, 파라미터가 아니라 속성, feature engineering에 씀 — [day35 LangGraph·ML §3](day35_1022_langgraph_ml.ipynb)
- sklearn `score` — 분류 모델은 acc, 회귀 모델은 r2 — [day35 LangGraph·ML §3](day35_1022_langgraph_ml.ipynb)
- r2 음수 — 평균값 하나로 예측한 것보다 못함, 트리 하나는 과적합되기 쉬움 — [day35 LangGraph·ML §3 cf)](day35_1022_langgraph_ml.ipynb)
- 회귀의 `stratify` — 클래스 비율을 맞추는 옵션이라 연속값 y에는 안 씀 — [day35 LangGraph·ML §3 cf)](day35_1022_langgraph_ml.ipynb)
- XGBoost 중요도 기준 — weight(분기 횟수, F score)·gain(손실 감소 평균), `get_booster().get_score(importance_type=…)`, `plot_importance`, XGB `feature_importances_`는 gain 비율 — [day35 LangGraph·ML §4](day35_1022_langgraph_ml.ipynb)
- K-Fold 교차 검증 — k개 묶음으로 나눠 테스트셋을 번갈아 검증, 모델 일반화 성능 평가 — [day42 ML §2](day42_1102_ml.ipynb)
- 데이터셋 리스트 — 데이터셋을 리스트로 묶고 `enumerate`로 차례로 돌림, 이름 리스트를 같은 순서로 두고 `np.argmax`로 최고 모델만 뽑음 — [day45 ML §1](day45_1105_ml.ipynb)
- for문 파라미터 튜닝 — 후보 값을 리스트로 두고 이중 for문으로 모든 조합, `if score > best_score`로 갱신, test로 조합을 고르면 그 점수를 성능으로 쓸 수 없음 — [day45 ML §2](day45_1105_ml.ipynb)
- `GridSearchCV` — 모델을 감싸 조합 × fold를 전부 돌리고 최적 조합으로 다시 훈련, `parameters`는 딕셔너리 목록, `best_params_`·`best_estimator_`·`best_score_`, `verbose`·`n_jobs` — [day45 ML §3](day45_1105_ml.ipynb)
- `best_score_`와 `model.score` — 앞은 train을 나눠 만든 검증 점수, 뒤는 test 점수, 차이가 벌어지면 그 조합이 train에만 맞춰진 것 — [day45 ML §3 cf)](day45_1105_ml.ipynb)
- `RandomizedSearchCV` — 조합 전체가 아니라 `n_iter`개만 무작위로 뽑아 교차 검증, `random_state`로 뽑히는 조합 고정, 전체 최적이라는 보장은 없음 — [day46 ML §1](day46_1106_ml.ipynb)
- `refit` — 기본값은 최적 조합으로 train 전체를 다시 학습해 `best_estimator_`를 만듦, `refit=False`면 `model.score`·`model.predict`를 쓸 수 없고 `best_params_`·`best_score_`만 남음 — [day46 ML §1 cf)](day46_1106_ml.ipynb)
- `HalvingGridSearchCV` — 첫 iter에 샘플을 적게 주고 점수 낮은 조합을 버리면서 남은 조합에 샘플을 늘림, `factor`로 줄이는 비율과 늘리는 배수를 함께 정함 — [day47 ML §1](day47_1109_ml.ipynb)
- `min_resources` — 첫 iter의 샘플 수, 기본값 `'exhaust'`의 하한은 `n_splits * 2 * 클래스 수`, iter 수는 `n_required_iterations`와 `n_possible_iterations` 중 작은 쪽 — [day47 ML §1](day47_1109_ml.ipynb)
- Halving의 한계 — 초반 iter는 샘플이 적어 조합 점수가 흔들리고, 데이터가 적으면 iter를 늘릴 여유도 없어 전체를 쓰는 탐색을 씀 — [day47 ML §1 cf)](day47_1109_ml.ipynb)

### WSL·원격 접속

- WSL 배포판 — `wsl --install -d <배포판> --name <이름>`, `wsl -l -v`로 목록·`*`가 기본, `wsl -d <이름>`으로 접속, 긴 옵션은 `-` 두 개 — [day41 NPU §1](day41_1030_npu.ipynb)
- 윈도우 파일 접근 — `/mnt/c`가 C 드라이브, `explorer.exe .`로 탐색기 열기, 윈도우 프로그램은 `.exe`까지 — [day41 NPU §2](day41_1030_npu.ipynb)
- Zone.Identifier — 인터넷에서 받은 파일을 윈도우에서 복사하면 따라오는 메타데이터, 지워도 됨 — [day41 NPU §2 cf)](day41_1030_npu.ipynb)
- SSH 키 접속 — `ssh -i <키> -p <포트> <사용자>@<주소>`, 키는 `chmod 600`, 권한이 느슨하면 거부, 첫 접속은 호스트 지문 수락 후 `known_hosts`에 저장 — [day41 NPU §3](day41_1030_npu.ipynb)
- 포트 매핑 — 접속할 때 치는 포트가 서버 안의 포트와 다를 수 있음, 바깥 포트를 안의 ssh 포트(22)로 연결해 둔 것 — [day41 NPU §3 cf)](day41_1030_npu.ipynb)
- `furiosa-smi` — NPU 상태 확인, `info`·`status`·`ps`·`topo`·`version`, GPU의 `nvidia-smi`와 같은 자리 — [day41 NPU §4](day41_1030_npu.ipynb)
- `watch` — 명령을 일정 간격으로 다시 실행해 화면 갱신, `-n <초>`로 간격 지정, `Ctrl + C`로 종료 — [day41 NPU §5](day41_1030_npu.ipynb)
- 서버 사양 확인 — `lscpu`(논리 코어 = 소켓 × 코어 × 스레드, 캐시, NUMA)·`free -h`(메모리) — [day41 NPU §6](day41_1030_npu.ipynb)
- `free`의 available — `buff/cache`는 필요하면 회수되므로 `free`가 작아도 부족한 것이 아님, 실제 여유는 `available` — [day41 NPU §6 cf)](day41_1030_npu.ipynb)
- Miniconda 설치 — `curl -O`로 스크립트 받아 `bash`로 실행, 라이선스·설치 위치·`conda init`을 물어봄, `source ~/.bashrc` 후 `(base)` 표시 — [day41 NPU §7](day41_1030_npu.ipynb)
- conda 환경 — `conda create -n <이름> python=<버전>`, `conda env list`로 목록·경로, `conda activate`/`deactivate`, 환경은 설치 경로 아래 `envs/<이름>`에 생김 — [day41 NPU §8](day41_1030_npu.ipynb)
- 윈도우에서 바로 접속 — `.ssh/config`에 `Host`·`HostName`·`User`·`Port`·`IdentityFile`·`IdentitiesOnly`, 키 권한은 `icacls`로 좁힘, `known_hosts`는 WSL과 따로 — [day41 NPU §9](day41_1030_npu.ipynb)
- VS Code 원격 폴더 — `Remote - SSH` 확장 + `Ctrl + Shift + P` → `Connect to Host` → `Open Folder`, 적는 경로는 서버 쪽 — [day41 NPU §9 cf)](day41_1030_npu.ipynb)
- 미니콘다와 아나콘다 — 기능 차이는 작고 아나콘다는 패키지가 미리 깔려 있음, 큰 조직은 아나콘다에 유료 라이선스가 필요해 미니콘다를 씀 — [day41 NPU §7 cf)](day41_1030_npu.ipynb)

### 양자화·로컬 LLM

- 양자화 파이프라인 — 원본 BF16 → GGUF 변환(형식만 바꿈) → 양자화(비트 줄임), 중간 BF16 GGUF를 두고 타입만 바꿔 다시 양자화 — [day44 양자화 §1](day44_1104_quantization.ipynb)
- GGUF — llama.cpp가 쓰는 모델 파일 형식, 가중치와 토크나이저·메타데이터를 한 파일에 담음 — [day44 양자화 §1](day44_1104_quantization.ipynb)
- BPW — 가중치 하나당 평균 비트 수, 양자화 전후를 비교하는 기준 — [day44 양자화 §1](day44_1104_quantization.ipynb)
- Ollama — `curl -fsSL … | sh`로 설치하면 systemd 서비스로 등록, `ollama run <모델>:<태그>`는 없으면 받고 실행, `ollama list`로 목록, 양자화는 못 함 — [day44 양자화 §2](day44_1104_quantization.ipynb)
- llama.cpp 빌드 — `cmake -B build -DCMAKE_BUILD_TYPE=Release`로 설정 후 `cmake --build build -j <개수>`, 실행 파일은 `build/bin` 아래 — [day44 양자화 §3](day44_1104_quantization.ipynb)
- 파이썬 가상환경 — `python3 -m venv <경로>` + `source <경로>/bin/activate`, 셸 세션마다 다시 활성화해야 함, `which python3`으로 확인 — [day44 양자화 §4](day44_1104_quantization.ipynb)
- requirements.txt가 목록만 있을 때 — llama.cpp는 `-r`로 다른 파일을 불러오는 줄만 있고 실제 패키지는 `requirements/` 아래 스크립트별로 나뉨, `pip check`로 의존성 확인 — [day44 양자화 §4 cf)](day44_1104_quantization.ipynb)
- requirements의 버전 핀 — 불러오는 파일에 `==`로 박혀 있으면 수동 업그레이드가 재설치 때 되돌아가므로, 올린 뒤에는 다시 설치하지 않음 — [day44 양자화 §4 cf)](day44_1104_quantization.ipynb)
- GGUF 변환 — `convert_hf_to_gguf.py`의 `--outfile`·`--outtype`·`--remote <계정>/<모델>`, `-it`는 instruction tuned — [day44 양자화 §5](day44_1104_quantization.ipynb)
- 변환 결과 검증 — 에러 없이 끝나도 내용이 깨질 수 있어 양자화 전에 `llama-completion`으로 한 번 돌려 확인 — [day44 양자화 §5](day44_1104_quantization.ipynb)
- 변환이 막힐 때 — `ModuleNotFoundError`는 가상환경 미활성, `AttributeError: 'list' object has no attribute 'keys'`는 트랜스포머가 그 모델을 모를 때 — [day44 양자화 §5 cf)](day44_1104_quantization.ipynb)
- 디스크가 꽉 찰 때 — WSL `ext4.vhdx`가 더 못 늘어나 `Input/output error`, 가상 디스크는 자동으로 줄지 않아 `wsl --manage <배포판> --set-sparse true`로 회수 — [day44 양자화 §5 cf)](day44_1104_quantization.ipynb)
- 양자화 실행 — `llama-quantize <입력> <출력> <타입>` 인자 순서, `Q4_K_M`은 4비트 K-quant의 중간 크기, 결과 BPW는 4비트보다 큼 — [day44 양자화 §6](day44_1104_quantization.ipynb)
- 어느 단계가 깨졌는지 가리기 — `llama-completion`으로 대화 템플릿을 빼고 돌려보고, BF16과 양자화 결과를 각각 확인, `gguf_dump.py --no-tensors`로 메타데이터 확인 — [day44 양자화 §6 cf)](day44_1104_quantization.ipynb)
- GGUF의 `chat_template` — 지시 학습 모델을 제대로 변환하면 들어가는 키, 있으면 llama.cpp가 대화 모드로 바뀌고 Jinja 형식은 `--jinja` 필요 — [day44 양자화 §6 cf)](day44_1104_quantization.ipynb)

### NPU 서빙

- `furiosa_llm`의 `LLM` — 서버를 띄우지 않고 파이썬에서 모델을 NPU에 올림, `SamplingParams`로 생성 옵션, `apply_chat_template`으로 대화 형식 문자열, `generate`는 목록을 받고 목록을 돌려줌 — [day43 NPU §1](day43_1103_npu.ipynb)
- `top_p`·`top_k` — 다음 토큰 후보를 누적 확률로 자르는 값과 개수로 자르는 값, `temperature`는 분포 자체를 바꿈 — [day43 NPU §1 cf)](day43_1103_npu.ipynb)
- Tool Calling — `tools`에 함수 이름·설명·JSON 스키마를 적어 넘기면 모델이 함수와 인자를 골라 `tool_calls`로 돌려줌, 실행은 코드가 함 — [day43 NPU §2](day43_1103_npu.ipynb)
- `tool_choice` — 모델이 고르는 단계를 건너뛰고 쓸 함수를 지정, 생략하면 모델이 직접 고르거나 함수를 안 씀 — [day43 NPU §2](day43_1103_npu.ipynb)
- `fxb` — 모델을 NPU 전용 실행 파일로 미리 컴파일, `fxb build <모델> <이름>`·`fxb cache ls`·`fxb add`, `LLM(fxb=…)`이나 `serve --fxb`로 씀 — [day43 NPU §3](day43_1103_npu.ipynb)
- 교차 컴파일 도구 — `fxb build`에 arm64용 `gcc-aarch64-linux-gnu`·`libc6-dev-arm64-cross`가 필요, `--no-install-recommends`로 적어 둔 것만 설치 — [day43 NPU §3](day43_1103_npu.ipynb)
- OpenAI 호환 API — `furiosa-llm serve`로 띄운 서버에 `OpenAI(base_url=…/v1, api_key=…)`로 붙음, `client.models.list().data[0].id`로 올라간 모델 이름을 받음 — [day45 NPU §1](day45_1105_npu.ipynb)
- `serve` 옵션 — `--devices npu:<번호>`로 장치 지정, `--host 0.0.0.0`은 어느 쪽에서 온 요청도 받음, `--port`로 받을 포트 — [day45 NPU §1](day45_1105_npu.ipynb)
- `ChatOpenAI` — LangChain에서 OpenAI 호환 서버에 붙는 객체, 문자열 하나를 `invoke`로 넘기고 답변은 `content`로 꺼냄 — [day45 NPU §2](day45_1105_npu.ipynb)
- 처리량 측정 — `max_tokens`를 고정하고 `time.perf_counter()` 차이로 나눠 tokens/s, 조건을 고정해야 비교됨 — [day45 NPU §3](day45_1105_npu.ipynb)
- `temperature`·`top_p` — 다음 토큰을 고를 때 주는 무작위성과, 누적 확률로 자르는 후보 범위 — [day45 NPU §3](day45_1105_npu.ipynb)
- 스트리밍 — `stream=True`로 조각을 받아 `chunk.choices[0].delta.content`를 `print(…, end="", flush=True)`로 이어 붙임, 첫 글자가 나오는 시점만 빨라짐 — [day45 NPU §4](day45_1105_npu.ipynb)
- LangChain 스트리밍 — `llm.stream("…")`으로 조각을 받고 `chunk.content`로 꺼냄, 빈 조각을 거르는 `if`가 필요 없음 — [day45 NPU §4 cf)](day45_1105_npu.ipynb)
- 공통 설정 모듈 — 객체를 만드는 코드를 `get_llm` 함수로 빼 `common.py`에 두고 `from common import get_llm`으로 불러 씀, 주소가 바뀌면 한 군데만 고침 — [day45 NPU §5](day45_1105_npu.ipynb)
- 지연 시간 측정 — 같은 호출을 반복해 평균을 냄, 입력 길이를 늘려도 전체 시간은 출력 토큰 수가 정함(prefill은 한 번, decode는 토큰마다) — [day47 NPU §1](day47_1109_npu.ipynb)
- 첫 응답 시간 — 스트리밍 첫 조각까지의 시간, `first_time is None`으로 한 번만 기록하거나 `break`·`next(iter(...))`로 끊음 — [day47 NPU §2](day47_1109_npu.ipynb)
- `usage_metadata` — 응답에 오는 `input_tokens`·`output_tokens`·`total_tokens`와 `cache_read`, TPS는 `output_tokens`로 나눔 — [day47 NPU §3](day47_1109_npu.ipynb)
- 폴더로 나눈 뒤 import — `from furiosa1.common import get_llm`처럼 폴더 이름을 점으로 이어 적고, `__init__.py`는 없어도 됨 — [day47 NPU §3 cf)](day47_1109_npu.ipynb)
- continuous batching — 서버가 들어온 요청을 묶어 함께 처리, 동시 요청 수를 늘리면 전체 시간과 TPS는 좋아지고 요청별 지연은 늘어남 — [day48 NPU §1](day48_1110_npu.ipynb)
- `ThreadPoolExecutor` — `max_workers`로 동시 요청 수를 정하고 `pool.map`으로 질문마다 호출, `with`를 벗어날 때 모두 기다림 — [day48 NPU §1](day48_1110_npu.ipynb)
- `run_once` — 시간·토큰·본문을 딕셔너리로 돌려주는 측정 함수, `usage_metadata`가 없을 때는 `or {}`와 `.get(키, 0)`으로 받음 — [day48 NPU §2](day48_1110_npu.ipynb)
- KV Cache·prefix cache — 입력 앞부분 계산 결과를 들고 있다가 겹치는 입력에서 건너뜀, 겹치는 입력이 많을수록 적중률이 올라감 — [day48 NPU §3](day48_1110_npu.ipynb)
- `/metrics` — 서버 상태 수치를 내주는 주소, `urllib.request.urlopen`으로 읽어 줄마다 마지막 칸을 꺼냄, 쌓이는 값이라 전후 차이로 봄 — [day48 NPU §3](day48_1110_npu.ipynb)
- RPS와 TPS — 앞은 초당 요청 수, 뒤는 초당 출력 토큰 수, 순차 요청은 요청 하나의 지연에 묶이고 동시 요청은 둘이 같이 올라감 — [day49 NPU §1](day49_1111_npu.ipynb)
- `submit`과 `futures` — 요청을 하나씩 맡기고 `f.result()`로 꺼냄, 인자가 여러 개여도 되고 `max_workers`를 넘는 요청은 기다림 — [day49 NPU §2](day49_1111_npu.ipynb)
- 적중률 전후 차이 — 쌓이는 메트릭은 요청 전후를 읽어 뺀 값으로 계산, 적중해도 입력 토큰 수는 그대로 — [day49 NPU §3](day49_1111_npu.ipynb)
- 메트릭 거르기 — `requests.get`·`raise_for_status`로 받아 키워드로 줄을 남김, `num_requests_running`·`num_requests_waiting`으로 처리·대기 요청 수를 봄 — [day49 NPU §4](day49_1111_npu.ipynb)
- 스케줄링 모드 — eager는 배치를 끝낸 뒤 다음 배치를 준비, `--enable-overlap-scheduling`은 NPU가 돌아가는 동안 CPU가 준비 — [day51 NPU §1](day51_1113_npu.ipynb)
- 모드 바꿔 띄우기 — 서버를 내리지 않고 같은 장치로 다시 띄우면 `EBUSY`, 프로세스를 `kill`하고 `furiosa-smi status`로 확인 — [day51 NPU §1 cf)](day51_1113_npu.ipynb)

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
  - SVG는 viewBox 크기의 흰 배경 `<rect fill="#ffffff">`를 맨 먼저 그립니다. 배경이 투명하면 어두운 테마에서 박스 밖 글씨가 보이지 않습니다
- 줄바꿈은 `<br/>` 대신 빈 줄(문단 구분)로
