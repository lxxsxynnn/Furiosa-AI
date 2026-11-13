from common import run_once, get_cache_metrics

prefix = "NPU 추론 최적화에 대한 설명입니다." * 150

for i in range(1, 5):
    print(f"\n========{i}회 요청 시작========")
    prompt = prefix + f"\n질문 {i}: KV Cache란?"

    before = get_cache_metrics()    # 요청 전 캐시 상태
    result = run_once(prompt, 100)
    after = get_cache_metrics()     # 요청 후 캐시 상태

    # 이번 요청에서 증가한 횟수
    hits = after["hits"] - before["hits"]
    queries = after["queries"] - before["queries"]

    if(queries > 0):
        hit_rate = hits / queries * 100
    else:
        hit_rate = 0

    print(f"\n========{i}회 요청 결과========")

    print("응답 시간: ", round(result["elapsed"], 3), "초")
    print("입력 토큰: ", result["input_tokens"])
    print("이번 조회 횟수: ", int(queries))
    print("이번 적중 횟수: ", int(hits))
    print("이번 적중률: ", round(hit_rate, 2), "%")


'''
========1회 요청 시작========
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 6804
Prefix Cache 적중 횟수: 5064
Prefix Cache 적중률: 74.43
KV Cache 사용률: 1.57
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 8471
Prefix Cache 적중 횟수: 6731
Prefix Cache 적중률: 79.46
KV Cache 사용률: 1.60

========1회 요청 결과========
응답 시간:  0.886 초
입력 토큰:  1668
이번 조회 횟수:  1667
이번 적중 횟수:  1667
이번 적중률:  100.0 %

========2회 요청 시작========
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 8471
Prefix Cache 적중 횟수: 6731
Prefix Cache 적중률: 79.46
KV Cache 사용률: 1.60
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 10138
Prefix Cache 적중 횟수: 8398
Prefix Cache 적중률: 82.84
KV Cache 사용률: 1.63

========2회 요청 결과========
응답 시간:  0.849 초
입력 토큰:  1668
이번 조회 횟수:  1667
이번 적중 횟수:  1667
이번 적중률:  100.0 %

========3회 요청 시작========
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 10138
Prefix Cache 적중 횟수: 8398
Prefix Cache 적중률: 82.84
KV Cache 사용률: 1.63
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 11805
Prefix Cache 적중 횟수: 10065
Prefix Cache 적중률: 85.26
KV Cache 사용률: 1.66

========3회 요청 결과========
응답 시간:  0.852 초
입력 토큰:  1668
이번 조회 횟수:  1667
이번 적중 횟수:  1667
이번 적중률:  100.0 %

========4회 요청 시작========
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 11805
Prefix Cache 적중 횟수: 10065
Prefix Cache 적중률: 85.26
KV Cache 사용률: 1.66
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 13472
Prefix Cache 적중 횟수: 11732
Prefix Cache 적중률: 87.08
KV Cache 사용률: 1.69

========4회 요청 결과========
응답 시간:  0.978 초
입력 토큰:  1668
이번 조회 횟수:  1667
이번 적중 횟수:  1667
이번 적중률:  100.0 %
'''