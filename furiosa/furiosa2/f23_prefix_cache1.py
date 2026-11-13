from common import run_once, get_cache_metrics

prefix = "NPU 추론 최적화에 대한 설명입니다." * 150

for i in range(1, 5):
    prompt = prefix + f"\n질문 {i}: KV Cache란?"

    print(f"\n========{i}회 요청========")
    result = run_once(prompt, 100)

    print("응답 시간: ", round(result["elapsed"], 3), "초")
    print("입력 토큰: ", result["input_tokens"])

    print("\nPrefix Cache 메트릭")
    get_cache_metrics()

'''
========1회 요청========
응답 시간:  1.05 초
입력 토큰:  1668

Prefix Cache 메트릭
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 1803
Prefix Cache 적중 횟수: 93
Prefix Cache 적중률: 5.16
KV Cache 사용률: 1.46

========2회 요청========
응답 시간:  0.844 초
입력 토큰:  1668

Prefix Cache 메트릭
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 3470
Prefix Cache 적중 횟수: 1750
Prefix Cache 적중률: 50.43
KV Cache 사용률: 1.50

========3회 요청========
응답 시간:  0.855 초
입력 토큰:  1668

Prefix Cache 메트릭
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 5137
Prefix Cache 적중 횟수: 3407
Prefix Cache 적중률: 66.32
KV Cache 사용률: 1.54

========4회 요청========
응답 시간:  0.885 초
입력 토큰:  1668

Prefix Cache 메트릭
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 6804
Prefix Cache 적중 횟수: 5064
Prefix Cache 적중률: 74.43
KV Cache 사용률: 1.57
'''