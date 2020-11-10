#MAP@7 평가 척도 구하는 코드

import numpy as np

def apk(actual, predicted, k = 7, default = 0.0 ):
    #MAP@7이므로, 최대 7개만 사용
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # 점수 매기기
    for i,p in enumerate(predicted): #p = 예측값
        if p in actual and p not in predicted[:i]: #예측 값이 정답에 있고 중복 아닌 값
            num_hits += 1.0
            score += num_hits / (i+1.0)

    #정답 값이 공백일 경우, 무조건 0.0 점을 반환
    if not actual:
        return default

    #정답의 개수(len(actual)) 로 average precision 구함
    return score / min(len(actual), k)

def mapk(actual, predicted, k =7, default = 0.0):
    #list of list인 정답 값(actual)과 예측 값(predicted)에서 고객별 Average Precision을 구하고, np.mean()을 통해 평균 계산
    return np.mean([apk(a,p,k,default) for a, p in zip(actual, predicted)])