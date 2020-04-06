---
layout: default
title: Context-Dependent AM
nav_order: 5
parent: Acoustic Models
permalink: /docs/am/cdam
---

# Context-Dependent Acoustic Models
{: .no_toc }

주변 음소 정보까지 모델링에 고려하는 Context-Dependent Acoustic Model을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## Motivation

그림1은 영어 모음(vowel) `eh`의 스펙트럼(spectrum)을 나타낸 것입니다. 첫번째는 `WED`의 `eh`, 두번째는 `YELL`의 `eh`, 세번째는 `BEN`의 `eh`입니다. 같은 모음이지만 앞뒤(context)에 어떤 음소가 오느냐에 따라 그 특질이 확연하게 달라짐을 확인할 수 있습니다. [기존 은닉 마코프 모델(Hidden Markov Model) 기반 음성 인식 모델](https://ratsgo.github.io/speechbook/docs/am/legacy)에서는 상태(state)를 음소보다 작은 단위의 subphone으로 모델링하고 있는데요. 그림1과 같이 동일한 음소라도 그 특징이 크게 다르다면 인식 품질이 확 낮아지게 될 겁니다.

## **그림1** CD Phone Motivation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/mDaoZcN.png" width="400px" title="source: imgur.com" />

이에 도입된 개념이 **Context Dependent Phone(CD Phone)**입니다. 앞뒤 컨텍스트를 반영하는 것입니다. 예컨대 `YELL`의 `eh`을 봅시다. 이 `eh`는 그 이전에 `y`, 그 이후에 `l`이 등장합니다. 이에 `YELL`의 `eh`를 `y-eh+l`로 표현하게 됩니다. 마찬가지로 `WED`의 `eh`는 `e-eh+d`, `BEN`의 `eh`는 `b-eh+n`으로 표시해 `YELL`의 `eh`와 구별합니다. CD Phone을 모델링할 때 기준 음소를 포함해 그 앞뒤 음소 총 3개를 고려하는 triphone을 일반적으로 사용합니다. 

CD phone은 subphone과는 대비되는 개념입니다. CD Phone 모델링시 음소보다 작은 단위의 subphone을 쓸 수 있지만 그 자체로 CD phone이 되는 것은 아닙니다. CD Phone을 모델링한다는 것은 '음소보다 작은 단위를 쓴다'에 강조점이 있는 것이 아니라 '앞뒤 컨텍스트를 고려한다'에 방점이 찍혀 있기 때문입니다. 한편 CD Phone의 대척점에 Context-Independent Phone이 있습니다. 음소 모델링시 앞뒤 컨텍스트를 고려하지 않고 독립적으로 본다는 뜻입니다.

---

## Subphone Clustering

문제는 이렇게 CD phone을 상정하게 되면 고려해야 하는 경우의 수가 폭증한다는 점에 있습니다. 50개 음소가 있는 언어이고 CD phone을 기준 음소와 그 앞뒤 음소 총 3개(triphone)으로 모델링한다면 우리는 $50^3=125000$개의 CI Phone을 염두에 두어야 합니다. 이에 그림2의 `iy`와 같이 비슷한 특징을 가지는 CD phone을 하나로 합쳐 고려하는 **Subphone Clustering**이 제시됐습니다. 아래처럼 비슷한 CD phone은 동일한 상태(state)로 간주해 계산량을 줄이는 기법입니다.


## **그림2** Subphone Clustering Motivation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/9rkLtL6.png" width="400px" title="source: imgur.com" />

[기존 음성 인식 시스템](https://ratsgo.github.io/speechbook/docs/am/legacy)에서는 상태(state)가 주어졌을 때 관측치(observation)가 나타날 확률, 즉 방출(emission) 확률 함수를 [가우시안 믹스처 모델(Gaussian Mixture Model)](https://ratsgo.github.io/speechbook/docs/am/gmm)로 모델링합니다. 상태 개수만큼의 가우시안 믹스처 모델이 필요합니다. 앞서 언급했듯 CD Phone을 상정하게 되면 학습해야 하는 가우시안 믹스처 모델이 기하급수적으로 많아지게 됩니다. Subphone Clustering을 실시하게 되면 비슷한 특질을 가지는 CD phone을 묶을(tying) 수 있고 가우시안 믹스처 모델 역시 공유할 수 있게 됩니다. 그림3과 같습니다.

## **그림3** Subphone Clustering
{: .no_toc .text-delta }
<img src="https://i.imgur.com/RbFFhEc.png" width="400px" title="source: imgur.com" />

그림4는 `iy`를 CD Phone으로 모델링할 때 클러스터링 절차를 도식적으로 나타낸 것입니다. (1) 처음에는 `iy`를 Context-Independent Phone이라고 보고 은닉마코프모델 + 가우시안 믹스처 모델을 학습합니다. (2) `iy`를 CD Phone으로 모델링한 은닉마코프모델 + 가우시안 믹스처 모델을 구축하되, `iy`에 관련된 모든 CD phone에 대응되는 가우시안 믹스처 모델의 초기값으로 (1)에서 학습한 가우시안 믹스처 모델의 파라메터(평균, 공분산)를 줍니다.

## **그림4** Clustering Process
{: .no_toc .text-delta }
<img src="https://i.imgur.com/XyvDYuo.png" title="source: imgur.com" />

(3)에서는 비슷한 특질을 가지는 CD Phone끼리 클러스터링을 수행하고 그 결과에 따라 가우시안 믹스처 모델 파라메터를 공유할지(tying) 말지 결정해 줍니다. 마지막으로 (4)에서는 (3) 결과를 바탕으로 은닉마코프모델 + 가우시안 믹스처 모델을 다시 학습합니다.

---

## Decision Tree

Subphone Clustering에 사용되는 군집화 알고리즘은 바로 [의사결정나무(Decision Tree)](https://ratsgo.github.io/machine%20learning/2017/03/26/tree)입니다. 음성 피처(MFCC)가 주어졌을 때 은닉마코프모델 + 가우시안 믹스처 모델의 우도(likelihood)가 높아지면 분기(split)하도록 학습합니다. 그림5는 이렇게 학습된 의사결정나무 모델의 예시를 나타낸 그림입니다.

## **그림5** Decision Tree
{: .no_toc .text-delta }
<img src="https://i.imgur.com/RyM3Ei1.png" width="400px" title="source: imgur.com" />

의사결정나무 분기 조건은 그림6과 같이 음운론, 음성학 전문가들이 추출해 놓은 음성 특징을 기준으로 합니다.

## **그림6** Phonetic Features
{: .no_toc .text-delta }
<img src="https://i.imgur.com/EBToHMp.png" width="400px" title="source: imgur.com" />




---

## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3)
- [Stanford CS224S - Spoken Language Processing](https://web.stanford.edu/class/cs224s)


---