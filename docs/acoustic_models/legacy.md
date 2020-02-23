---
layout: default
title: Legacy Acoustic Model
nav_order: 4
parent: Acoustic Models
permalink: /docs/am/legacy
---

# Legacy Acoustic Model
{: .no_toc }

기존 음성 인식 모델의 근간이었던 은닉마코프모델(Hidden Markov Model) + 가우시안 믹스처 모델(Gaussian Mixture Model)에 대해 살펴봅니다. 
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---


## Hidden Markov Model

[은닉 마코프 모델(Hidden Markov Model)](https://ratsgo.github.io/speechbook/docs/am/hmm)에서는 일반적으로 상태 전이에 대한 제약을 두지 않습니다. 하지만 은닉 마코프 모델을 음성 인식에 적용할 때는 `left-to-right` 제약을 둡니다. 다시 말해 현재 시점 기준으로 이후 상태로 전이만 가능하게 할 뿐, 이전 시점 상태로의 전이는 허용하지 않는다는 것입니다. 음성 인식을 위한 은닉 마코프 모델에서 은닉 상태(hidden state)는 음소(또는 단어)가 될텐데요. 음소(또는 단어)는 시간에 대해 불가역적이기 때문에 이렇게 모델링한 것 아닌가 싶습니다. 그림1을 봅시다.


## **그림1** 음성 인식을 위한 HMM 모델링
{: .no_toc .text-delta }
<img src="https://i.imgur.com/0xXAVXX.png" title="source: imgur.com" />


그림1을 자세히 보면 `self-loop`를 허용하고 있는 점 역시 눈에 띕니다. 화자나 상황에 따라서 특정 음소를 길게 발음할 수도 있고 짧게 말할 수 있는 점을 고려한 모델링입니다. 만약 특정 음소를 길게 발음하는 데이터에 대해서는 동일한 은닉 상태를 반복하는 형태로 학습(train)/추론(inference)하게 되는 것입니다. 반대로 짧게 발음하는 화자에 대해서는 그 다음 상태(음소)로 전이하도록 합니다.

또다른 특이점은 음소를 더 작은 subphone으로 모델링하고 있다는 점입니다. 그림1에서는 음소 하나를 3개의 subphone으로 나누었는데요. 이는 음소의 음성 패턴과 관계가 있습니다. 그림2는 영어 모음 [ay, k]의 스펙트로그램을 나타냅니다. 'ay'의 경우 동일한 음소인데도 시간 변화에 따라 스펙트로그램이 달라지고 있음을 확인할 수 있습니다. 'ay'를 하나의 은닉 상태로 모델링하는 것보다는 여러 개로 나누어 모델링하는 것이 음성 인식 성능 향상에 도움이 될 겁니다.


## **그림2** 영어 모음 [ay, k]의 스펙트로그램
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3wFS9DA.png" width="500px" title="source: imgur.com" />


---


## Gaussian Mixture Model

히든 마코프 모델을 음성 인식에 적용하려면 continous 입력에 대한 고려가 필요합니다. 그도 그럴 것이 음성 인식의 입력값은 음성 신호가 될텐데요. 이 입력값은 히든 마코프 모델 프레임워크에서 관측치(observation) 역할을 합니다. 히든 마코프 모델 프레임워크에 사용되는 입력 음성 신호는 [MFCCs](https://ratsgo.github.io/speechbook/docs/fe/mfcc)입니다.

히든 마코프 모델의 주요 컴포넌트는 전이 확률(transition probability)과 방출 확률(emission probability)입니다. continous 입력을 고려한다면 이 가운데 방출 확률이 문제가 됩니다. 예컨대 그림3에서 은닉 상태(음소) 'sh'가 주어졌을 때 해당 관측치(MFCCs)가 나타날 확률, 즉 방출 확률을 구하려면 continous 입력에 확률값을 내어주는 모델(probability model)이 필요합니다.


## **그림3** Gaussian Mixture Model의 역할
{: .no_toc .text-delta }
<img src="https://i.imgur.com/OsnYPSv.png" width="500px" title="source: imgur.com" />


이때 요긴하게 쓰일 수 있는 것이 다변량 가우시안 분포(Multivariate Gaussian Distibution)입니다. 정규분포를 다차원 공간에 확장한 분포로 고차원 continous 입력에 대해 확률값을 부여할 수 있습니다. [가우시안 믹스처 모델(Gaussian Mixture Model)](https://ratsgo.github.io/speechbook/docs/am/gmm)은 한발 더 나아가 데이터가 여러 개의 정규분포로부터 생성되었다고 가정하는 모델입니다. 각 상태(음소)별 MFCC의 분포를 단일 가우시안 모델보다 좀 더 유려하게 모델링할 수 있는 장점이 있습니다.

음성 인식을 위한 히든 마코프 모델 프레임워크에서 가우시안 믹스처 모델은 히든 마코프 모델과 뗄레야 뗄 수 없는 관계를 가지고 있습니다. 가우시안 믹스처 모델이 방출 확률을 부여하기 때문입니다. 그림4를 두고 예를 들면, 상태(음소)가 'sh'일 때는 그림4의 왼쪽 가우시안 믹스처 모델에, 'k'라면 오른쪽 모델에 입력해 확률값을 계산하는 방식입니다. 따라서 음성 인식을 위한 히든 마코프 모델 프레임워크에서 은닉 상태(음소) 개수만큼의 가우시안 믹스처 모델이 필요하게 됩니다. 


## **그림4** Gaussian Mixture Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/2vCewl8.png" width="500px" title="source: imgur.com" />


---


## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3)
- [Speech Recognition — GMM, HMM](https://medium.com/@jonathan_hui/speech-recognition-gmm-hmm-8bb5eff8b196)
- [Stanford CS224S - Spoken Language Processing](https://web.stanford.edu/class/cs224s)


---