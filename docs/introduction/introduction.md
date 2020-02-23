---
layout: default
title: Introduction
nav_order: 2
permalink: /docs/introduction
---

# Automatic Speech Recognition
{: .no_toc }

자동 음성 인식(Automatic Speech Recognition)의 문제 정의와 아키텍처 전반을 소개합니다. 자동 음성 인식 모델은 크게 음향 모델(Acoustic Model)과 언어 모델(Language Model)로 구성되는데요. 음향 모델의 경우 기존에는 '히든 마코프 모델(Hidden Markov Model)과 가우시안 믹스처 모델(Gaussian Mixture Model)', 언어 모델은 통계 기반 n-gram 모델이 주로 쓰였습니다. 최근에는 딥러닝 기반 기법들이 주목 받고 있습니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## Problem Setting

자동 음성 인식(Automatic Speech Recognition)이란 음성 신호(acoustic signal)를 단어(word) 혹은 음소(phoneme) 시퀀스로 변환하는 시스템을 가리킵니다. 사람 말소리를 텍스트로 바꾸는 모델(Speech to Text model)이라고 말할 수 있겠습니다.

자동 음성 인식 모델은 입력 음성 신호 $X$($x_1, x_2, ..., x_t$)에 대해 가장 그럴듯한(likely) 음소/단어 시퀀스 $Y$($y_1, y_2, ..., y_n$)를 추정합니다. 자동 음성 인식 모델의 목표는 $P(Y\|X)$를 최대화하는 음소/단어 시퀀스 $Y$를 추론(inference)하는 데에 있습니다. 이를 식으로 표현하면 수식1과 같습니다.


## **수식1** Automatic Speech Recognition (1)
{: .no_toc .text-delta }

$$\DeclareMathOperator*{\argmax}{argmax} \hat { Y } =\argmax_{ Y }{ P(Y|X) }$$



$P(Y\|X)$를 바로 추정하는 모델을 구축하는 것이 가장 이상적입니다. 하지만 같은 음소나 단어라 하더라도 사람마다 발음하는 양상이 다릅니다. 화자가 남성이냐 여성이냐에 따라서도 음성 신호는 달라질 수 있습니다. 다시 말해 음성 신호의 다양한 변이형을 모두 커버하는 모델을 만들기가 쉽지 않다는 것입니다. 이에 베이즈 정리(Bayes' Theorem)를 활용해 수식2처럼 문제를 다시 정의합니다. 


## **수식2** Automatic Speech Recognition (2)
{: .no_toc .text-delta }

$$\hat { Y } =\argmax_{ Y }{ \frac { P(X|Y)P(Y) }{ P(X) }  }$$


수식2의 우변에 등장한 $P(X)$는 [베이즈 정리](https://ratsgo.github.io/statistics/2017/07/01/bayes)에서 `evidence`로 불립니다. `evidence`는 $Y$의 모든 경우의 수에 해당하는 $X$의 발생 확률이기 때문에 추정하기가 매우 어렵습니다. 그런데 다행히 추론(inference) 과정에서 입력 신호 $X$는 $Y$와 관계없이 고정되어 있습니다. 따라서 추론 과정에서 $P(X)$를 계산에서 생략할 수 있습니다. $Y$의 후보 시퀀스가 2가지($Y_1, Y_2$)뿐이라면 수식3처럼 예측 결과($Y_1$)를 만들 때 분자만 고려하면 됩니다.


## **수식3** ASR inference
{: .no_toc .text-delta }

$$\frac { P(X|{ Y }_{ 1 })P({ Y }_{ 1 }) }{ P(X) } >\frac { P(X|{ Y }_{ 2 })P({ Y }_{ 2 }) }{ P(X) }$$


결론적으로 음성 인식 모델은 수식4처럼 크게 두 가지 컴포넌트로 구성됩니다. 수식4 우변의 첫번째 항 $P(X\|Y)$는 음향 모델(Acoustic Model), $P(Y)$는 언어 모델(Language Model)로 불립니다. 음향 모델은 '음소/단어 시퀀스'와 '입력 음성 신호'가 어느 정도 관계를 맺고 있는지 추출하고, 언어 모델은 해당 음소/단어 시퀀스가 얼마나 자연스러운지 확률값 형태로 나타냅니다.


## **수식4** Automatic Speech Recognition (3)
{: .no_toc .text-delta }

$$\hat { Y } =\argmax_{ Y }{ P(X|Y)P(Y) }$$


---


## Architecture

그림1은 자동 음성 인식 모델의 전체 아키텍처를 도식화한 것입니다. 음향 모델(Acoustic Model)은 $P(X\|Y)$를 반환합니다. 음향 모델은 음소(또는 단어) 시퀀스 $Y$가 주어졌을 때 입력 음성 신호 시퀀스 $X$가 나타날 확률을 부여한다는 이야기입니다(그림2 참조). 이를 바꿔 이해하면 음향 모델은 음성 신호와 음소(또는 단어)와의 관계를 표현(`represent the relationship between an audio signal and the phonemes or other linguistic units that make up speech`)하는 역할을 담당합니다. 기존 자동 음성 인식 모델에서 음향 모델은 [히든 마코프 모델(Hidden Markov Model, HMM)](https://ratsgo.github.io/speechbook/docs/am/hmm)과 [가우시안 믹스처 모델(Gaussian Mixture Model, GMM)](https://ratsgo.github.io/speechbook/docs/am/gmm) 조합이 자주 쓰였습니다.


## **그림1** HMM, GMM 기반 음성인식 모델 
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pfVpxL7.png" width="600px" title="source: imgur.com" />

## **그림2** Acoustic Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/OsnYPSv.png" width="500px" title="source: imgur.com" />


한편 언어 모델(Language Model)은 음소(또는 단어) 시퀀스에 대한 확률 분포(`a probability distribution over sequences of words`)입니다. 다시 말해 음소(또는 단어) 시퀀스 $Y$가 얼마나 그럴듯한지(likely)에 관한 정보, 즉 $P(Y)$를 반환합니다. 기존 자동 음성 인식 모델에서 언어 모델은 통계 기반 n-gram 모델이 자주 쓰였습니다.

딥러닝(Deep Learning)이 대세가 되면서 그림1의 각 컴포넌트들이 딥러닝 기법으로 대체되고 있습니다. 음향 특징 추출(Acoustic Feature Extraction), 음향 모델, 언어모델 등 거의 모든 컴포넌트가 딥러닝으로 바뀌는 추세입니다. 딥러닝 기반 특징 추출 기법과 음향 모델은 각각 [Neural Feature Extraction](https://ratsgo.github.io/speechbook/docs/neuralfe), [Neural Acoustic Models](https://ratsgo.github.io/speechbook/docs/neuralam)을 참고하시면 되겠습니다.


## **그림2** 딥러닝 기반 음성인식 모델 
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3efQFu8.png" width="600px" title="source: imgur.com" />


최근에는 그림2보다 한발 더 나아가 수식1의 $P(Y\|X)$을 바로 추정하는 엔드투엔드(end-to-end) 자동 음성 인식 모델 역시 제안되고 있습니다. 엔드투엔드 자동 음성 인식 모델과 관련해서는 [End-to-end Models](https://ratsgo.github.io/speechbook/docs/e2e) 항목을 보시면 좋을 것 같습니다. 


---


## Acoustic Features


기존 자동 음성 인식 모델의 주요 컴포넌트인 'HMM+GMM'이 사용하는 음향 특징(Acoustic Feture)이 바로 [MFCCs(Mel-Frequency Cepstral Coefficients)](https://ratsgo.github.io/speechbook/docs/fe/mfcc)입니다. 사람이 잘 인식하는 말소리 특성을 부각시키고 그렇지 않은 특성은 생략하거나 감소시킨 피처(feature)입니다. 피처를 만드는 과정은 사전에 정의된 수식에 따라 진행됩니다. 즉 연구자들이 한땀한땀 만들어낸 룰(rule)에 기반한 피처라고 할 수 있겠습니다. MFCC 추출 과정은 그림3과 같습니다.


## **그림3** MFCC
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Pn5LGTk.png" width="600px" title="source: imgur.com" />


그림2에서 제시된 것처럼 음향 특징 추출도 딥러닝으로 대체되는 추세입니다. [Wav2Vec](https://ratsgo.github.io/speechbook/docs/neuralfe/wav2vec), [SincNet](https://ratsgo.github.io/speechbook/docs/neuralfe/sincnet) 등 다양한 기법이 제시되었습니다. 그림4는 SincNet을 도식화한 것입니다. 입력 음성 신호에 다양한 싱크 함수(sinc function)을 통과시켜 문제 해결에 도움이 되는 주파수 영역대를 부각시키고 나머지는 버립니다. 이때 각 싱크 함수가 주로 관장하는 주파수 영역대가 학습 대상(trainable parameter)이 되는데요. 룰 기반 피처인 MFCC와 달리 딥러닝 기반 음향 특징 추출 기법들은 그 과정이 결정적(deterministic)이지 않고 확률적(probabilistic)입니다.


## **그림4** SincNet
{: .no_toc .text-delta }
<img src="https://i.imgur.com/n1EXsWV.png" width="400px" title="source: imgur.com" />


---