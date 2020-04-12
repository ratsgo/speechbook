---
layout: default
title: Modeling Variation
nav_order: 1
parent: Sophisticated Models
permalink: /docs/sophisticated/variation
---

# Modeling Variation
{: .no_toc }

화자, 사투리, 노이즈 등 다양한 환경 변화에도 강건한 음성 인식 모델을 구축하는 방법을 살펴봅니다. 이 글은 [Speech and Language Processing 2nd Edition](https://www.amazon.com/Speech-Language-Processing-Daniel-Jurafsky/dp/0131873210)을 정리한 것입니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## Noise

화자가 발성할 때 주변이 조용하면 좋겠지만 잡음(noise)이 있을 수 있습니다. 자동 음성 인식(Automatic Speech Recognition) 모델을 만들 때는 화자가 잡음이 심한 환경에서도 높은 인식 성능을 유지할 수 있도록 해야 합니다. 잡음은 그 종류에 따라 크게 세 가지로 나눠 그 대응 방법을 생각해볼 수 있는데요. 차례대로 살펴보겠습니다.

**additive noise** : 엔진이나 바람 소리 같이 상대적으로 큰 변화 없이 지속되는 잡음이 있다면, 이 잡음 신호를 [시간(time) 도메인 음성 신호](https://ratsgo.github.io/speechbook/docs/fe/ft#discrete-fourier-transform)에 불필요하게 상수(constanct)처럼 더해진 **additive noise**로 간주하고 입력 신호에서 해당 잡음 신호의 평균치만큼을 빼버리는 처리(**spectral subtraction**)를 할 수 있습니다. 잡음 신호의 평균치는 음성이 아닌 구간(non-speech regions)에 대해 진폭(amplitude)들을 평균을 취해 계산합니다.

이와 관련해 **롬바드 효과(Lombard Effect)**라는 것이 있습니다. 화자가 잡음이 있는 환경에서 말을 하게 되면 자신의 뜻을 보다 명백하게 전달하기 위해 말하는 경향을 달리하게 되는데 이 때 나타나는 조음상의 변화를 가리킵니다. 화자는 잡음 환경에서 효과적인 의사 전달을 위하여 목소리(intensity)를 키우고 대체로 성도를 좁힌 상태에서 발성하는 경향이 있습니다. 음향음성학적으로는 롬바드 효과에 의하여 음성신호는 피치(기본 주파수, F0) 증가, 진폭(amplitude) 증가, 모음 길이 증가, spectral tilt 증가, 1차 포만트 주파수(F1) 증가 등 변화가 있으며 화자에 따라 그 효과가 크게 달라진다고 합니다. 음성인식기는 이러한 음질 차이에도 민감하여 이에 대한 명시적인 모델링을 하지 않을 경우 인식률이 떨어집니다.
{: .fs-3 .ls-1 .code-example }

**convolutional noise** : 채널(channel) 특성으로 생기는 잡음을 **convolutional noise**라고 합니다. 전화기, 마이크 따위가 달라지면 음성 신호 역시 바뀌는 것과 관련이 있습니다. 이러한 잡음에 대해서는 [멜 스펙트럼(Mel spectrum) 등을 시간(time) 축에 대해 평균을 취한 뒤 이를 원래 스펙트럼에서 빼주는 처리](https://ratsgo.github.io/speechbook/docs/fe/mfcc#post-processing)(**cepstral mean normalization**)로 대응합니다. 모든 음성 피처에 대해 이렇게 정규화를 수행해주면 음성 인식기를 만들 때 채널이 마치 하나로 고정된 것처럼 모델링할 수 있습니다.

**non-verbal sounds** : 신호에는 사람 말 소리 말고도 기침, 숨소리, 목 고르기, 전화 소리, 문 여닫는 소리 등 사람 말이 아닌 소리 역시 섞여 있습니다. 이 역시 음성 인식 모델 성능에 상당한 영향을 미칠 수 있습니다. 이는 해당 소리를 음소(phoneme) 취급해서 학습데이터를 구축하고 음성 인식 모델을 학습하는 것으로 대응할 수 있습니다. 

---


## Speaker Variation

화자에 따라 발성, 조음 방법이 조금씩 다르므로 가능하다면 개별 화자에 특화된 음성 인식 시스템을 만드는 것이 최선일 겁니다. 이같이 화자별로 특화된 시스템을 구축하는 것을 **화자 적응(speaker adaption**)이라고 합니다. 하지만 데이터를 화자별로 수집하기가 쉽지는 않습니다. 

그럼에도 불구하고 뚜렷이 구분되고 데이터를 충분히 확보할 수 있는 그룹이 있습니다. 바로 성별(性別)입니다. 남자, 여자 음성에 대응되는 모델 두 개 정도는 비교적 어렵지 않게 만들 수 있고 음성 인식 성능도 높일 수 있습니다. 이같이 성별에 대응하는 음성 인식 모델을 구축하는 것을 **gender-dependent acoustic modeling**이라고 합니다.

gender-dependent acoustic modeling과 별개로 [기존 음성 인식 시스템](https://ratsgo.github.io/speechbook/docs/am)에서 화자 적응 모델로 널리 쓰였던 기법이 **Maximum Likelihood Linear Regression(MLLR)**입니다. 

기존 음성 인식 시스템은 은닉마코프모델(Hidden Markov Model)과 가우시안 믹스처 모델(Gaussian Mixture Model) 조합인데요. MLLR은 이 가운데 입력 음성 신호와 음소(phoneme) 간의 관계를 추정하는 음향 모델(Acostic Model) 역할을 하는 가우시안 믹스처 모델을 소량의 개별 화자 데이터(문장 세 개 내지 10초 가량의 음성)로 튜닝하는 기법입니다. 

아시다시피 가우시안 믹스처 모델의 학습 파라메터는 평균(mean)과 공분산(covariance)인데요. 이 가운데 평균만 개별 화자 데이터에 맞도록 튜닝해 줍니다. 가우시안 믹스처 모델의 기존 평균 벡터를 $\mathbf{\mu}$, 새로 튜닝한 평균 벡터를 $\hat{ \mathbf{\mu} }$라고 할 때 MLLR은 수식1과 같습니다. 

## **수식1** Maximum Likelihood Linear Regression
{: .no_toc .text-delta }

$$\hat { \mathbf{\mu}  } =\mathbf{W} \mathbf{\mu} + \mathbf{\omega}$$


소량의 개별 화자 데이터의 우도(likelihood)를 최대화하도록 수식1의 선형변환(linear transformation) 행렬 $\mathbf{W}$과 bias 벡터 $\mathbf{\omega}$를 학습하고 학습이 끝나면 $\hat{ \mathbf{\mu} }$을 가우시안 믹스처 모델의 새 평균 벡터로 사용합니다. 우도를 계산할 때는 기존 음성 인식 시스템의 학습 기법인 [Baum-Welch Algorithm](https://ratsgo.github.io/speechbook/docs/am/baumwelch)을 사용합니다. 

이밖에 화자 적응 기법으로는 **MAP adaptation**, **Speaker Clustering**, **Vocal Tract Length Normalization** 등이 있습니다.

---

## Other Issues

대개 대화 음성을 인식하는 모델을 만드는 것이 낭독 음성 인식기보다 어렵다고 합니다. 대화할 때 화자별 발음, 조음 방법 변이가 크기 때문입니다. 뿐만 아니라 사전에 없는 단어(unseen word)를 다루는 것도 중요한 이슈가 될 수 있습니다. [기존 음성 인식 시스템](https://ratsgo.github.io/speechbook/docs/am)에서는 학습데이터에 없는 새로운 단어가 추가되었을 경우 처음부터 다시 학습을 시작하는 수밖에 없습니다. 이에 학습을 처음부터 다시 하지 않더라도 새로운 단어에 대응하는 모델을 만들기 위한 여러 방법이 제시되었습니다.


---


## References

- [Speech and Language Processing 2nd Edition](https://www.amazon.com/Speech-Language-Processing-Daniel-Jurafsky/dp/0131873210)