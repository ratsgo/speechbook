---
layout: default
title: Discriminative Training
nav_order: 6
parent: Acoustic Models
permalink: /docs/am/discriminative
---

# Discriminative Training
{: .no_toc }

Discriminative Training에 대해 살펴봅니다. Discriminative Training은 최대우도추정(Maximum Likelihood Estimation)의 단점을 보완하기 위해 제시된 학습 방법인데요. 데이터($X$)를 잘 설명하는 모델을 찾는 데 관심을 두는 최대우도추정**($P(X\|Y)$를 최대화)**과는 달리 답($Y$)을 잘 찾는 모델을 학습**($P(Y\|X)$를 최대화)**하는 데 초점을 둡니다. 이 글에서는 Discriminative Training 기법 두 가지를 설명하고자 합니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## Maximum Mutual Information Estimation

최대우도추정(Maximum Likelihood Estimation)을 식으로 나타내면 수식1과 같습니다. $\lambda$는 우리가 찾으려는 모델(Hidden Markov Model + Gaussian Mixture Model의 파라메터)입니다. $O$는 관측치(observation) 시퀀스입니다. [MFCC](https://ratsgo.github.io/speechbook/docs/fe/mfcc) 같은 음성피처라고 보시면 될 것 같습니다. 마지막으로 $M_k$는 특정 단어 시퀀스($W_k$)에 대응하는 은닉마코프모델인데요. 해당 단어 시퀀스를 은닉마코프모델의 상태(state), 즉 subphone 시퀀스로 변환해놓은 것이라고 이해했습니다(예: $M_{ \text{go} }$=[g1, g2, g3, o1, o2, o3]).


## **수식1** Maximum Likelihood Estimation
{: .no_toc .text-delta }

$$\hat { \lambda  } _{ \text{ MLE } }={ P }_{ \lambda  }\left( O|{ M }_{ k } \right)$$


우리는 음성 인식 모델이 출력한 결과가 최대한 정답과 비슷하기를 원합니다. 하지만 수식1로 찾은 모델 $\lambda$는 이를 보장하지 않습니다. $M_k$가 주어졌을 때 관측치 시퀀스 $O$가 나타날 확률을 최대화할 뿐이기 때문입니다. 다시 말해 $O$를 잘 설명하는 $\lambda$가 최대우도추정의 결과물입니다. 이때문에 수식2와 같은 **Maximum Mutual Information Estimation(MMIE)** 기법이 제안됐습니다. 

수식2는 **Conditional Maximum Likelihood Estimation(CMLE)**이라고도 불립니다. $M_k$가 고정되어 있을 경우 **$M_k$와 $O$ 사이의 상호정보량(Mutual Information)을 최대화**하는 것과 **조건부 확률(사후확률)인 $P(M_k\|O)$를 최대화**하는 것이 동치(equivalent)라고 합니다.


## **수식2** Maximum Mutual Information Estimation (1)
{: .no_toc .text-delta }

$$\hat { \lambda  } _{ \text{ MMIE } }={ P }_{ \lambda  }\left( { M }_{ k } | O \right)=\frac { { P }_{ \lambda  }\left( O|{ M }_{ k } \right) P\left( { M }_{ k } \right)  }{ { P }_{ \lambda  }\left( O \right)  } $$


수식2에서 확인할 수 있듯 관측치 시퀀스 $O$가 주어졌을 때 정답 상태 시퀀스 $M_k$가 나타날 확률을 최대화하는 모델 $\lambda$를 찾는 게 MMIE의 목표입니다. 다시 말해 '정답을 잘 맞춰야 한다'는 목표를 명시적으로 모델에 부여한 셈입니다. 수식2를 [전확률공식(law of total probability)](https://ko.wikipedia.org/wiki/%EC%A0%84%EC%B2%B4_%ED%99%95%EB%A5%A0%EC%9D%98_%EB%B2%95%EC%B9%99)에 따라 다시 쓰면 수식3과 같습니다. $\mathcal{L}$은 상정 가능한 모든 단어 시퀀스 집합입니다.


## **수식3** Maximum Mutual Information Estimation (2)
{: .no_toc .text-delta }

$$\hat { \lambda  } _{ \text{ MMIE } }={ P }_{ \lambda  }\left( { M }_{ k }|O \right) =\frac { { P }_{ \lambda  }\left( O|{ M }_{ k } \right) P\left( { M }_{ k } \right)  }{ \sum _{ M\in \mathcal{L} }^{  }{ { P }_{ \lambda  }\left( O|M \right) P\left( M \right)  }  } $$


수식3을 최대화하려면 우변의 분자는 키우고 분모는 줄여야 합니다. 우변의 분자는 정답과 관측치가 동시에 나타날 결합확률(joint probability), 즉 $P_{\lambda}(O, M_k)$인데요. 정답 케이스에 대해서는 이 값을 높입니다. 반대로 나머지 오답 케이스들에 대해서는 값을 줄여야 분모를 작게할 수 있을 겁니다. 요컨대 **MMIE는 모델로 하여금 정답과 오답을 분명하게 구분(discriminating)하도록 유도**한다는 것입니다. 그저 데이터를 잘 설명하는 데 관심을 두는 최대우도추정과는 분명히 다릅니다.

은닉마코프모델은 대개 [Baum-Welch Algorithm](https://ratsgo.github.io/speechbook/docs/am/baumwelch)으로 학습을 하는데요. 여기에 최대우도추정 대신 MMIE를 적용한 기법을 **Extended Baum-Welch Algorighm**이라고 합니다. 크게 두 개 단계를 거칩니다.

- 기존 Baum-Welch Alogorithm 그대로 정답 단어 시퀀스 기준으로 전방/후방 확률을 계산한다.
- 오답 시퀀스에 대해서도 전방/후방 확률을 계산한 뒤 앞에서 계산한 정답 시퀀스에 대한 확률에서 이를 빼준다.

기존의 Baum-Welch Algorithm에서는 정답 단어 시퀀스만을 고려했는데요. Extended Baum-Welch Algorighm에서는 정답을 제외한 **모든 오답** 시퀀스를 계산해야 하기 때문에 시간 복잡도가 매우 높습니다. 그래서 실제로는 [Word Lattice](http://ratsgo.github.io/speechbook/docs/decoding/multipass#word-lattice)에 등장하는 오답 시퀀스만을 대상으로 전방/후방확률을 계산해 줍니다. Word Lattice에 등장했다는 것은 후보 시퀀스들 가운데 확률값이 높은 $N$개 중 하나에 들었다는 뜻인데요. 모델이 마치 정답처럼 헷갈리는 오답 시퀀스들만을 대상으로 한다는 이야기가 됩니다. 다시 말해 자주 틀리는 걸 정리한 오답노트를 집중적으로 학습하는 셈이 되는 것이죠. 

---


## Acoustic Models based on Posterior Classifiers

[HMM-GMM 구조](https://ratsgo.github.io/speechbook/docs/am/legacy)에서 [가우시안 믹스처 모델(Gaussian Mixture Model)](https://ratsgo.github.io/speechbook/docs/am/gmm)은 우도(likelihood)인 $P(o_t\|q_j)$를 계산합니다. $j$번째 상태(state)일 때 $t$번째 관측치 $o_t$가 나타날 확률입니다. 이를 프레임 레벨의 분류기(classifier)로 대체할 수 있습니다. 분류기는 사후확률(posterior)인 $P(q_j\|o_t)$를 계산하는데요. $o_t$가 어떤 상태인지 분류하는 역할을 수행합니다. 분류기는 [Discriminative Model](https://en.wikipedia.org/wiki/Discriminative_model)이기 때문에 이 방식 역시 Discriminative Training의 일종이 됩니다.

분류기의 입/출력은 가우시안 믹스처 모델보다 유연하게 설정할 수 있습니다. 프레임 하나만 입력으로 들어가는 가우시안 믹스처 모델과 달리 분류기의 입력은 주변 프레임을 모두 쓸 수 있습니다. 출력으로 은닉마코프모델의 상태(state, 즉 subphone)만 적용할 수 있는 가우시안 믹스처 모델과 달리 분류기의 출력은 이보다 더 큰 단위인 음소(phone)로 모델링할 수 있습니다. 서포트벡터머신(Support Vector Machine)이나 멀티레이어 퍼셉트론(MultiLayer Perceptron) 등이 분류기로 쓰일 수 있습니다.  

HMM-GMM 구조에서 가우시안 믹스처 모델을 분류기로 대체하기 위해서는 분류기 출력(posterior)을 가우시안 믹스처의 출력(likelihood)처럼 변환할 필요가 있습니다. 수식4의 첫번째 행은 가우시안 믹스처 모델의 출력인 우도를 조건부 확률의 정의를 활용해 다시 적은 것입니다. 이 식 양변을 $P(q_j)$로 나눠준 결과가 수식4의 두번째 행입니다.


## **수식4** Scaled Likelihood
{: .no_toc .text-delta }


$$P\left( { q }_{ j }|{ o }_{ t } \right) =\frac { P\left( { o }_{ t }|{ q }_{ j } \right) P\left( { q }_{ j } \right)  }{ P\left( { o }_{ t } \right)  } \\ \frac { P\left( { o }_{ t }|{ q }_{ j } \right)  }{ P\left( { o }_{ t } \right)  } =\frac { P\left( q_{ j }|{ o }_{ t } \right)  }{ P\left( { q }_{ j } \right)  } $$


수식4 두번째 행 우변의 분자는 분류기의 출력입니다. 우변 분모는 상태의 전확률(total probability)로 시간(time step) 전체에 걸쳐 분류기가 내놓는 $j$번째 상태 확률을 모두 더한 값이 됩니다. 우변 분자와 분모는 모두 분류기로부터 계산할 수 있습니다.

수식4 두번째 행 좌변의 분자는 대체 대상인 우도 값입니다. 그런데 식을 보면 이 우도가 $P(o_t)$로 나눠진 것을 확인할 수 있습니다. 수식4의 두번째 행이 **Scaled Likelihood**라고 불리는 이유입니다. 

그런데 $P(o_t)$는 상태(state)와 관계 없이 입력이 고정되면 그 값이 일정할 것이므로 음성 인식(디코딩) 결과에 영향을 미치지 않습니다. 다시 말해 분류기로부터 나온 두번째 행 우변의 값을 가우시안 믹스처 모델 출력으로 대체해도 무방하다는 것입니다.

분류기를 학습하려면 프레임 단위의 레이블(label)을 가지고 있어야 합니다. 하지만 우리는 학습데이터로 웨이브(wave) 파일과 그에 대응하는 스크립트(transcript)만 확보하고 있을 뿐입니다. 보통 음성인식에서 25ms 정도 길이의 음성 입력을 하나의 프레임(시프트는 10ms)으로 적용하는데요. 이 정도로 세밀한 레이블을 만드는 것은 고비용일 뿐더러 사람은 음소 단위로 인식하기 때문에 ms 단위의 매우 짧은 시간의 음성에 대해 subphone 단위로 레이블링한 결과도 신뢰하기 어렵습니다.

분류기의 레이블은 HMM-GMM 구조를 학습하면서 만들 수 있습니다. 학습 과정에서 각각의 입력 음성 프레임이 어떤 상태일지 추정이 되기 때문입니다. 이른바 [Embedded Training](https://ratsgo.github.io/speechbook/docs/am/terms#embedded-training)입니다. 우선 랜덤에 가까운 레이블로 분류기를 학습하고, 이 분류기가 내놓는 Scaled Likelihood로 은닉마코프모델을 학습합니다. 학습된 은닉마코프모델이 내놓는 결과(입력 음성 프레임-추정 상태)를 가지고 다시 분류기를 학습합니다. 이를 반복하는 방식입니다(iterative method).


---


## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3)
- [Stanford CS224S - Spoken Language Processing](https://web.stanford.edu/class/cs224s)


---