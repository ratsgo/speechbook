---
layout: default
title: Concepts
nav_order: 1
parent: Decoding strategy
permalink: /docs/decoding/concepts
---

# Concepts
{: .no_toc }

학습이 완료된 음성인식 모델을 가지고 입력 음성이 주어졌을 때 최적의 음소/단어 시퀀스를 디코딩하는 방법을 알아봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## Acoustic + Language Model

대부분의 음성 인식 시스템에서는 디코딩시 음향 모델(Acoustic Model)과 언어 모델(Language Model)을 함께 씁니다. 음향모델을 우도(likelihood) 함수로, 언어 모델을 사전확률(prior) 함수로 보고 베이지안 관점에서 모델링하는 것입니다. 수식1과 같습니다.


## **수식1** 음성인식 모델의 베이지안 모델링 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/aS65SE9.png" width="200px" title="source: imgur.com" />


길이에 따른 패널티가 없도록 없도록 변경


## **수식2** 음성인식 모델의 베이지안 모델링 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/5dYX3xf.png" width="230px" title="source: imgur.com" />


## **수식3** 음성인식 모델의 베이지안 모델링 (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/t29krA6.png" width="250px" title="source: imgur.com" />


## **수식4** 음성인식 모델의 베이지안 모델링 (4)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Aj2Hkoi.png" width="350px" title="source: imgur.com" />


---

## Hidden Markov Model based


기존 음성 인식은 은닉마코프모델(Hidden Markov Model) 기반인 경우가 많습니다. 은닉마코프모델에 가우시안 믹스처 모델(Gaussian Mixture Model)을 결합하거나, 은닉마코프모델과 딥러닝을 함께 쓰는 방식 등이 바로 그것입니다. 

그림1은 은닉마코프모델 기반 음성 인식 모델의 디코딩 전반을 도식화한 그림입니다.


## **그림1** HMM 기반 디코딩
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bpoTguW.png" width="600px" title="source: imgur.com" />


---

## E2E model based

최근에는 언어모델의 도움을 받지 않고 엔드투엔드로 접근하려는 시도도 계속되고 있습니다.


## **수식5** E2E model
{: .no_toc .text-delta }

$$ 
\DeclareMathOperator*{\argmax}{argmax}
\hat{W} = \argmax_{W \in \mathcal{L} }{P(W|O)}
$$


---

## Evaluation


## **수식5** Word Error Rate
{: .no_toc .text-delta }
<img src="https://i.imgur.com/56x4uBL.png" width="350px" title="source: imgur.com" />


## **수식6** Sentence Error Rate
{: .no_toc .text-delta }
<img src="https://i.imgur.com/keDJO0f.png" width="350px" title="source: imgur.com" />


## **그림2** WER/SER 계산 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/jc6YI7B.png" width="500px" title="source: imgur.com" />


## **그림3** MAPSSWE 계산 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/IoJpHEi.png" width="500px" title="source: imgur.com" />


---

## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

---