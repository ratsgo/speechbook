---
layout: default
title: Gaussian Mixture Model
nav_order: 3
parent: Acoustic Models
permalink: /docs/am/gmm
---

# Gaussian Mixture Model
{: .no_toc }

기존 음성 인식 모델의 근간이었던 Gaussian Mixture Model에 대해 살펴봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## Normal Distribution

정규분포는 가우스(Gauss, 1777-1855)에 의해 제시된 분포로서 일명 가우스분포(Gauss Distribution)라고 불립니다. 물리학 실험 등에서 오차에 대한 확률분포를 연구하는 과정에서 발견되었다고 합니다. 가우스 이후 이 분포는 여러 학문 분야에서 이용되었습니다. 초기의 통계학자들은 모든 자료의 히스토그램이 정규분포의 형태와 유사하지 않으면 비정상적인 자료라고까지 생각하였다고 합니다. 이러한 이유로 이 분포에 ‘정규(normal)’라는 이름이 붙게 된 것입니다.

정규분포는 특성값이 연속적인 무한모집단 분포의 일종으로서 평균이 $\mu$이고 표준편차가 $\sigma$인 경우 정규분포의 확률밀도함수(Probability Density Function)는 수식1과 같습니다. 수식1은 입력 변수 $x$가 1차원의 스칼라값인 단변량 정규분포(Univariate Normal Disribution)를 가리킵니다.


## **수식1** Univariate Normal distribution
{: .no_toc .text-delta }

$$N(x|\mu ,{\sigma}^2 )=\frac { 1 }{ \sqrt { 2\pi  } \sigma  } \exp\left( -\frac { { (x-\mu ) }^{ 2 } }{ 2{ \sigma  }^{ 2 } }  \right)$$


## **그림1** 서로 다른 Univariate Normal distribution
{: .no_toc .text-delta }
<img src="https://i.imgur.com/28YetdP.png" width="350px" title="source: imgur.com" />


입력 변수가 $D$차원인 벡터인 경우를 다변량 정규분포(Multivariate Normal Distributution)라고 합니다. 수식2와 같습니다. 여기에서 $\mathbf{\mu}$는 $D$차원의 평균 벡터, $\mathbf{\Sigma}$는 $D \times D$ 크기를 가지는 공분산(covariance) 행렬을 의미합니다. $\|\mathbf{\Sigma}\|$는 $\mathbf{\Sigma}$의 행렬식(determinant)입니다.


## **수식1** Multivariate Normal distribution
{: .no_toc .text-delta }

$$N({\bf x}|{\pmb \mu}, {\bf \Sigma}) = \dfrac{1}{(2\pi)^{D/2}|{\bf \Sigma}|^{1/2}}\exp\left\{-\frac{1}{2}({\bf x}-{\pmb \mu})^T{\bf \Sigma}^{-1}({\bf x}-{\pmb \mu})\right\}$$


## **그림2** 서로 다른 Multivariate Normal distribution
{: .no_toc .text-delta }
<img src="https://i.imgur.com/V7DNP2Z.png" width="500px" title="source: imgur.com" />


## **그림2** 서로 다른 Multivariate Normal distribution의 공분산
{: .no_toc .text-delta }
<img src="https://i.imgur.com/yvfGIqv.png" width="500px" title="source: imgur.com" />


---


## Gaussian Mixture Model


## **수식1** Gaussian Mixture Model
{: .no_toc .text-delta }

$$
f\left( \mathbf{x} | \mathbf{\lambda}  \right) = \sum_{ i=1  }^{ M }{ c_i N({\bf x}_i|{\pmb \mu_i}, {\bf \Sigma}_i) } \\ \mathbf{\lambda} =\left\{ { c }_{ i }, \mathbf{ \mu  }_{ i },\mathbf{ \Sigma  }_{ i } \right\} \text{,   }i=1,...M
$$


## **그림3** Gaussian Mixture Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/u7tDOSE.png" width="300px" title="source: imgur.com" />



---


## Expectation-Maximization


## **그림4** Expectation-Maximization Algorithm
{: .no_toc .text-delta }
<img src="https://i.imgur.com/rXG3zOn.png" title="source: imgur.com" />

---


## Speech Recognition에 적용


MFCC 피처가 단변량 가우시안을 따르지 않기 때문에 $M$개 가우시안 함수를 합친 **가우시안 믹스처 모델**로 모델링합니다. 계산량을 줄이고 수렴을 가속화하기 위해 공분산이 diagonal, 즉 입력 벡터의 변수들끼리 서로 독립(independent)하다고 가정합니다.

---