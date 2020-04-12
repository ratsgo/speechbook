---
layout: default
title: Connectionist Temporal Classification
nav_order: 4
parent: Neural Acoustic Models
permalink: /docs/neuralam/ctc
---

# Connectionist Temporal Classification
{: .no_toc }

입력 음성 프레임 시퀀스와 타겟 단어/음소 시퀀스 간에 명시적인 얼라인먼트(alignment) 정보 없이도 음성 인식 모델을 학습할 수 있는 기법인 Connectionist Temporal Classification(CTC)를 살펴봅니다. 가급적 원 논문의 표기(notation)를 따랐으나 이해가 어렵다고 판단할 경우 일부 수정했음을 먼저 밝힙니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## Motivation

음성 인식 모델을 학습하려면 음성(피처) 프레임(그림1에서 첫번째 줄) 각각에 레이블 정보가 있어야 합니다. 음성 프레임 각각이 어떤 음소인지 정답이 주어져 있어야 한다는 이야기입니다. 그런데 [MFCC](https://ratsgo.github.io/speechbook/docs/fe/mfcc) 같은 음성 피처는 짧은 시간 단위(대개 25ms)로 잘게 쪼개서 만들게 되는데요. 음성 프레임 각각에 레이블(음소)을 달아줘야 하기 때문에 다량의 레이블링을 해야 하고(고비용) 인간은 이같이 짧은 구간의 음성을 분간하기 어려워 레이블링 정확도가 떨어집니다.

[Connectionist Temporal Classification(CTC)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)는 입력 음성 프레임 시퀀스와 타겟 단어/음소 시퀀스 간에 명시적인 얼라인먼트(alignment) 정보 없이도 음성 인식 모델을 학습할 수 있도록 고안됐습니다. 다시 말해 입력 프레임 각각에 레이블을 달아놓지 않아도 음성 인식 모델을 학습할 수 있다는 것입니다. 물론 음성 인식 태스크가 아니어도 소스에서 타겟으로 변환하는 모든 시퀀스 분류 과제에 CTC를 적용할 수 있습니다(예: 손글씨 실시간 인식, 이미지 프레임 시퀀스를 단어 시퀀스로 변환).


## **그림1** CTC Input
{: .no_toc .text-delta }
<img src="https://i.imgur.com/hpVlJXr.png" width="400px" title="source: imgur.com" />


CTC 기법은 시퀀스 분류를 위한 딥러닝 모델 맨 마지막에 **손실(loss) 및 그래디언트 계산 레이어**로 구현이 됩니다. CTC 레이어의 입력은 출력 확률 벡터 시퀀스(그림1 하단)입니다. 그림1에서는 Recurrent Neural Network가 CTC 입력에 쓰이는 확률 벡터 시퀀스를 만드는 데 쓰였습니다. 하지만 트랜스포머(Transformer) 등 시퀀스 출력을 가지는 어떤 아키텍처든 CTC 기법을 적용할 수 있습니다.

그림1에서 확인할 수 있듯 CTC 레이어가 입력 받는 확률 벡터의 차원수는 `레이블 수 + 1`입니다. 1이 추가된 이유는 $\varepsilon$, 즉 `blank`가 포함되어 있기 때문입니다. 예컨대 한국어 전체 음소 수가 42개라면 CTC에 들어가는 확률 벡터의 차원 수는 43차원이 됩니다. CTC 레이어에서는 그림1 하단과 같은 확률 벡터 시퀀스를 입력 받아 손실을 계산하고 그에 따른 그래디언트를 계산해 줍니다. 이후 여느 딥러닝 모델을 학습하는 것처럼 역전파(backpropagation)로 모델을 업데이트하면 됩니다.

그림2는 CTC 기법으로 학습한 모델을 예측하는 과정을 도식화한 것입니다. 먼저 입력 프레임별 예측 결과를 디코딩(decoding)합니다. 이후 반복된 음소와 `blank`를 제거하는 등의 후처리를 해서 최종 결과물을 산출합니다. 디코딩과 관련해서는 [이 글의 Decoding 챕터](https://ratsgo.github.io/speechbook/docs/neuralam/ctc#decoding)를 참고하시면 좋겠습니다.


## **그림2** CTC Prediction
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LRjrS68.png" width="550px" title="source: imgur.com" />



---


## All Possible Paths


그림2에서 눈치 채셨겠지만 CTC 기법에서는 레이블과 레이블 사이의 전이(transition)를 다음 세 가지의 경우로만 한정합니다.

1. **selp-loop** : 자기 자신을 반복합니다.
2. **left-to-right** : `non-blank` 레이블을 순방향으로 하나씩 전이합니다. 역방향은 허용하지 않습니다. `non-blank`를 두 개 이상 건너뛰는 것 역시 허용이 안 됩니다.
3. **blank 관련** : `blank`에서 `non-blank`, `non-blank`에서 `blank`로의 전이를 허용합니다.

CTC 입력 확률 벡터 시퀀스가 8개이고 정답 레이블 시퀀스 $\mathbf{l}$이 `h,e,l,l,o`일 때 위와 같은 제약을 바탕으로 가능한 모든 경로(path)를 상정해본 결과는 그림3과 같습니다. $\mathbf{l}$의 시작과 끝, 레이블 사이사이에 `blank`를 추가한 시퀀스를 $\mathbf{l}'$이라고 합니다(이제부터는 `blank`를 `-`로 표기하겠습니다). 따라서 $\mathbf{l}'$의 길이는 $2 \times \|\mathbf{l}\| + 1$이 됩니다. 그림3 가로축은 이해를 돕기 위해 $\mathbf{l}'$을 순서대로 그린 것에 해당하고요. 사실은 같은 시점에 중복된 레이블에 해당하는 $y$ 확률값은 동일하며 $\mathbf{l}'$ 말고도 나머지 레이블 역시 있다고 상상하면 좋을 것 같습니다. 그림3에서 녹색 실선에 대응하는 것은 `hello---`, 검정색 실선에 대응하는 것은 `---hello`입니다.

## **그림3** All Possible Paths
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tRNSz3q.png" width="400px" title="source: imgur.com" />

그림4는 그림3에서 해당 언어 조음 규칙상 절대로 발음될 수 없는 일부 경로들을 제거해 컴팩트하게 다시 그린 것입니다. 녹색 실선에 대응하는 것은 `hel-lo--`, 검정색 실선에 대응하는 것은 `--hel-lo`입니다. 앞으로는 그림4를 기준으로 손실 및 그래디언트를 계산해 보겠습니다.

## **그림4** All Possible Paths
{: .no_toc .text-delta }
<img src="https://i.imgur.com/W6RDQxj.png" width="400px" title="source: imgur.com" />

그림4에서 가로축은 시간(time, 인덱스는 $t$로 표기)을 나타냅니다. 세로축은 상태(state, 음소, 인덱스는 $s$로 표기)입니다. 각 세로 열은 각 시점에 해당하는 확률 벡터입니다. 동그라미는 개별 확률값을 의미합니다. 예컨대 그림4 2번째 행, 2번째 열의 동그라미는 $y_h^2$, 즉 $t=2$ 시점에 상태가 $h$일 확률을 가리킵니다. 상태를 인덱스(index)로 표현하면 2번째 행, 2번째 열의 동그라미는 $y_2^2$, 3번째 행, 2번째 열의 동그라미는 $y_3^2$로 표기할 수도 있습니다. 각각 $t=2$ 시점에 `h`가 나타날 확률, $t=2$ 시점에 `-`가 나타날 확률을 뜻합니다.

$\pi$란 그림4의 좌측 상단에서 출발해 우측 하단에 도착하는 많은 경로 가운데 하나를 가리킵니다. $\pi$는 `-`으로 시작하거나 레이블 시퀀스 $\mathbf{l}$의 첫번째 레이블로 시작할 수 있습니다. 아울러 $\pi$는 $\mathbf{l}$의 마지막 레이블로 끝나거나 `-`로 종료할 수 있습니다. 그림4로 예로 들면 검정색 실선이 $\pi$가 될 수 있고, 녹색 실선이 될 수도 있습니다. 만일 검정색 실선이 $\pi$라고 가정하면 $\pi$는 `--hel-lo`이 됩니다. $\pi_t=3$는 `h`, 즉 $s=2$가 됩니다.

CTC 기법에서는 각 상태가 조건부 독립(conditional independence)라고 가정합니다. 다시 말해 입력 음성 피처 시퀀스 $\mathbf{x}$에만 $y$ 확률값들이 변할 뿐, 이전/이후 상태가 어떻든 그 값이 변하지 않는다고 가정하는 것입니다. 이렇게 가정하면 $\mathbf{x}$가 주어졌을 때 $\pi$가 나타날 확률을 수식1처럼 $y_{ { \pi  }_t }^t$의 곱으로 나타낼 수 있습니다.


## **그림3** Conditional Indepedence
{: .no_toc .text-delta }
<img src="https://i.imgur.com/leR0ren.png" width="150px" title="source: imgur.com" />

## **수식1** Path Probability
{: .no_toc .text-delta }

$$p\left( \pi |\mathbf{x} \right) =\prod _{ t=1 }^{ T }{ { y }_{ { \pi  }_{ t } }^{ t } }$$


입력 음성 피처 시퀀스 $\mathbf{x}$가 주어졌을 때 정답 레이블 시퀀스 $\mathbf{l}$이 나타날 확률은 **상정 가능한 모든 경로($\pi$)들의 확률, 즉 $p(\pi\|\mathbf{x})$를 더해주면 됩니다**. 이를 식으로 나타낸 결과는 수식2와 같습니다. 여기에서 $\cal{B}$에 주목할 필요가 있습니다. $\cal{B}$는 `blank`와 중복된 레이블을 제거하는 함수입니다. 예컨대 $\cal{B}(\text{hheell-lo-})=\cal{B}(\text{hello})$입니다. $\cal{B}^{-1}(\mathbf{l})$은 `blank`와 중복된 레이블을 제거해서 $\mathbf{l}$이 될 수 있는 모든 가능한 경로들의 집합을 의미합니다. 그림4에서 회색 화살표 위를 지나는 모든 경로들이 $\cal{B}^{-1}(\mathbf{l})$에 해당합니다.


## **수식2** All Paths Probability
{: .no_toc .text-delta }

$$p( \mathbf{l} | \mathbf{x} )=\sum _{ \pi \in { \cal{B}  }^{ -1 } \left( \mathbf{l} \right) }^{  }{ p\left( \pi | \mathbf{x} \right)  } $$


---

## Forward Computation

수식1과 수식2에서 살펴보았듯이 $p(\mathbf{l}\|\mathbf{x})$를 구하려면 상정 가능한 모든 경로들에 대해 모든 시점, 상태에 대한 확률을 계산해야 합니다. 시퀀스 길이가 길어지거나 상태(음소) 개수가 많아지면 계산량이 폭증하는 구조인데요. CTC에서도 이를 방지하기 위해 [히든 마코프 모델(Hidden Markov Model)](https://ratsgo.github.io/speechbook/docs/am/hmm)과 같이 Forward/Backward Algorithm을 사용합니다. 그림5의 예를 들어 설명해 보겠습니다.

그림5의 파란색 칸의 전방 확률(Forward Probability)를 구해봅시다. 이는 $\alpha_3(4)$로 표기할 수 있는데요. $t=1$ 시점 $s=1$의 상태(`-`) 혹은 $t=1$ 시점 $s=2$의 상태(`h`)에서 시작해 시간 순으로 전이가 이루어져 $t=3$ 시점에 $s=4$의 상태(`e`)가 나타날 확률을 가리킵니다. $t=3$ 시점에 `e`가 나타나려면 모든 경로 가운데 4가지 경우의 수만 존재합니다. `-he`, `hhe`, `h-e`, `hee`가 바로 그것인데요. 해당 경로들은 그림5에서 파란색 화살표로 표시되어 있습니다.


## **그림5** Forward Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/dpfr6Oi.png" width="400px" title="source: imgur.com" />

`-he`가 나타날 확률은 어떻게 구할까요? 우리는 이미 조건부 독립을 가정했으므로 각 시점에 해당 상태가 나타날 확률을 단순히 곱셈을 해주면 됩니다. 마찬가지로 나머지 3개 경로, 즉 `hhe`, `h-e`, `hee`에 대해서도 같은 방식으로 구합니다. 이를 모두 더한 결과가 $\alpha_3(4)$이 될 겁니다. 수식4와 같습니다.

## **수식4** Forward Computation Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 4 \right) & = p\left( \text{"-he"} | \mathbf{x} \right) +p\left( \text{"hhe"} | \mathbf{x} \right) +p\left( \text{"h-e"} | \mathbf{x} \right) +p\left( \text{"hee"} | \mathbf{x} \right) \\
& = { y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 }\cdot { y }_{ e }^{ 3 }
\end{align*}
$$

전방확률 계산을 일반화한 식은 수식5입니다. 전방 확률은 $1:t$ 시점에 걸쳐 레이블 시퀀스가 $1:s$가 나타날 확률을 의미합니다. 수식5를 이해할 때 핵심은 계산 대상 경로인 $\pi$(수식4의 예시에서는 `-he`, `hhe`, `h-e`, `hee`)를 찾는 일이 퇼텐데요. 계산 대상이 되는 $\pi$의 조건은 다음과 같습니다. 

1. $1:t$ 시점까지의 $\pi$에서 `blank`와 중복 레이블을 제거한 결과가 $1:s$까지의 정답 레이블 시퀀스와 일치해야 합니다. 수식4의 예시를 보면 `blank`와 중복 레이블을 제거한 결과가 `he`인 모든 경로(`-he`, `hhe`, `h-e`, `hee`)를 의미합니다.
2. $\pi$는 개별 요소가 상태(음소)인 시퀀스인데요. 요소 각각이 $N$개의 범주를 가지며 시퀀스 전체 길이는 $T$입니다. CTC는 입력 음소 시퀀스(그 길이는 $T$) 각각에 대해 음소 레이블을 부여(alignment)해주는 역할을 수행한다는 사실과 연관지어 이해해보면 좋을 것 같습니다.  


## **수식5** Forward Probability
{: .no_toc .text-delta }

$${ \alpha  }_{ t }\left( s \right) =\sum _{ \pi \in { N }^{ T }: \cal{B} \left( { \pi  }_{ 1:t } \right) ={ \mathbf{l} }_{ 1:s } }^{  }{ \prod _{ t'=1 }^{ t }{ { y }_{ { \pi  }_{ t' } }^{ t' } }  }$$


수식5처럼 계산할 경우 계산량이 너무 많아집니다. 겹치는 부분은 저장해뒀다가 다시 써먹으면 효율적일 겁니다. 이같은 알고리즘을 다이내믹 프로그래밍(Dynamic Programming)이라고 합니다. 수식4에서 겹치는 부분을 찾아 다시 쓰면 수식6과 같습니다. $\alpha_2(2)$, $\alpha_2(3)$, $\alpha_2(4)$는 그림5에서 녹색으로 칠한 칸을 나타냅니다. 전방확률 계산에 다이내믹 프로그래밍을 적용한 기법을 Forward Algorithm이라고 합니다. Forward Algorithm을 적용하게 되면 $t$시점에 전방확률을 구할 때 이전 시점($t-1$)에 이미 계산해 놓은 전방확률 값들을 재사용할 수 있게 됩니다.


## **수식6** Dynamic Programming Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 4 \right) &=p\left( \text{"-he"} | \mathbf{x} \right) +p\left( \text{"hhe"} | \mathbf{x} \right) +p\left( \text{"h-e"} | \mathbf{x} \right) +p\left( \text{"hee"} | \mathbf{x} \right) \\ 
&={ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 }\cdot { y }_{ e }^{ 3 }\\ 
&=\left\{ \left( { y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 } \right) +\left( { y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 } \right) +\left( { y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 } \right)  \right\} { y }_{ e }^{ 3 }\\ 
&=\left\{ { \alpha  }_{ 2 }\left( 2 \right) +{ \alpha  }_{ 2 }\left( 3 \right) +{ \alpha  }_{ 2 }\left( 4 \right)  \right\} { y }_{ e }^{ 3 }
\end{align*}
$$

Forward Algorithm을 일반화해서 적용할 때 고려해야 하는 경우가 두 가지 있습니다. 계산 대상 현재 상태가 `blank`이거나, 현재 상태와 그 직직전 상태가 일치할 경우(`CASE1`)와 그 이외 케이스(`CASE2`)를 구분해야 한다는 점입니다. 앞서 Forward Algorithm 계산 예시로 든 수식6의 경우 `CASE2`에 해당합니다.

## **그림6** Case 1
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TdNSUy7.png" width="100px" title="source: imgur.com" />

## **그림7** Case 2
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LImeAHA.png" width="100px" title="source: imgur.com" />

이상의 논의를 종합하여 전방 확률 계산을 도식적으로 나타낸 그림은 그림8과 같습니다. 경로는 `blank` 혹은 레이블 시퀀스 $\mathbf{l}'$의 첫번째 요소로 시작합니다. $\mathbf{l}'$의 마지막 요소 혹은 `blank`로 끝을 맺습니다. 경로 중간에 있는 상태들의 전방 확률 계산은 `CASE1` 혹은 `CASE2`에만 해당하므로 해당 케이스에 맞춰 계산해 줍니다. 이를 식으로 표현하면 수식7과 같습니다.

## **그림8** Forward Flow
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1vbZ0gt.png" width="500px" title="source: imgur.com" />


## **수식7** Forward Algorithm
{: .no_toc .text-delta }

$$
\require{ams}
\begin{equation*}
    { \alpha  }_{ t }\left( s \right) =
    \begin{cases}
      \left\{ { \alpha  }_{ t-1 }\left( s \right) +{ \alpha  }_{ t-1 }\left( s-1 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{if}\ { \mathbf{l}' }_{ s }=b \text{ or } { \mathbf{l}' }_{ s - 2 }={ \mathbf{l}' }_{ s }\\
      \left\{ { \alpha  }_{ t-1 }\left( s \right) +{ \alpha  }_{ t-1 }\left( s-1 \right) +{ \alpha  }_{ t-1 }\left( s-2 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{otherwise}
    \end{cases}
\end{equation*}
$$

---


## Backward Computation

이번에는 그림9의 오렌지색 칸의 후방 확률(Backward Probability)을 계산해 보겠습니다. 이는 $\beta_6(9)$로 표기할 수 있는데요. $t=8$ 시점 $s=11$의 상태(`-`) 혹은 $t=8$ 시점 $s=10$의 상태(`o`)에서 시작해 시간의 역순으로 전이가 이루어져 $t=6$ 시점에 $s=9$의 상태(`-`)가 나타날 확률을 가리킵니다. 다시 말해 우측 하단 끝 지점에서 시작해 $t=6$ 시점에 `-`가 나타나려면 모든 경로 가운데 2가지 경우의 수만 존재한다는 이야기입니다. 이해를 돕기 위해 시간 순으로 정렬해 해당 경로들을 나타내면 `--o`, `-oo`, `-o-`가 바로 그것입니다. 이 경로들은 그림9에서 붉은색 화살표로 표시되어 있습니다. 그림9의 $\beta_6(9)$를 계산한 결과는 수식8과 같습니다.


## **그림9** Backward Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nq67yUG.png" width="400px" title="source: imgur.com" />

## **수식8** Backward Computation Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \beta  }_{ 6 }\left( 9 \right) =&p\left( { \text{"--o"} | \mathbf{x} } \right) +p\left( { \text{"-oo"} | \mathbf{x} } \right) +p\left( { \text{"-o-"} | \mathbf{x} } \right) \\ 
=&{ y }_{ - }^{ 6 }\cdot { y }_{ - }^{ 7 }\cdot { y }_{ o }^{ 8 }+{ y }_{ - }^{ 6 }\cdot { y }_{ o }^{ 7 }\cdot { y }_{ o }^{ 8 }+{ y }_{ - }^{ 6 }\cdot { y }_{ o }^{ 7 }\cdot { y }_{ - }^{ 8 }
\end{align*}
$$

후방확률 계산을 일반화한 식은 수식9입니다. 입력 음성 시퀀스의 길이가 $T$, 타겟 레이블 시퀀스가 $\mathbf{l}$이라고 할 때 $t$ 시점의 후방 확률은 $t:T$ 시점에 걸쳐 레이블 시퀀스가 $s:\|\mathbf{l}\|$일 확률을 의미합니다. 계산 대상이 되는 $\pi$의 조건은 다음과 같습니다.

1. $t:T$ 시점까지의 $\pi$에서 `blank`와 중복 레이블을 제거한 결과가 $s:\|\mathbf{l}\|$까지의 정답 레이블 시퀀스와 일치해야 합니다. 수식8의 예시를 보면 `blank`와 중복 레이블을 제거한 결과가 `o`인 모든 경로(`--o`, `-oo`, `-o-`)를 의미합니다.
2. $\pi$는 개별 요소가 상태(음소)인 시퀀스인데요. 요소 각각이 $N$개의 범주를 가지며 시퀀스 전체 길이는 $T$입니다. 

## **수식9** Backward Probability
{: .no_toc .text-delta }

$${ \beta  }_{ t }\left( s \right) =\sum _{ \pi \in { N }^{ T }: \cal{B} \left( { \pi  }_{ t:T } \right) ={ \mathbf{l} }_{ s:| \mathbf{l} | } }^{  }{ \prod _{ t'=t }^{ T }{ { y }_{ { \pi  }_{ t' } }^{ t' } }  } $$


전방확률 계산 때와 마찬가지로 수식8처럼 계산하는 것 대신 다이내믹 프로그래밍을 적용하면 계산량이 줄어듭니다. 수식8에서 겹치는 부분을 찾아 다시 쓰면 수식10과 같습니다. 후방 확률은 전방 확률의 역이므로, Backward Algorithm을 적용하게 되면 $t$시점 후방 확률을 구할 때 다음 시점($t+1$)에 이미 계산해 놓은 후방확률 값들을 재사용할 수 있게 됩니다. 이같이 후방 확률 계산에 다이내믹 프로그래밍을 적용한 기법을 Backward Algorithm이라고 합니다. $\beta_7(9)$, $\beta_7(10)$은 그림9에서 노란색으로 칠한 칸을 나타냅니다.

## **수식10** Dynamic Programming Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \beta  }_{ 6 }\left( 9 \right) =&p\left( { \text{"--o"} | \mathbf{x} } \right) +p\left( { \text{"-oo"} | \mathbf{x} } \right) +p\left( { \text{"-o-"} | \mathbf{x} } \right) \\ 
=&{ y }_{ - }^{ 6 }\cdot { y }_{ - }^{ 7 }\cdot { y }_{ o }^{ 8 }+{ y }_{ - }^{ 6 }\cdot { y }_{ o }^{ 7 }\cdot { y }_{ o }^{ 8 }+{ y }_{ - }^{ 6 }\cdot { y }_{ o }^{ 7 }\cdot { y }_{ - }^{ 8 } \\
=&\left\{ \left( { y }_{ - }^{ 7 }\cdot { y }_{ o }^{ 8 } \right) +\left( { y }_{ o }^{ 7 }\cdot { y }_{ o }^{ 8 }+{ y }_{ o }^{ 7 }\cdot { y }_{ - }^{ 8 } \right)  \right\} { y }_{ - }^{ 6 } \\
=&\left\{ { \beta  }_{ 7 }\left( 9 \right) +{ \beta  }_{ 7 }\left( 10 \right)  \right\} { y }_{ - }^{ 6 }
\end{align*}
$$


Backward Algorithm 역시 Forward Algorithm 때처럼 `CASE1`과 `CASE2`로 나누어 계산합니다. 전방 확률 계산과 그 방향이 달라졌을 뿐 본질적인 계산 방식은 동일합니다. 그림9의 예제는 `CASE1`에 해당합니다. Backward Algorithm을 일반화한 식은 수식11과 같습니다.

## **수식11** Backward Algorithm
{: .no_toc .text-delta }

$$
\require{ams}
\begin{equation*}
    \beta _{ t }\left( s \right) =
    \begin{cases}
      \left\{ { \beta  }_{ t+1 }\left( s \right) +\beta _{ t+1 }\left( s+1 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{if}\ { \mathbf{l}' }_{ s }=b \text{ or } { \mathbf{l}' }_{ s + 2 }={ \mathbf{l}' }_{ s }\\
      \left\{ { \beta  }_{ t+1 }\left( s \right) +{ \beta  }_{ t+1 }\left( s+1 \right) +\beta _{ t+1 }\left( s+2 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{otherwise}
    \end{cases}
\end{equation*}
$$


---


## Complete Path Calculation


이번엔 특정 시점, 특정 상태를 지나는 경로에 대한 등장 확률을 구해보겠습니다. 이는 앞서 구한 전방확률과 후방확률로 간단히 계산할 수 있습니다. 그림10에서 파란색으로 칠한 칸을 봅시다. $t=3$ 시점에 상태가 `h`($s=2$)인 모든 경로에 대한 확률을 구해보자는 것입니다. 해당 경로는 `--hel-lo`, `-hhel-lo`, `hhhel-lo` 3가지가 존재합니다.

## **그림10** Complete Path Calculation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ctsh5p9.png" width="400px" title="source: imgur.com" />

`--hel-lo`, `-hhel-lo`, `hhhel-lo` 3가지 경로에 대한 확률은 물론 모든 시점, 상태에 대해 일일이 곱셈을 하여서 구할 수 있습니다. 하지만 지금까지 설명했던 전방확률과 후방확률을 활용하면 좀 더 효율적으로 구할 수 있습니다. 파란색 칸에 대한 전방확률과 후방확률은 각각 수식12, 수식13과 같습니다.

## **수식12** Forward Probability Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 2 \right) &=p\left( \text{"--h"} | \mathbf{x} \right) +p\left( \text{"-hh"} | \mathbf{x} \right) +p\left( \text{"hhh"} | \mathbf{x} \right) \\ 
&={ y }_{ - }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ h }^{ 3 }+{ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }
\end{align*}
$$


## **수식13** Backward Probability Example
{: .no_toc .text-delta }

$$
\begin{align*}
\beta _{ 3 }\left( 2 \right) &=p\left( \text{"hel-lo"} | \mathbf{x} \right) \\ 
&={ y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }
\end{align*}
$$

수식14는 수식12의 전방확률과 수식13의 후방확률을 곱한 것입니다. 식을 자세히 보시면 전방확률과 후방확률의 곱은 우리가 구하고자 하는 `--hel-lo`, `-hhel-lo`, `hhhel-lo` 3가지 경로에 대한 확률을 더한 값에 $y_h^3$이 곱해진 형태임을 확인할 수 있습니다. 수식11의 양변을 $y_h^3$으로 나눠주면 우리가 구하고자 하는 값을 구할 수 있습니다. 이는 수식15에 나와 있습니다.

## **수식14** Forward/Backward Probability Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 2 \right) \cdot \beta _{ 3 }\left( 2 \right) =&{ y }_{ - }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }\\
&+{ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }\\
&+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }\\
=&\left\{ p\left( \text{"--hel-lo"} | \mathbf{x} \right) +p\left( \text{"-hhel-lo"} | \mathbf{x} \right) +p\left( \text{"hhhel-lo"} | \mathbf{x} \right)  \right\} { y }_{ h }^{ 3 }\\ 
\end{align*}
$$


## **수식15** Complete Path Probability Example
{: .no_toc .text-delta }

$$p\left( \text{"--hel-lo"} | \mathbf{x} \right) +p\left( \text{"-hhel-lo"} | \mathbf{x} \right) +p\left( \text{"hhhel-lo"} | \mathbf{x} \right) =\frac { { \alpha  }_{ 3 }\left( 2 \right) \cdot \beta _{ 3 }\left( 2 \right)  }{ { y }_{ h }^{ 3 } }$$


---

## Likelihood Computation

우리의 관심은 입력 음성 피처 시퀀스 $\mathbf{x}$가 주어졌을 때 레이블 시퀀스 $\mathbf{l}$이 나타날 확률, 즉 우도(likelihood)를 최대화하는 모델 파라메터(parameter)를 찾는 데 있습니다(Maximum Likelihood Estimation). 입력 음성 피처의 각 프레임에 대해서 모두 레이블이 있다면 지금까지 어렵사리 전방확률이니, 후방확률이니 하는 값들을 구할 필요 없이 네트워크 마지막 레이어의 출력인 음소 확률 벡터에서 정답 레이블에 관련된 확률들을 높여주면 그만일 겁니다(=MLE=Cross Entropy 최소화). 하지만 우리가 가진 데이터는 입력 음성 피처 시퀀스 $\mathbf{x}$와 레이블 시퀀스 $\mathbf{l}$뿐입니다. 둘의 길이가 다르기 때문에 크로스 엔트로피를 적용할 수 없습니다. 그래서 이 먼 길을 돌아오게 된 겁니다.

우도를 최대화하려면 먼저 우도 값을 구해야 합니다. CTC 문제에서 우도는 입력 데이터 시퀀스 $\mathbf{x}$가 주어졌을 때 `blank` 없는 원래 레이블 시퀀스 $\mathbf{l}$이 나타날 확률을 가리킵니다. 지금까지 제시했던 예시 기준으로 하면 `h,e,l,l,o`가 나타날 확률입니다. 이와 별개로 $\mathbf{l}'$은 `blank`가 포함된 레이블 시퀀스인데요. 우도, 즉 `h,e,l,l,o`가 나타날 확률을 구하려면 `hel-lo--`에서부터 `--hel-lo`에 이르는 $\mathbf{l}'$에 관한 모든 가능한 경로들(그림11에서 회색 화살표의 모든 경우의 수)의 확률을 모두 고려(sum)해야 합니다.

그런데 우도 계산 역시 전방확률과 후방확률이 있으면 편히 구할 수 있습니다. 우선 수식15 계산 결과는 `blank`가 포함된 레이블 시퀀스 $\mathbf{l}'$ 가운데 $t=3$ 시점에 상태 $h$를 지나는 경로들에 대한 확률입니다. 이 경로들은 전체 경로 가운데 일부에 해당합니다(그림10 참조). 이미 말씀드렸듯 $\mathbf{l}'$ 관련 전체 경로의 확률 합이 우도인데요. $t=3$을 기준으로 우도를 구한다고 하면 그림11의 파란색으로 칠한 다섯 칸에 해당하는 Complete Path 확률을 모두 더한 값이 됩니다. 이는 수식16과 같습니다(이해를 돕기 위해 상태 인덱스 대신 `h`, `-`, `e` 따위로 직접 표현하였습니다).

## **그림11** Likelihood Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/IYSiNlx.png" width="400px" title="source: imgur.com" />

## **수식16** Likelihood Computation Example
{: .no_toc .text-delta }

$$
\begin{align*}
p\left( \text{"hello"} | \mathbf{x} \right) =&\frac { { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right)  }{ { y }_{ h }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } }\\ 
&+\frac { { \alpha  }_{ 3 }\left( e \right) \cdot \beta _{ 3 }\left( e \right)  }{ { y }_{ e }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( l \right) \cdot \beta _{ 3 }\left( l \right)  }{ { y }_{ l }^{ 3 } }
\end{align*}
$$

우도 계산을 일반화한 식은 수식17과 같습니다. 시점(time)을 고정해 놓고 상태(state)에 대해 모두 합을 취하면 우도가 됩니다. 식에서 알 수 있듯이 어떤 시점에서 계산하든 우도 값은 동일합니다. 각 시점, 상태일 확률이 조건부 독립(conditional independence)임을 가정하기 때문입니다.

## **수식17** Likelihood Computation
{: .no_toc .text-delta }

$$p\left( \mathbf{l} | \mathbf{x} \right) =\sum _{ s=1 }^{ | \mathbf{l}' | }{ \frac { { \alpha  }_{ t }\left( s \right) \cdot \beta _{ t }\left( s \right)  }{ { y }_{ { \mathbf{l}' }_{ s } }^{ t } }  } $$



---


## Gradient Computation

우리는 우도, 즉 $p(\mathbf{l}\|\mathbf{x})$를 최대화하는 모델 파라메터를 찾고자 합니다. 이를 위해서는 우도에 대한 그래디언트(gradient)를 구해야 합니다. 이 그래디언트를 모델 전체 학습 파라메터에 역전파(backpropagation)하는 것이 CTC 기법을 적용한 모델의 학습(train)이 되겠습니다. 우도 계산은 보통 로그 우도(log-likelihood)로 수행하는데요. $t$번째 시점 $k$번째 상태(음소)에 대한 로그 우도의 그래디언트는 수식18과 같습니다. $\ln(x)$를 $x$로 미분하면 $1/x$이고 우도에 로그를 취한 것을 합성 함수라고 이해하면 체인룰(chain)에 의해 수식18 우변처럼 정리할 수 있습니다.


## **수식18** Gradient Computation (1)
{: .no_toc .text-delta }

$$\frac { \partial \ln { \left( p\left( \mathbf{l} | \mathbf{x} \right)  \right)  }  }{ \partial { y }_{ k }^{ t } } =\frac { 1 }{ p\left( \mathbf{l} | \mathbf{x} \right)  } \frac { \partial p\left( \mathbf{l}| \mathbf{x} \right)  }{ \partial { y }_{ k }^{ t } } $$


수식18의 우변 마지막항 일부를 다시 정리한 결과는 수식19와 같습니다. 우도는 수식17과 같고 상수 $a$에 대해 $a/x$를 $x$로 미분한 결과는 $-a/x^2$라는 점을 고려하면 수식19 우변을 이해할 수 있습니다. 수식19에서 $\text{lab}(\mathbf{l}, k)$는 $k$라는 음소 레이블이 $\mathbf{l}'$에서 나타난 위치들을 가리킵니다. 


## **수식19** Gradient Computation (2)
{: .no_toc .text-delta }

$$\frac { \partial p\left( \mathbf{l} | \mathbf{x} \right)  }{ \partial { y }_{ k }^{ t } } = - \frac { 1 }{ { { y }_{ k }^{ t } }^{ 2 } } \sum _{ s\in \text{lab} \left( \mathbf{l},k \right)  }^{  }{ { \alpha  }_{ t }\left( s \right) \cdot \beta _{ t }\left( s \right)  }$$


식만 봐서는 감이 잘 안올테니 바로 예시를 봅시다. 그림12에서 구하고자 하는 것은 $\partial p(\ln (\mathbf{l} \| \mathbf{x})) / \partial y_h^3$입니다. 바로 파란색이 칠해져 있는 칸에 해당하는 그래디언트입니다. 이 값을 구하려면 우선 우도 $p(\mathbf{l} \| \mathbf{x})$부터 구해야 합니다. 수식20과 같습니다. 

## **그림12** Gradient Computation Example (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Zhscxzw.png" width="400px" title="source: imgur.com" />

## **수식20** Gradient Computation Example (1)
{: .no_toc .text-delta }

$$
\begin{align*}
p\left( \text{"hello"} | \mathbf{x} \right) =&\frac { { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right)  }{ { y }_{ h }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } }\\
&+\frac { { \alpha  }_{ 3 }\left( e \right) \cdot \beta _{ 3 }\left( e \right)  }{ { y }_{ e }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( l \right) \cdot \beta _{ 3 }\left( l \right)  }{ { y }_{ l }^{ 3 } }
\end{align*}
$$

수식18의 로그 우도에 대한 그래디언트를 구하기 위해서는 수식19의 우도에 대한 그래디언트를 구해야 합니다. 우도에 대한 그래디언트는 수식20을 $y_h^3$으로 편미분한 결과입니다. 수식20 우변에서 $y_h^3$과 관련된 항은 첫번째 항뿐이기 때문에 나머지는 상수 취급해서 소거할 수 있습니다. 다시 말해 `h`라는 음소 레이블이 $\mathbf{l}'$에서 나타난 위치(2)에 해당하는 항만 남긴다는 이야기입니다. 수식21에서 구한 값은 수식18에 대입해 최종적으로 로그 우도에 대한 그래디언트를 계산합니다.

## **수식21** Gradient Computation Example (1)
{: .no_toc .text-delta }

$$\frac { \partial p\left( \text{"hello"} | \mathbf{x} \right)  }{ \partial { y }_{ h }^{ 3 } } = - \frac { 1 }{ { { y }_{ h }^{ 3 } }^{ 2 } } { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right) $$


그림13에서 구하고자 하는 것은 $\partial p(\ln (\mathbf{l} \| \mathbf{x})) / \partial y_l^5$입니다. 바로 파란색이 칠해져 있는 두 개 칸에 해당하는 그래디언트입니다. 이 값을 구하려면 우선 우도 $p(\mathbf{l} \| \mathbf{x})$부터 구해야 합니다. 수식22와 같습니다. 

## **그림13** Gradient Computation Example (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HgVRXpa.png" width="400px" title="source: imgur.com" />

## **수식22** Gradient Computation Example (2)
{: .no_toc .text-delta }

$$p\left( \text{"hello"} | \mathbf{x} \right) =\frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } } +\frac { { \alpha  }_{ 5 }\left( - \right) \cdot \beta _{ 5 }\left( - \right)  }{ { y }_{ - }^{ 5 } } +\frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } } $$

우도에 대한 그래디언트는 수식22를 $y_l^5$로 편미분한 결과입니다. 수식22 우변에서 $y_l^5$과 관련된 항은 첫번째 항과 두번째 항이기 때문에 나머지는 상수 취급해서 소거할 수 있습니다. 다시 말해 `l`이라는 음소 레이블이 $\mathbf{l}'$에서 나타난 위치(6, 8)에 해당하는 항만 남긴다는 이야기입니다. 수식23에서 구한 값은 수식18에 대입해 최종적으로 로그 우도에 대한 그래디언트를 계산합니다.

## **수식23** Gradient Computation Example (2)
{: .no_toc .text-delta }

$$\frac { \partial p\left( \text{"hello"} | \mathbf{x} \right)  }{ \partial { y }_{ l }^{ 5 } } = - \frac { 1 }{ { { y }_{ l }^{ 5 } }^{ 2 } } \left\{ { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right) + { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  \right\} $$


---


## Rescaling

지금까지 CTC에 대한 핵심적인 설명을 마쳤습니다. 단 CTC 저자에 따르면 앞의 챕터 방식대로 전방확률과 후방확률을 계산하게 되면 그 값이 너무 작아져 언더플로(underflow) 문제가 발생한다고 합니다. 이에 수식24와 수식25와 같은 방식으로 전방확률과 후방확률 값을 리스케일링(rescaling)해 줍니다. 그래디언트 역시 리스케일한 전방/후방확률을 기준으로 계산하게 됩니다. 자세한 내용은 [원 논문](https://www.cs.toronto.edu/~graves/icml_2006.pdf)을 참고하시면 좋을 것 같습니다.


## **수식24** Rescaled Forward Probability
{: .no_toc .text-delta }

$$
\begin{align*}
{ C }_{ t }=&\sum _{ s }^{  }{ { \alpha  }_{ t }\left( s \right)  } \\ 
{ \hat { \alpha  }  }_{ t }\left( s \right) =& \frac { { \alpha  }_{ t }\left( s \right)  }{ { C }_{ t } } 
\end{align*}
$$


## **수식25** Rescaled Backward Probability
{: .no_toc .text-delta }

$$
\begin{align*}
{ D }_{ t }=&\sum _{ s }^{  }{ { \beta  }_{ t }\left( s \right)  } \\ 
{ \hat { \beta  }  }_{ t }\left( s \right) =& \frac { { \beta  }_{ t }\left( s \right)  }{ { D }_{ t } } 
\end{align*}
$$


---


## Decoding


CTC로 학습한 모델의 출력은 확률 벡터 시퀀스이므로 그 결과를 적절히 디코딩(decoding)해 주어야 합니다. 가장 간단한 방법은 **Best Path Decoding**이라는 기법입니다. 시간 축을 따라 가장 확률값이 높은 레이블을 디코딩 결과로 출력하는 방법입니다. 

그림14는 CTC로 학습한 모델이 출력한 확률 벡터 시퀀스를 시간 축을 따라 쭉 이어붙인 것입니다. 가로축은 시간(time), 세로축은 음소를 나타냅니다. 가로축을 보니 입력 음성 피처 시퀀스의 길이가 30이네요. Best Path Decoding 방식으로 그림14를 디코딩하면 `---B-OO--XXX-__--BBUUNN-NI--->`이 됩니다.

## **그림14** Best Path Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TWnm2Vi.png" title="source: imgur.com" />

**Prefix Decoding**은 매 스텝마다 가장 확률값이 높은 prefix를 디코딩 결과로 출력하는 것입니다. 그림 15와 같습니다. 그림15의 첫번째 스텝에서 가장 확률값이 높은 prefix는 `X`입니다. 그런데 `X`의 확률값(0.7)은 두번째 스텝에서 각 상태가 지니는 확률의 합($p(X)=0.1$, $p(Y)=0.5$, $p(e)=0.1$)과 같습니다. `X`의 확률값(0.7)은 $t=1$ 시점에 상태가 `X`일 확률을 가리키는데요. 이는 앞서 설명한 Forward/Backward Algorithm으로 계산 가능합니다. 어쨌든 그림15와 같은 예시일 때 Prefix Decoding 결과는 `XYe`가 됩니다.

## **그림15** Prefix Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bjbfVAV.png" width="200px" title="source: imgur.com" />

**Beam Search**는 매 스텝마다 가장 확률이 높은 후보 시퀀스를 beam 크기만큼 남겨서 디코딩하는 방법입니다. 그림16은 beam 크기가 3일 때 Beam Search의 일반적인 디코딩 과정을 도식적으로 나타내고 있습니다. 시간이 아무리 흘려도 살아남는 후보 시퀀스 갯수는 beam 크기가 됩니다. 그림17과 그림18은 CTC로 학습한 모델의 Beam Search 과정을 나타냅니다. 일반적인 Beam Search와 거의 유사하나 디코딩 과정에서 여러 번 등장한 음소를 합치거나 `blank` 등을 적절히 처리해 줍니다.

## **그림16** Beam Search (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HTIszHk.png" width="500px" title="source: imgur.com" />


## **그림17** Beam Search (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bB1pi0h.png" width="600px" title="source: imgur.com" />


## **그림18** Beam Search (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TGba0fl.png" width="600px" title="source: imgur.com" />


---


## Properties


그림19는 프레임 단위 레이블로 학습한 음성 인식 모델과 CTC로 학습한 모델 간 차이를 나타내는 그림으로 CTC 원저자가 작성한 것입니다. 저자에 따르면 `dh`라는 음소의 경우 프레임 단위 모델은 정답을 잘 맞췄지만 다른 음소와의 구분이 잘 되지 않았습니다. 그에 반해 CTC 모델은 구분이 잘 되어 있는 걸 확인할 수 있습니다. 아울러 CTC 모델은 프레임 단위 모델 대비 음소 확률 분포가 뾰족(spiky)하고 희소(sparse)합니다.


## **그림19** Activations
{: .no_toc .text-delta }
<img src="https://i.imgur.com/jQwZHyh.png" title="source: imgur.com" />


하지만 개인적으로는 그림19가 역설적이게도 CTC 모델의 단점을 드러낸 모델이 아닌가 싶습니다. 입력 음성 피처 시퀀스별로 레이블을 부여(Forced Alignment)하는 태스크에는 CTC가 제대로 작동하지 않을 염려가 있다고 생각되기 때문입니다. 실제로 CTC 모델이 `s` 음소를 인식 결과로 리턴할 수 있는 구간은 프레임 단위 모델 대비 짧은 편입니다. 


---


## References

- [모두의 연구소, 음성인식 풀잎스쿨]()
- [Connectionist Temporal Classification, Labelling Unsegmented Sequence Data with RNN](https://www.youtube.com/watch?v=UMxvZ9qHwJs&t=2379s)
- [Sequence Modeling
With CTC](https://distill.pub/2017/ctc/)

---