---
layout: default
title: Viterbi Decoding
nav_order: 1
parent: Decoding strategy
permalink: /docs/decoding/viterbi
---

# Viterbi Decoding
{: .no_toc }

비터비 알고리즘(Viterbi Algorithm)은 가장 널리 쓰이는 디코딩 방법 가운데 하나입니다. 이 글에서는 은닉마코프모델(Hidden Markov Model)을 예시로 비터비 디코딩 기법을 설명하도록 하겠습니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---


## 개요

비터비 디코딩 과정을 개념적으로 만든 그림은 그림1과 같습니다([출처](https://www.researchgate.net/publication/273123953_Animation_of_the_Viterbi_algorithm_on_a_trellis_illustrating_the_data_association_process)). 현재 상태로 전이할 확률이 가장 큰 직전 스테이트를 모든 시점, 모든 상태에 대해 구합니다. 

## **그림1** Viterbi Decoding (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bHji1M9.gif" width="500px" title="source: imgur.com" />

모든 시점, 모든 상태에 대해 구한 결과는 그림2와 같습니다. (원래는 그물망처럼 촘촘하게 되어 있으나 경로가 끊어지지 않고 처음부터 끝까지 연결되어 있는 경로가 유효할 것이므로 그래프를 그린 사람이 이해를 돕기 위해 이들만 남겨 놓은 것 같습니다)

## **그림2** Viterbi Decoding (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/PXxizNe.png" width="500px" title="source: imgur.com" />

위 패스에서 만약 최대 확률을 내는 $k+2$번째 시점의 상태가 $θ_0$라면 *backtrace* 방식으로 구한 최적 상태열은 다음과 같습니다. 비터비 디코딩과 관련해 추가로 살펴보시려면 [이곳](https://ratsgo.github.io/data%20structure&algorithm/2017/11/14/viterbi)을 참고하시면 좋을 것 같습니다.

- $[θ_0, θ_2, θ_2, θ_1, θ_0, θ_1]$

---


## Forward Computation


비터비 알고리즘의 계산 대상인 비터비 확률(Viterbi Probability) $v$는 다음과 같이 정의됩니다. $v_t(j)$는 $t$번째 시점의 $j$번째 은닉상태의 비터비 확률을 가리킵니다. 수식1과 같습니다.


## **수식1** viterbi probability
{: .no_toc .text-delta }

$${ v }_{ t }(j)=\max _{ i } ^{n}{ \left[ { v }_{ t-1 }(i)\times { a }_{ ij }\times { b }_{ j }({ o }_{ t }) \right]  }$$



자세히 보시면 [Forward Algoritm](https://ratsgo.github.io/speechbook/docs/am/hmm#forwardbackward-algorithm)에서 구하는 전방확률 $α$와 디코딩 과정에서 구하는 비터비 확률 $v$를 계산하는 과정이 거의 유사한 것을 확인할 수 있습니다. `Forward Algorithm`은 각 상태에서의 $α$를 구하기 위해 가능한 모든 경우의 수를 고려해 그 확률들을 더해줬다면(sum), 디코딩은 그 확률들 가운데 최대값(max)에 관심이 있습니다.

이제 예를 들어보겠습니다. 학습이 완료된 은닉마코프모델이 준비돼 있습니다. 전이확률 $A$와 방출확률 $B$의 추정을 모두 마쳤다는 이야기입니다. 이제 $P(O\|W), 즉 우도(likelihood)를 계산해 보겠습니다. 예시에서는 상태(state)가 `f(j=1)`, `ay(j=2)`, `v(j=3)` 3가지뿐이고 self-loop이거나 left-to-right로 전이할 확률은 각각 0.5로 동일(uniform)합니다. 방출확률 $P(o_t\|q_j)=b_j(o_t)$는 표1의 $B$에 나열돼 있습니다. 


## **그림3** Forward Computation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3LzVWZV.png" width="400px" title="source: imgur.com" />

수식1에서 확인할 수 있듯 $t$번째 시점에 $j$번째 상태인 비터비 확률 $v_t(j)$는 전체 $n$개 직전 상태 각각에 해당하는 전방 확률들의 최댓값입니다. 이를 그림3과 표1을 비교해서 보면 이렇습니다. $t=1$, $j=3$일 때를 봅시다. $v_1(3)$은 0.8이 됩니다. 첫번째 상태에서는 이전 상태가 존재하지 않기 때문에 첫번째 상태의 비터비 확률 $v_1(1)$은 초기 상태 분포(initial state distribution)에서 뽑아 씁니다(0.8). 마찬가지로 구한 결과가 $v_1(2)=0$, $v_1(3)=0$라고 해보겠습니다.

$t=2$, $j=1$일 때를 봅시다. 최댓값 비교 대상이 되는 값들은 전체 3개 직전 상태 각각에 해당하는 전방 확률들의 최댓값입니다. `f(j=1)→f(j=1)`, `ay(j=2)→f(j=1)`, `v(j=3)→f(j=1)` 가운데 최댓값을 취하면 됩니다(the most probale path). 

그런데 음성 인식을 위한 은닉마코프모델에서는 self-loop와 left-to-right 두 가지 경우로 전이(transition)에 대한 제약을 두고 있으므로 `f(j=1)→f(j=1)`만 고려 대상이 됩니다. `f(j=1)→f(j=1)`는 $v_1(1) \times a_{11} \times b_1(o_2)$가 되므로 $0.8 \times 0.5 \times 0.8=0.32$입니다. 최댓값을 취해야 하는데 비교 대상 값이 하나뿐이므로 $v_2(1)=0.32$가 됩니다.

$t=2$, $j=2$일 때를 봅시다. 최댓값 비교 대상이 되는 값들은 전체 3개 직전 상태 각각에 해당하는 전방 확률들의 최댓값입니다. `f(j=1)→ay(j=2)`, `ay(j=2)→ay(j=2)`, `v(j=3)→ay(j=2)` 가운데 최댓값을 취하면 됩니다. 

그런데 음성 인식을 위한 은닉마코프모델에서는 self-loop와 left-to-right 두 가지 경우로 전이(transition)에 대한 제약을 두고 있으므로 `f(j=1)→ay(j=2)`, `ay(j=2)→ay(j=2)`만 고려 대상이 됩니다. 

`f(j=1)→ay(j=2)`는 $v_1(1) \times a_{12} \times b_2(o_2)$가 되므로 $0.8 \times 0.5 \times 0.1=0.04$입니다. `ay(j=2)→ay(j=2)`는 $v_1(2) \times a_{22} \times b_2(o_2)$이므로 $0 \times 0.5 \times 0.1=0$입니다. 둘 중 최댓값을 취한 것이 비터비 확률이므로 $v_2(2)=0.04$입니다.


## **표1** Forward Computation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/OgBywaq.png" width="600px" title="source: imgur.com" />


표1 상단의 각 상태에 해당하는 값들은 해당 시점/상태의 비터비 확률들입니다.


---


## Viterbi Search

그림3은 Forward Computation으로부터 베스트 경로 하나만 남겨서 비터비 경로를 탐색하는 과정을 개념적으로 나타낸 것입니다. 비터비 알고리즘을 찾은 후보 상태열의 경로들을 **Viterbi Trellis**라고 합니다. 이후 최종적으로 베스트 경로를 찾아 역추적(backtrace)를 해야 합니다. 역추적을 하는 이유는 매순간 최선의 선택이 전체 최적을 보장하지 못할 수도 있기 때문입니다.

이해를 돕기 위해 비터비 확률을 모두 구한 결과 3번째 시점에서 끝이 났다고 해봅시다. 그림3은 각 시점/상태에서 비터비 알고리즘으로 찾은 베스트 경로들(trellis)만 남긴 결과입니다. 이때 최종적인 비터비 확률은 $\max{(0.008, 0.048, 0.112)}=0.112$가 됩니다. 이 경로는 $t=3$일 때 상태가 `f`에 대응합니다. 이를 역추적(backtrace)하면 [f, f, f]가 됩니다.


## **그림3** Viterbi Trellis
{: .no_toc .text-delta }
<img src="https://i.imgur.com/J40M4Oh.png" width="400px" title="source: imgur.com" />


은닉마코프모델의 디코딩을 바이그램 등 단어 수준으로 확대하더라도 비터비 알고리즘은 동일하게 적용합니다. 그림4, 그림5는 바이그램(bigram) 모델의 Viterbi Trellis를 개념적으로 도시한 것입니다. $t=4$번째 시점의 비터비 경로가 기존 단어 내부에 있을 수도 다른 단어의 시작이 될 수도 있습니다. 이처럼 단어와 단어 사이를 넘나들며 비터비 경로가 만들어질 수 있습니다.


## **그림4** Viterbi Trellis (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/vENtXE3.png" width="500px" title="source: imgur.com" />


## **그림5** Viterbi Trellis (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/sMwdIJd.png" width="500px" title="source: imgur.com" />


비터비 경로를 끝까지 찾고 최종적으로 가장 높은 확률값을 지닌 경로 하나를 선택한 뒤 해당 경로의 상태를 역추적(backtrace)해 디코딩 결과로 리턴합니다.


## **그림5** Viterbi Backtrace
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JXM6Yua.png" width="500px" title="source: imgur.com" />


---


## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

---