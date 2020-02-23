---
layout: default
title: Hidden Markov Model
nav_order: 1
parent: Acoustic Models
permalink: /docs/am/hmm
---

# Hidden Markov Model
{: .no_toc }

기존 음성 인식 모델의 근간이었던 은닉마코프모델(Hidden Markov Model)에 대해 살펴봅니다. 이는 마코프 체인을 전제로 한 모델로 음소(또는 단어) 시퀀스를 모델링하는 데 자주 쓰였습니다. 우도(likelihood) 계산, 디코딩(decoding) 등을 중심으로 설명할 예정입니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## Markov Chain

은닉마코프모델(Hidden Markov Model)은 마코프 체인(Markov chain)을 전제로 한 모델입니다. 마코프 체인이란 마코프 성질(Markov Property)을 지닌 이산확률과정(discrete-time stochastic process)을 가리킵니다. 마코프 체인은 러시아 수학자 마코프가 1913년경에 러시아어 문헌에 나오는 글자들의 순서에 관한 모델을 구축하기 위해 제안된 개념입니다.

한 상태(state) $q_i$가 나타날 확률은 단지 그 이전 상태 $q_{i-1}$에만 의존한다는 것이 마코프 성질의 핵심입니다. 즉 한 상태에서 다른 상태로의 전이(transition)는 그동안 상태 전이에 대한 긴 이력(history)을 필요로 하지 않고 바로 직전 상태에서의 전이로 추정할 수 있다는 이야기입니다. 마코프 성질은 수식1과 같이 도식화됩니다.


## **수식1** Markov Property
{: .no_toc .text-delta }
$$P({ q }_{ i }|{ q }_{ 1 },...,{ q }_{ i-1 })=P({ q }_{ i }|{ q }_{ i-1 })$$


마코프 체인은 시간 변화에 따른 상태들의 분포를 추적하는 데 관심이 있습니다. 그런데 시점에 따라 전이할 확률이 얼마든지 달라질 수 있습니다. 보통의 마코프 체인에서는 모델링을 간소화하기 위해 전이 확률 값이 전이 시점에 관계 없이 상태에만 의존한다고 가정합니다. 이른바 시간안정성(time-homogeneous, time stationary) 가정입니다. 수식2와 같습니다.


## **수식2** Time Stationary Property
{: .no_toc .text-delta }
$$P( q_{i+1}=x | q_i = y ) = P( q_2 = x | q_1 = y) = P_{xy}$$


---


## Hidden Markov Models

은닉마코프모델(Hidden Markov Model)은 각 상태가 마코프체인을 따르되 은닉(hidden)되어 있다고 가정합니다. 예컨대 당신이 100년 전 기후를 연구하는 학자인데, 주어진 정보는 당시 아이스크림 소비 기록뿐이라고 칩시다. 이 정보만으로 당시 날씨가 더웠는지, 추웠는지, 따뜻했는지를 알고 싶은 겁니다. 우리는 아이스크림 소비 기록의 연쇄를 관찰할 수 있지만, 해당 날짜의 날씨가 무엇인지는 직접적으로 관측하기 어렵습니다. 

은닉마코프모델은 이처럼 관측치(observation) 뒤에 은닉되어 있는 상태(state)를 추정하고자 합니다. 바꿔 말하면 우리가 보유하고 있는 데이터(observation)는 실제 은닉 상태들(true hidden state)이 노이즈가 낀 형태로 실현된 것이라고 보는 것입니다. 이렇게 모델링하는 기법을 `noisy channel`이라고 합니다. 날씨를 예시로 은닉마코프모델을 도식화한 그림은 그림1과 같습니다.


## **그림1** Hidden Markov Model
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nZi5O1B.png" width="600px" title="source: imgur.com" />


은닉마코프모델은 크게 두 가지 요소로 구성돼 있습니다. 하나는 전이 확률(transition probability) $A$, 다른 하나는 방출 확률(emission probablity) $B$입니다. 그림1에서 각 노드는 은닉 상태, 엣지는 전이를 가리킵니다. 엣지 위에 작게 써 있는 숫자는 각 상태에서 상태로 전이할 확률을 나타냅니다. 각 노드 기준으로 전이 확률의 합은 1입니다.

$B_1$은 날씨가 더울 때 아이스크림을 1개 소비할 확률이 0.2, 2개 내지 3개 먹을 확률은 각각 0.4라는 걸 나타냅니다. $B_1$은 날씨가 더울 때 조건부확률이므로 `HOT`이라는 은닉 상태와 연관이 있습니다. 은닉된 상태로부터 관측치가 튀어나올 확률이라는 의미에서 방출(emission) 확률이라는 이름이 붙은 것 같습니다.

그림1에서 $\pi$는 초기 상태 분포(initial probability distribution)을 가리킵니다. 예컨대 $\pi_1$(0.8)은 첫번째 은닉 상태가 `HOT`, $\pi_2$(0.2)는 첫번째 상태가 `COLD`일 확률을 의미합니다. 한편 은닉마코프모델의 학습(train)은 전이 확률 $A$와 방출 확률 $B$를 데이터(observation)으로부터 추정(estimate)하는 과정입니다. 그런데 이 글에서는 은닉마코프모델 전체 프레임워크를 이해하는 데 주안점이 있으므로 $A$와 $B$는 이미 추정이 완료되었다고 가정하고 설명하겠습니다.


---


## Likelihood Computation


은닉마코프모델 추론(inference)과 디코딩(decoding)의 핵심은 우도(likelihood)를 계산하는 데 있습니다. 은닉마코프모델의 **우도란 모델 $\lambda$가 주어졌을 때 관측치(observation) 시퀀스 $O$가 나타날 확률**, 즉 $P(O \| \lambda)$를 가리킵니다. 우리는 아이스크림 소비 개수를 $O$로 놓고 모델링을 하고 있는데요. 이렇게 관측된 $O$가 아이스크림 [3개, 1개, 1개]라고 합시다. 모델로부터 $O$가 관측될 확률을 구하는 것이 우도 계산 과정입니다. 이해를 돕기 위해 그림2를 봅시다.


## **그림2** 날씨가 [hot, hot, cold]일 때 아이스크림 개수 [3, 1, 3]이 관측될 likelihood
{: .no_toc .text-delta }
<img src="https://i.imgur.com/DuZ9kFU.png" width="300px" title="source: imgur.com" />


그림2는 그림1에서 우리 관심사인 우도 계산에 필요한 요소들을 찾아서 다시 그린 것일뿐 안에 있는 숫자들은 바뀐 게 없습니다. 그림2로부터 날씨가 [hot, hot, cold]일 때 아이스크림 개수 [3, 1, 3]이 관측될 확률, 즉 우도를 계산한 것은 수식3과 같습니다. 우선 사흘 간의 날씨가 [hot, hot, cold]일 상태 확률을 구합니다(마코프 체인을 따른다고 가정하므로 각 상태 확률은 직전 상태에서 현재 상태의 전이확률들만을 고려, 단 첫번째 상태가 `HOT`일 확률은 초기 상태 분포 $\pi$에서 참조). 그리고 사흘 간 각각 [3, 1, 3]개의 아이스크림이 팔릴 확률을 구합니다(방출확률들의 곱셈). 이 사건은 동시에 일어나므로 우도는 이들 모든 확률들의 곱셈이 됩니다.

## **수식3** 날씨가 [hot, hot, cold]일 때 아이스크림 개수 [3, 1, 3]이 관측될 likelihood
{: .no_toc .text-delta }
$$P(3\quad 1\quad 3,hot\quad hot\quad cold)=P(hot|start)\times P(hot|hot)\times P(cold|hot)\\ \times P(3|hot)\times P(1|hot)\times P(3|cold)\\=0.8\times0.6\times0.3\times0.4\times0.2\times0.1 = 0.001536
$$


우리가 구한 것은 경우의 수의 일부분입니다. 사흘 간 각 날짜별로 날씨가 더울 수도 있고 추울 수도 있습니다. 따라서 $2^3$가지의 경우의 수가 존재합니다. 8가지 경우의 수를 모두 계산해 해당 확률들(표1의 value)을 모두 더한 것(각 날짜별로 날씨가 더울 수도 있고 추울 수 있으므로 or 조건)이 사흘 간의 아이스크림 소비가 [3, 1, 3]일 확률, 즉 우도가 됩니다. 그 결과는 0.02856입니다. 표1과 같습니다.


## **표1** likelihood (naive)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/55kR9nK.png" title="source: imgur.com" />


---


## Forward/Backward Algorithm


위의 방식대로 우도를 계산하려면 비효율이 큽니다. 은닉 상태가 $N$개, 관측치 수가 $T$개일 때 계산복잡도(time complexity)가 $O(T \times N^T)$나 됩니다. 이러한 비효율성을 완화하기 위해 다이내믹 프로그래밍(dynamic programming) 기법을 씁니다. 다이내믹 프로그래밍은 중복되는 계산을 저장해 두었다가 나중에 다시 써먹는 것이 핵심 원리입니다. 히든마코프모델에서 우도를 계산할 때 시간 순으로 다이내믹 프로그래밍을 적용한 기법을 가리켜 `Forward Algorithm`이라고 합니다. 반대로 시간의 역순으로 적용한 기법을 `Backward Alogorithm`이라고 합니다. 그림4를 보겠습니다.


## **그림4** Forward Algorithm
{: .no_toc .text-delta }
<img src="https://i.imgur.com/yXknXfe.png" title="source: imgur.com" />

예컨대 아이스크림 3개($o_1$)와 1개($o_2$)가 연속으로 관측됐고 두 번째 시점($t=2$)의 날씨가 추웠을($q_1$) 확률은 $α_2(1)$입니다. 마찬가지로 아이스크림 3개($o_1$)가 관측됐고 첫 번째 시점($t=1$)의 날씨가 추웠을($q_1$) 확률은 α1(1)입니다. 또한 아이스크림 3개($o_1$)가 관측됐고 첫 번째 시점($t=1$)의 날씨가 더웠을($q_2$) 확률은 $α_1(2)$입니다. 이들 $\alpha$들을 전방 확률(forward probability)이라고 합니다. 전방 확률을 구하는 식은 다음과 같습니다.


## **수식4** forward computation
{: .no_toc .text-delta }
$${ \alpha  }_{ 1 }(1)=P(cold|start)\times P(3|cold)\\ { \alpha  }_{ 1 }(2)=P(hot|start)\times P(3|hot)\\ { \alpha  }_{ 2 }(1)={ \alpha  }_{ 1 }(1)\times P(cold|cold)\times P(1|cold) +{ \alpha  }_{ 1 }(2)\times P(cold|hot)\times P(1|cold)$$



`Forward Algorithm`의 핵심 아이디어는 이렇습니다. 중복되는 계산은 그 결과를 어딘가에 저장해 두었다가 필요할 때마다 불러서 쓰자는 겁니다. 위 그림과 수식을 보시다시피 $α_2(1)$를 구할 때 직전 단계의 계산 결과인 $α_1(1)$, $α_1(2)$를 활용하게 됩니다. 이해를 돕기 위한 예시여서 지금은 계산량 감소가 도드라져 보이지는 않지만 데이터가 조금만 커져도 그 효율성은 명백해집니다. $j$번째 상태에서 $o_1,…,o_t$가 나타날 전방확률 $α$는 수식5와 같이 정의됩니다.


## **수식5** forward probability
{: .no_toc .text-delta }
$${ \alpha  }_{ t }(j)=\sum _{ i=1 }^{ n }{ { \alpha  }_{ t-1 }(i)\times { a }_{ ij } } \times { b }_{ j }({ o }_{ t })$$
- $a_{ij}$: $i$번째 상태에서 $j$번째 상태로 전이할 확률
- $b_j(o_t)$: $j$번째 상태에서 $t$번째 관측치 $o_t$가 나타날 방출 확률


그림5와 그림6은 그림4/수식5에 제시된 $α_2(1)$, $α_2(2)$를 각각 엑셀로 구현한 것을 나타냅니다. 이들을 구할 때 직전 단계의 계산 결과인 $α_1(1)$, $α_1(2)$를 활용함을 알 수 있습니다. 이 엑셀 구현은 조희승 님이 2020년 모두의 연구소 풀잎스쿨 '음성인식 부트캠프'에서 발표한 내용을 인용한 것임을 밝혀둡니다. 엑셀 파일은 [이곳](https://drive.google.com/open?id=1riQx5tcnekuxCbgZELYAEVizOqbBmqNT)에서 다운로드받을 수 있습니다.


## **그림5** forward computation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tp7bIXc.png" width="700px" title="source: imgur.com" />

## **그림6** forward computation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/7aQEyh5.png" width="700px" title="source: imgur.com" />


표2는 Forward/Backward computation으로 구한 우도 계산 결과입니다. 관측치 [3, 1, 3]에 대한 Forward 방식으로 구한 우도는 $α_3(1) + α_3(2) = 0.023496 + 0.005066 = 0.028562$입니다. 이는 앞서 표1에서 나이브하게 구한 결과와 동치입니다. 아울러 본문에서는 생략을 설명했지만 시간의 역순으로 다이내믹 프로그래밍을 적용하는 `Backward Algorithm`으로 구한 결과 역시 동일함을 확인할 수 있습니다.


## **표2** likelihood (forward and backward)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/BEx2aSl.png" title="source: imgur.com" />


`Forward/Backward algorithm`을 적용하게 되면 계산복잡도가 $O(T \times N^T)$에서 $O(N^2 \times T)$로 줄어든다고 합니다. 드라마틱한 감소를 보이는 셈입니다. 관측치 길이 $T$와 은닉 상태 수 $N$에 따른 계산복잡도 비교 표는 다음과 같습니다.


## **표3** 계산복잡도 비교
{: .no_toc .text-delta }

|$T$|$N$|naive|forword/backward|
|---|---|---|---|
|10|5|97656250|250|
|10|10|1E+11|1000|
|10|15|5.7665E+12|2250|
|10|20|1.024E+14|4000|
|10|25|9.53674E+14|6250|
|5|10|500000|500|
|10|10|1E+11|1000|
|15|10|1.5E+16|1500|
|20|10|2E+21|2000|
|25|10|2.5E+26|2500|


---


## Decoding

우리의 두 번째 관심은 모델 $λ$과 관측치 시퀀스 $O$가 주어졌을 때 가장 확률이 높은 은닉상태의 시퀀스 $Q$를 찾는 것입니다. 이를 디코딩(decoding)이라고 합니다. 음성 인식 문제로 예를 들면 입력 음성 신호의 연쇄를 가지고 음소(또는 단어) 시퀀스를 찾는 것입니다. 우리가 은닉마코프모델을 만드려는 근본 목적에 닿아 있는 문제가 됩니다. 은닉마코프모델의 디코딩 과정엔 비터비 알고리즘(Viterbi Algorithm)이 주로 쓰입니다.


### Viterbi Probability

비터비 알고리즘의 계산 대상인 비터비 확률(Viterbi Probability) $v$는 다음과 같이 정의됩니다. $v_t(j)$는 $t$번째 시점의 $j$번째 은닉상태의 비터비 확률을 가리킵니다. 수식6과 같습니다.


## **수식6** viterbi probability
{: .no_toc .text-delta }
$${ v }_{ t }(j)=\max _{ i } ^{n}{ \left[ { v }_{ t-1 }(i)\times { a }_{ ij }\times { b }_{ j }({ o }_{ t }) \right]  }$$


자세히 보시면 `Forward Algoritm`에서 구하는 전방확률 $α$와 디코딩 과정에서 구하는 비터비 확률 $v$를 계산하는 과정이 거의 유사한 것을 확인할 수 있습니다. `Forward Algorithm`은 각 상태에서의 $α$를 구하기 위해 가능한 모든 경우의 수를 고려해 그 확률들을 더해줬다면(sum), 디코딩은 그 확률들 가운데 최대값(max)에 관심이 있습니다. 디코딩 과정을 설명한 예시 그림은 그림8과 같습니다.


## **그림8** viterbi decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/8KkpNl3.png" title="source: imgur.com" />


각 상태에서의 비터비 확률 $v$를 구하는 식은 다음과 같습니다. 전방확률을 계산하는 과정과 비교해서 보면 그 공통점(각 상태에서의 전이확률과 방출확률 간 누적 곱)과 차이점(sum vs max)을 분명하게 알 수 있습니다. 비터비 확률 역시 직전 단계의 계산 결과를 활용하는 다이내믹 프로그래밍 기법을 씁니다.


## **수식7** viterbi probability
{: .no_toc .text-delta }
$$v_{ 1 }(1)=\max { \left[ P(cold|start)\times P(3|cold) \right]  } = P(cold|start)\times P(3|cold)\\ { v }_{ 1 }(2)=\max { \left[ P(hot|start)\times P(3|hot) \right]  } =P(hot|start)\times P(3|hot)\\ { v }_{ 2 }(1)=\max { \left[ { v }_{ 1 }(2)\times P(cold|hot)\times P(1|cold),\\ { v }_{ 1 }(1)\times P(cold|cold)\times P(1|cold) \right]  }$$


그림9와 그림10은 그림8/수식7에 제시된 비터비 확률 $v_2(1)$, $v_2(2)$를 각각 엑셀로 구현한 것을 나타냅니다. 참고용으로 남겨둡니다.


## **그림9** viterbi probability
{: .no_toc .text-delta }
<img src="https://i.imgur.com/IKlnTDa.png" width="700px" title="source: imgur.com" />

## **그림10** viterbi probability
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kmboccP.png" width="700px" title="source: imgur.com" />



### Viterbi Backtrace

`Forward Algorithm`과 비터비 알고리즘 사이에 가장 큰 차이점은 비터비에 역추적(backtracking) 과정이 있다는 점입니다. 디코딩의 목적은 비터비 확률이 얼마인지보다 최적 상태열이 무엇인지에 관심이 있으므로 당연한 이치입니다. 그림8에서 파란색 점선으로 된 역방향 화살표가 바로 역추적을 나타내고 있습니다. 이해를 돕기 위해 그림8을 다시 가져왔습니다.

## **그림8** viterbi backtrace
{: .no_toc .text-delta }
<img src="https://i.imgur.com/8KkpNl3.png" title="source: imgur.com" />

예컨대 2번째 시점 2번째 상태 $q_2$(=HOT)의 backtrace $b_{t_2}(2)$는 $q_2$입니다. $q_2$를 거쳐서 온 우도값($0.32×0.12=0.0384$)이 $q_1$을 거쳐서 온 것($0.02×0.25=0.005$)보다 크기 때문입니다. 

2번째 시점의 1번째 상태 $q_1$(=COLD)의 backtrace $b_{t_2}(1)$는 $q_2$입니다. $q_2$를 거쳐서 온 우도값($0.32×0.20=0.064$)이 $q_1$을 거쳐서 온 것($0.02×0.10=0.002$)보다 크기 따문입니다.

$t$번째 시점 $j$번째 상태의 backtrace는 수식8과 같이 정의됩니다.

## **수식8** viterbi backtrace
{: .no_toc .text-delta }
$$\DeclareMathOperator*{\argmax}{argmax} { b }_{ { t }_{ t } }(j)=\argmax _{ i=1 }^n{ \left[ { v }_{ t-1 }(i)\times { a }_{ ij }\times { b }_{ j }({ o }_{ t }) \right]  }$$


### Best Path

이제 최적 상태열을 구할 준비가 되었습니다. 예컨대 우리가 구하고 싶은 것이 아이스크림 [3개, 1개]가 관측됐을 때 가장 확률이 높은 은닉상태의 시퀀스라고 합시다. 우선 비터비 확률을 끝까지 모두 계산합니다. 그리고 backtrace 또한 구해 놓습니다. 

비터비 디코딩은 마지막 상태부터 첫번째 상태까지 시간의 역순으로 이뤄집니다. 마지막 시점(지금 예시에서는 $t=2$) 비터비 확률이 최대인 상태 $q$를 고릅니다. 예시에서는 $v_2(1)=0.064$, $v_2(2)=0.038$이므로 $v_2(1)$에 대응하는 상태 $q_1$(=COLD)이 비터비 디코딩의 시작이 되겠습니다.

우리는 이미 backtrace를 구해 놓았으므로 2번째 시점의 1번째 상태 $q_1$(=COLD)의 backtrace $b_{t_2}(1)$를 알 수 있습니다. 바로 $q_2$(=HOT)입니다. 이러한 방식으로 time step상 첫번째 상태가 될 때까지 반복합니다. 최적상태열은 이렇게 구한 backtrace들이 리스트에 저장된 결과입니다. 예컨대 위 그림에서 아이스크림 [3개, 1개]가 관측됐을 때 가장 확률이 높은 은닉상태의 시퀀스는 [HOT, COLD]가 되는 것입니다. 

한편 비터비 디코딩 과정을 애니메이션으로 만든 그림은 그림9와 같습니다([출처](https://www.researchgate.net/publication/273123953_Animation_of_the_Viterbi_algorithm_on_a_trellis_illustrating_the_data_association_process)). 현재 상태로 전이할 확률이 가장 큰 직전 스테이트를 모든 시점, 모든 상태에 대해 구합니다. 

## **그림10** Viterbi Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bHji1M9.gif" width="500px" title="source: imgur.com" />

모든 시점, 모든 상태에 대해 구한 결과는 그림11과 같습니다. (원래는 그물망처럼 촘촘하게 되어 있으나 경로가 끊어지지 않고 처음부터 끝까지 연결되어 있는 경로가 유효할 것이므로 그래프를 그린 사람이 이해를 돕기 위해 이들만 남겨 놓은 것 같습니다)

## **그림11** Viterbi Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/PXxizNe.png" width="500px" title="source: imgur.com" />

위 패스에서 만약 최대 확률을 내는 $k+2$번째 시점의 상태가 $θ_0$라면 *backtrace* 방식으로 구한 최적 상태열은 다음과 같습니다. 비터비 디코딩과 관련해 추가로 살펴보시려면 [이곳](https://ratsgo.github.io/data%20structure&algorithm/2017/11/14/viterbi)을 참고하시면 좋을 것 같습니다.

- $[θ_0, θ_2, θ_2, θ_1, θ_0, θ_1]$


---


## Training

은닉마코프모델의 파라메터는 전이확률 $A$와 방출확률 $B$입니다. 그런데 이 두 파라메터를 동시에 추정하기는 어렵습니다. 지금까지 설명한 날씨 예제를 기준으로 하면, 우리가 관측가능한 것은 아이스크림의 개수뿐이고 궁극적으로 알고 싶은 날씨는 숨겨져 있습니다. 이럴 때 주로 사용되는 것이 **Expectation-Maximization(EM) 알고리즘**입니다. 은닉마코프모델에서는 이를 '바움-웰치 알고리즘'이라고 부릅니다. 그림12와 같습니다. 자세한 내용은 [이 아티클](https://ratsgo.github.io/speechbook/docs/am/baumwelch)을 참고하시면 좋겠습니다.


## **그림12** EM algorithm
{: .no_toc .text-delta }
<img src="https://i.imgur.com/8ozQ8lB.png" width="500px" title="source: imgur.com" />


---


## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/9.pdf)
- [조희승, 4주차 Hidden Markov Models, 모두의연구소 음성인식 부트캠프, 2020. 2. 8.](https://home.modulabs.co.kr/product/%ec%9d%8c%ec%84%b1-%ec%9d%b8%ec%8b%9d-%eb%b6%80%ed%8a%b8%ec%ba%a0%ed%94%84)
- [Animation of the Viterbi algorithm on a trellis illustrating the data association process](https://www.researchgate.net/publication/273123953_Animation_of_the_Viterbi_algorithm_on_a_trellis_illustrating_the_data_association_process)


---