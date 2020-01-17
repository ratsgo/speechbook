---
layout: default
title: SincNet
nav_order: 2
parent: Neural Feature Extraction
permalink: /docs/neuralfe/sincnet
---

# SincNet (중간 진행)
{: .no_toc }


뉴럴네트워크 기반 피처 추출 기법 가운데 하나인 SincNet 모델을 살펴봅니다. SincNet은 벤지오 연구팀이 2019년 발표한 [SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET](https://arxiv.org/pdf/1808.00158.pdf) 논문에서 제안됐는데요. 음성 피처 추출에 유리한 컨볼루션 신경망(Convolutional Neural Network)의 첫번째 레이어에 싱크 함수(sinc function)를 도입해 계산 효율성과 성능 두 마리 토끼를 잡아서 주목받았습니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## 모델 개요

SincNet 모델 개요는 다음과 같습니다. 저자들은 음성 피처 추출에 첫번째 레이어가 가장 중요하다고 보고 해당 레이어에 `싱크 함수(sinc function)`로 컨볼루션 필터를 만들었습니다. 이들 필터들은 원래 음성(raw waveform) 각각의 주파수 영역대 정보를 추출해 상위 레이어로 보냅니다. 

## **그림1** sincnet 
{: .no_toc .text-delta }
<img src="https://i.imgur.com/n1EXsWV.png" width="400px" title="source: imgur.com" />

그 위 레이어들은 여느 뉴럴넷에 있는 구조와 별반 다르지 않고요. 화자(speaker)가 누구(index)인지 맞추는 과정에서 SincNet이 학습됩니다. SincNet 모델을 이해하기 위해서는 `시간(time) 도메인에서의 컨볼루션 연산` 개념을 이해할 필요가 있습니다. 다음 장에서 차례대로 살펴보겠습니다.


---


## 시간 도메인에서의 컨볼루션 연산

시간 도메인에서의 컨볼루션 연산의 정의는 다음 수식1과 같습니다. $x\left[ n \right]$은 시간 도메인에서의 $n$번째 raw wave sample, $h\left[ n \right]$은 컨볼루션 필터(filter, 1D 벡터)의 $n$번째 요소값, $y\left[ n \right]$은 컨볼루션 수행 결과의 $n$번째 값을 각각 가리킵니다. $L$은 필터의 길이(length)를 나타냅니다.


## **수식1** 시간 도메인에 대한 컨볼루션 연산
{: .no_toc .text-delta }
$$y\left[ n \right] =x\left[ n \right] \ast h\left[ n \right] =\sum _{ l=0 }^{ L-1 }{ x\left[ l \right] \cdot h\left[ n-1 \right]  }$$



예컨대 필터 길이 $L$이 3이라면 수식1에 따라 게산되는 과정은 다음과 같을 겁니다.


## **수식2** 컨볼루션 연산 예시
{: .no_toc .text-delta }
$$y\left[ 0 \right] =x\left[ 0 \right] \cdot h\left[ 0 \right] \\ y\left[ 1 \right] =x\left[ 0 \right] \cdot h\left[ 1 \right] +x\left[ 1 \right] \cdot h\left[ 0 \right] \\ y\left[ 2 \right] =x\left[ 0 \right] \cdot h\left[ 2 \right] +x\left[ 1 \right] \cdot h\left[ 1 \right] +x\left[ 2 \right] \cdot h\left[ 0 \right] \\ y\left[ 3 \right] =x\left[ 0 \right] \cdot h\left[ 3 \right] +x\left[ 1 \right] \cdot h\left[ 2 \right] +x\left[ 2 \right] \cdot h\left[ 1 \right] $$


이렇게만 보면 정말 알쏭달쏭하죠. 그림으로 이해해 봅시다. 일단 입력 wave 시그널이 그림2, 컨볼루션 필터가 그림3과 같다고 합시다. 

## **그림2** input waveform 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TNF1k2k.jpg" width="400px" title="source: imgur.com" />

## **그림3** 컨볼루션 필터 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/7Fp6Zn0.jpg" width="400px" title="source: imgur.com" />

컨볼루션 연산은 **컨볼루션 필터를 Y축으로 뒤집어 필터와 time step별 입력을 내적(inner product)하는 것과 같습니다.** 그림4를 수식2와 비교해서 보면 정확히 들어맞음을 확인할 수 있습니다.

## **그림4** 컨볼루션 연산 예시
{: .no_toc .text-delta }
<img src="https://i.imgur.com/AQ8wz1C.jpg" width="400px" title="source: imgur.com" />
<br>
<img src="https://i.imgur.com/x2qYYXw.jpg" width="400px" title="source: imgur.com" />
<br>
<img src="https://i.imgur.com/BL9UK1h.jpg" width="400px" title="source: imgur.com" />

컨볼루션 연산 결과물인 $y$은 입력 시그널 $x$와 그에 곱해진 컨볼루션 필터 $h$와의 관련성이 높을 수록 커집니다. 혹은 특정 입력 시그널을 완전히 무시할 수도 있습니다. 다시 말해 **컨볼루션 필터는 특정 주파수 성분을 입력 신호에서 부각하거나 감쇄시킨다**는 것입니다. 

컨볼루션 연산을 좀 더 직관적으로 이해해 보기 위해 그림5를 봅시다. 예컨대 시간 도메인에 적용되는 컨볼루션 필터가 그림5 하단 중앙처럼 되어 있다고 가정해 봅시다. 그러면 이 컨볼루션 필터는 입력 시그널 $x$를 그대로 통과시키게 될 겁니다. 이를 주파수 도메인에서 생각해보면 그림5의 상단처럼 연산이 이뤄집니다. 다시 말해 **시간(time) 도메인에서의 컨볼루션 연산은 주파수(frequency) 도메인에서의 곱셈(multiplication) 연산과 동일하다**는 것입니다.

## **그림5** 주파수 도메인의 곱셈 VS 시간 도메인의 컨볼루션 (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/w3ODRrt.jpg" width="400px" title="source: imgur.com" />


또다른 예시를 봅시다. 그림6과 같은 구형 함수를 주파수 도메인에서 입력 신호와 곱하면(multiplication) $f_1$과 $f_2$ 사이의 주파수만 남고 나머지는 없어질 겁니다(그림6의 상단). 이는 시간 도메인에서 Sinc function으로 입력 신호에 컨볼루션 연산을 수행한 결과에 대응합니다(그림6의 하단).


## **그림6** 주파수 도메인의 곱셈 vs 시간 도메인의 컨볼루션 (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bVDA6Qo.jpg" width="400px" title="source: imgur.com" />


---


## Bandpass filter : Sinc Function

우리는 음성 신호 $x$ 가운데 화자 인식 등 문제 해결에 도움이 되는 주파수 영역대(band)는 살리고, 나머지 주파수 영역대는 무시하길 원합니다. 이렇게 특정 주파수 영역대만 남기는 역할을 하는 함수를 `bandpass filter`라고 부릅니다. 주파수 도메인에서 이런 역할을 이상적으로 할 수 있는 필터의 모양은 사각형(Rectangular) 모양일 겁니다. 그림7처럼 말이죠.


## **그림7** Bandpass Filter
{: .no_toc .text-delta }
<img src="https://i.imgur.com/FgzqVBY.jpg" width="400px" title="source: imgur.com" />


이때 싱크 함수라는게 제법 요긴하게 쓰입니다. **주파수(frequency) 도메인에서 구형 함수(Rectangular function)으로 곱셈 연산**을 수행한 결과는 **시간(time) 도메인에서 싱크 함수로 컨볼루션 연산**을 적용한 것과 동치(equivalent)이기 때문입니다. SincNet 저자들이 첫번째 컨볼루션 레이어의 필터로 싱크 함수를 사용하고 모델 이름도 SincNet이라고 지은 이유입니다. 그 식은 수식3과 같습니다.


## **수식3** sinc function
{: .no_toc .text-delta }
$$\text{ sinc } {\left( x \right)} ={ \sin { \left( x \right) }  }/{ x }$$ 


조금 더 자세히 살펴보겠습니다. 싱크 함수를 푸리에 변환(Fourier Transform)한 결과는 구형 함수가 됩니다(수식4, 단 여기에서 싱크 함수는 [정규화된 싱크 함수\[normalized sinc function\]](https://en.wikipedia.org/wiki/Sinc_function)). 이 구형 함수를 역푸리에 변환(Inverse Fourier Transform)하면 다시 싱크 함수가 됩니다. 

그 역도 성립합니다. 구형 함수를 푸리에 변환한 결과는 싱크 함수가 됩니다(수식5). 이 싱크 함수를 역푸리에 변환을 하면 다시 구형 함수가 됩니다. 요컨대 **두 함수는 푸리에 변환을 매개로 한 쌍**을 이루고 있다는 이야기입니다. 그 관계는 다음 그림6과 같으며 구형 함수의 정의는 수식6과 같습니다.

## **그림6** 싱크/구형 함수의 관계
{: .no_toc .text-delta }
<img src="https://i.imgur.com/6LnAejl.jpg" width="400px" title="source: imgur.com" />

그림6 상단 각각의 봉우리를 `lobe`라고 합니다. 가장 높은 봉우리(main lobe)는 해당 컨볼루션 필터가 주로 잡아내는 주파수 영역대가 될 겁니다. 하지만 이것 말고도 작은 봉우리(side lobe)들이 많습니다. 이들 주파수 영역대도 컨볼루션 연산으로 살아남기 때문에 컨볼루션 연산을 적용한 이후에 노이즈로 작용할 수 있습니다. 이를 `side lobe effect`라고 합니다.
{: .fs-4 .ls-1 .code-example }

## **수식4** 싱크 함수의 푸리에 변환
{: .no_toc .text-delta }
<img src="https://i.imgur.com/2cZp0Ky.png" width="300px" title="source: imgur.com" />

## **수식5** 구형 함수의 푸리에 변환
{: .no_toc .text-delta }
<img src="https://i.imgur.com/hCoYfkY.png" width="400px" title="source: imgur.com" />

## **수식6** 구형 함수
{: .no_toc .text-delta }
<img src="https://i.imgur.com/u1xY7P1.png" width="250px" title="source: imgur.com" />


설명의 편의를 위해 `시간 도메인/싱크 함수`, `주파수 도메인/구형 함수`을 한 묶음으로 표현했으나 주파수 도메인에서 싱크 함수를, 시간 도메인에서 구형 함수를 정의할 수 있습니다. 이렇게 도메인이 바뀐 상태에서도 역시 싱크/구형 함수는 푸리에 변환을 매개로 한 쌍을 이룹니다.
{: .fs-4 .ls-1 .code-example }

---


## windowing

수식4에서 알 수 있듯 싱크 함수에 푸리에 변환을 적용해 완전한 형태의 구형 함수를  얻어내려면 해당 싱크 함수가 무한한 길이를 가져야 합니다(적분 구간 참조). SincNet 저자들은 시간 도메인의 입력 음성 신호 $x$에 싱크 함수로 컨볼루션 연산을 적용하려고 하는데요. 여기에 문제가 하나 있습니다. 컴퓨터 성능이 아무리 좋더라도 무한한 길이의 싱크 함수(=컨볼루션 필터)를 사용할 수는 없습니다. 따라서 그림7의 상단과 같이 싱크 함수를 적당히 잘라서 사용해야 할 겁니다.


## **그림7** Filter Truncation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/MwqRypM.jpg" width="250px" title="source: imgur.com" />


싱크 함수를 유한한 길이로 자르고 이를 푸리에 변환을 하면 그림8의 하단과 같은 모양이 됩니다. 이상적인 bandpass filter의 모양(사각형)에서 점점 멀어지게 되는 것이죠. 이렇게 되면 우리가 원하는 주파수 영역대 정보는 덜 캐치하게 되고, 버려야 하는 주파수 영역대 정보도 캐치하게 됩니다.


## **그림8** 싱크 함수의 길이별 비교
{: .no_toc .text-delta }
<img src="https://i.imgur.com/dPiXPQ6.png" width="350px" title="source: imgur.com" />


이러한 문제를 해결하기 위해 SincNet 저자들은 `윈도우(window)` 기법을 적용했습니다. 싱크 함수를 특정 길이로 자르고 해당 필터에 윈도우 함수값을 곱해 양끝을 스무딩한다는 개념입니다. SincNet 저자들은 윈도우 기법으로 해밍 윈도우(Hamming window)를 사용햇습니다. 수식7과 같습니다.

## **수식7** Hamming Window
{: .no_toc .text-delta }
$$w\left[ n \right] =0.54-0.46\cdot \cos { \left( \frac { 2\pi n }{ L }  \right)  }$$
<br>
<img src="https://i.imgur.com/tHPxKTg.png" width="350px" title="source: imgur.com" />


그림9는 해밍 윈도우를 푸리에 변환한 결과입니다. 중심 주파수 영역대는 잘 캐치하고 그 외 주파수 영역대는 무시하게 됩니다. 다시 말해 유한한 길이의 싱크 함수를 사용하더라도 해밍 윈도우 기법을 사용하면 원하는 주파수 영역대 정보를 잘 살리고, 버려야 할 주파수 영역대 정보는 잘 버리는 보완책이 될 수 있다는 이야기입니다.


## **그림9** 해밍 윈도우의 푸리에 변환
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Hsi7qpn.png" width="350px" title="source: imgur.com" />



**입력 신호에 대한 windowing**
<br>
본래 윈도우라는 개념은 시간 도메인에서의 입력 신호 $x$에 시행하는 것이 정석입니다. 음성 신호는 그 길이가 매우 길기 때문에 프레임별로 잘라서 처리를 합니다. 그런데 입력 신호를 인위적으로 자를 경우 문제가 발생합니다. 아래 그림에서 관찰할 수 있는 것처럼 자른 부분에서 직각 신호(square wave)가 생기는 것이죠. 
<br>
<img src="https://i.imgur.com/FI2dGV7.jpg" width="250px" title="source: imgur.com" />
<br>
이렇게 뾰족한(sharp) 구간에서는 [깁스 현상(Gibbs phenomenon)](https://en.wikipedia.org/wiki/Gibbs_phenomenon)이 발생한다고 합니다. 깁스 현상을 이해해보기 위해 아래 그림을 보겠습니다.
<br>
<img src="https://i.imgur.com/tRbgp3x.png" width="250px" title="source: imgur.com" />
<br>
위의 첫번째 그림은 서로 다른 주기(cycle)를 가진 사인(sin) 함수 다섯 개를 합해서 만든 그래프입니다. 세번째 그림은 125개를 썼습니다. 자세히 보시면 사인 함수를 많이 쓸 수록 직각에 가까운 신호를 만들어낼 수는 있지만, 굴절되는 구간에서 여전히 뾰족하게 튀어나온 부분을 확인할 수 있습니다. **주파수 성분(=사인 함수)를 무한히 더해야 직각 신호를 완벽하게 표현할 수 있다**가 깁스 현상의 핵심 개념입니다.
<br>
입력 신호를 프레임 단위로 자르는 과정에서 직각 신호가 생겼습니다. 결과적으로 입력 신호에 고주파 성분(신호가 갑자기 튀거나 작아지는 등)이 생깁니다. 이런 문제를 해결하기 위해 `윈도우(window)`가 제안됐습니다. 입력 신호에 윈도우를 곱해 신호 양 끝부분을 스무딩하는 것입니다. 
<br>
SincNet 저자들은 컨볼루션 필터(=싱크 함수)에 해밍 윈도우를 적용했는데요. 컨볼루션 연산 식을 보면 **(1) 컨볼루션 필터에 해밍 윈도우 적용 (2) 입력 신호에 해밍 윈도우 적용** 두 가지 방식이 수치 연산 면에서는 정확하게 동치임을 확인할 수 있습니다.
{: .fs-4 .ls-1 .code-example }


---


## Sinc/Rectangular function의 관계

특정 Sinc Function을 푸리에 변환(Fourier Transform)한 결과는 Rectangular function이 됩니다. 그 역도 성립합니다. 그 관계는 다음 그림5와 같습니다.

## **그림5** Sinc/Rectangular functions
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Fv8YDzS.jpg" width="400px" title="source: imgur.com" />


Sinc Function은 시간 도메인, Rectangular function은 주파수 도메인에 연관이 있습니다. 시간 도메인에서 Sinc Function으로 입력 신호를 컨볼루션 연산하는 것과 주파수 도메인에서 Rectangular function으로 multiplication하는 것은 동치(equivalent)입니다.








---


## SincNet

먼길을 돌아왔습니다. 자 이제 SincNet을 살펴볼 시간입니다. 수식4는 입력 음성 신호에 대해 모델의 첫번째 레이어에서 SincNet 컨볼루션을 수행하는 걸 나타냅니다. $n$은 입력 음성 시그널의 $n$번째 샘플, $g$는 $n$번째 입력 시


## **수식4** SincNet
{: .no_toc .text-delta }
$$y\left[ n \right] =x\left[ n \right] \ast g\left[ n,\theta  \right]$$


## **수식5** 주파수 도메인에서의 SincNet 연산
{: .no_toc .text-delta }
$$G\left[ f,{ f }_{ 1 },{ f }_{ 2 } \right] =\text{ rect } \left( \frac { f }{ { 2f }_{ 2 } }  \right) - \text{ rect } \left( \frac { f }{ { 2f }_{ 1 } }  \right)$$

## **수식6** 시간 도메인에서의 SincNet 연산
{: .no_toc .text-delta }
$$g\left[ f,{ f }_{ 1 },{ f }_{ 2 } \right] =2{ f }_{ 2 } \text{ sinc } \left( 2\pi { f }_{ 2 }n \right) -2{ f }_{ 1 } \text{ sinc } \left( 2\pi { f }_{ 1 }n \right)$$


## **수식5** constraint
{: .no_toc .text-delta }
$${ f }_{ 1 }^{ abs }=\left| { f }_{ 1 } \right| \\ { f }_{ 2 }^{ abs }={ f }_{ 1 }+\left| { f }_{ 2 }-{ f }_{ 1 } \right|$$


## **수식6** constraint
{: .no_toc .text-delta }
$${ g }_{ w }\left[ n,{ f }_{ 1 },{ f }_{ 2 } \right] ={ g }_{ w }\left[ n,{ f }_{ 1 },{ f }_{ 2 } \right] \cdot w\left[ n \right]$$


## **코드1** Hamming window
{: .no_toc .text-delta }
```python
import torch, math
import torch.nn.functional as F
import numpy as np

out_channels = 80
# kernel size는 홀수로 강제
# 계산효율성 위해 sinc conv filter를 symmetric하게 만들기 위해
# center 앞뒤로 n개씩 (2n+1)
kernel_size = 251
sample_rate = 16000
in_channels = 1
stride = 1
padding = 0
dilation = 1
# sincnet에서는 bias가 없다
bias = False
# sincnet에서는 그룹이 1이다
groups = 1
min_low_hz = 50
min_band_hz = 50

# initialize filterbanks such that they are equally spaced in Mel scale
low_hz = 30 # low_hz는 가청 주파수(frequency response)에 대응? 20~20000Hz
high_hz = sample_rate / 2 - (min_low_hz + min_band_hz) # 7900Hz


def to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


# low cut frequency f1과 high cut frequency f2의 초기값 만들기
# sincnet의 하이퍼파라메터를 바꾸지 않는 한 초기값은 항상 똑같다(deterministic)

# low_hz를 mel scale : 30 > 47.29
# high_hz를 mel scale : 7900 > 2826.99
# 이 범위의 구간을 (out_channels + 1)차원 벡터로 변환
# 따라서 mel[0] = 47.29, mel[-1] = 2826.99
mel = np.linspace(to_mel(low_hz),
                  to_mel(high_hz),
                  out_channels + 1)
# mel을 hz 단위로 다시 변경
# hz[0] = 30, hz[-1] = 7900
# 따라서 hz는 (out_channels + 1)차원 벡터이되
# 각 요소값들 차이는 mel scale이 됨 // 저주파수대 영역을 잘 보기 위해서
hz = to_hz(mel)

# filter lower frequency들의 초기값 (out_channels, 1)
# hz에서 마지막 요소를 제외하고 low_hz_ 만드는데 이용
# 따라서 low_hz_는 (out_channels, 1) 크기가 됨
# trainable scalar
low_hz_ = torch.nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

# filter higher frequency band들의 초기값 (out_channels, 1)
# trainable scalar
# np.diff는 각 요소값들 간 차이를 리턴
# ex) np.diff([1,2,4,8]) = array([1, 2, 4])
band_hz_ = torch.nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

# Hamming window
# computing only half of the window (because symmetric)
n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=int((kernel_size / 2)))
window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)

n = (kernel_size - 1) / 2.0
# Due to symmetry, I only need half of the time axes
n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate

# 이하는 forward
# 'low' cut freq f1과 'high' cut freq f2는 계속 갱신된다
low = min_low_hz + torch.abs(low_hz_)
# clamp : input이 min보다 작으면 min, max보다 크면 max, 그 사이이면 input 그대로 리턴
high = torch.clamp(input=low + min_band_hz + torch.abs(band_hz_),
                   min=min_low_hz,
                   max=sample_rate / 2)
band = (high - low)[:, 0]

f_times_t_low = torch.matmul(low, n_)
f_times_t_high = torch.matmul(high, n_)

# Equivalent of Eq.4 of the reference paper
# I just have expanded the sinc and simplified the terms.
# This way I avoid several useless computations.
band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n_ / 2)) * window_
band_pass_center = 2 * band.view(-1, 1)
band_pass_right = torch.flip(band_pass_left, dims=[1])

# symmetric
band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

band_pass = band_pass / (2 * band[:, None])

# filters shape : out_channels, in_channels, kernel_size
# 초기값에 따르면 0번 필터는 저주파수 영역대 관장, 마지막 필터는 고주파수 영역대 bandpass
# 하지만 전체 필터(out_channels=80개)는 학습 중에 각각이 담당하는 band pass가 변화하게 된다
filters = (band_pass).view(out_channels, 1, kernel_size)

# 이후 이 80개 필터들이 각각 입력 raw wave에 대해 1d conv 실시
sincnet_result = F.conv1d(waveforms, filters, stride=stride,
                          padding=padding, dilation=dilation,
                          bias=None, groups=1)
```