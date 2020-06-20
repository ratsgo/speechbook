---
layout: default
title: PASE
nav_order: 3
parent: Neural Feature Extraction
permalink: /docs/neuralfe/loss
---

# PASE
{: .no_toc }

음성 피처(featrue)를 추출하는 뉴럴넷 가운데 하나인 `PASE`, `PASE+` 모델을 살펴봅니다. 이 모델은 [MFCCs(Mel-Frequency Cepstral Coefficients)](https://ratsgo.github.io/speechbook/docs/fe/mfcc) 같은 음성 도메인의 지식과 공식에 기반한 피처 수준 혹은 그 이상의 성능을 보여 주목받았습니다. MFCCs는 고정된 형태의 피처인 반면 PASE/PASE+는 학습 가능하기 때문에 태스크에 맞춰 튜닝을 함으로써 해당 태스크의 성능을 더욱 끌어올릴 수 있다는 장점 역시 있습니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## PASE/PASE+

PASE는 [싱크넷(SincNet)](https://ratsgo.github.io/speechbook/docs/neuralfe/sincnet)을 기본 뼈대로 하는 아키텍처입니다. 그림1을 보면 싱크넷 위에 7개의 콘볼루션 레이어를 쌓고 1차례의 선형변환(linear transformation)을 수행한 뒤 배치 정규화(Batch Normalization)를 수행하고 있는 걸 볼 수 있습니다. 저자들은 '싱크넷~배치 정규화'에 이르는 레이어를 인코더(encoder)라고 부르고 있습니다.

PASE가 싱크넷과 가장 다른 점은 워커(worker)입니다. 그림1에서 Waveform, MFCC 등이 바로 그것입니다. 이 워커들은 인코더를 학습하기 위해 존재합니다. 예컨대 인코더에서 뽑은 100차원짜리 벡터(그림1에서 BN 상단의 녹색으로 칠한 네모)가 있습니다. 이를 프레임 벡터라고 부르겠습니다.

## **그림1** PASE
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TfLIap1.png" width="400px" title="source: imgur.com" />

워커가 `Waveform`이라고 가정해 봅시다. 이 경우 PASE의 학습은 마치 오토인코더(autoencoder)처럼 진행됩니다. 인코더가 원시 음성을 100차원의 히든 벡터(프레임 벡터)로 인코딩하고, 워커가 이를 원래 음성으로 다시 복원하는 셈이기 때문입니다.

이번엔 워커가 `MFCC`라고 가정해 봅시다. 이 경우 워커는 100차원짜리 프레임 벡터를 해당 프레임의 MFCC가 되도록 변형합니다. 만일 `Waveform`와 `MFCC` 워커가 제대로 학습이 되었다면 100차원짜리 프레임 벡터에는 원래의 음성 정보와 MFCC 정보가 적절하게 녹아들어 있을 것입니다. 

PASE 학습이 끝나면 워커를 제거하고 인코더만 사용합니다. 이 인코더는 원시 음성의 다양하고 풍부한 정보를 추출할 수 있는 능력이 있습니다. 워커 덕분입니다. 아울러 이 인코더는 딥러닝 기반의 모델로 다른 태스크를 수행할 때 같이 학습도 할 수 있다는 장점도 있습니다.

그림2는 PASE의 개선된 버전인 PASE+입니다. PASE 대비 워커의 종류가 늘었고 인코더 구조를 개선하였습니다. 이 글 나머지에서는 그림2의 PASE+를 기준으로 설명하겠습니다.

## **그림2** PASE+
{: .no_toc .text-delta }
<img src="https://i.imgur.com/rd8wH7e.png" width="400px" title="source: imgur.com" />


---

## Encoder

그림3은 PASE+ 인코더 구조를 도시한 것입니다. 우선 원시 음성을 입력으로 받습니다. 단 강건한(robust) 피처 추출기를 만들기 위해 각종 노이즈를 추가합니다(speech distortion). 음성 피처가 아키텍처와 처음 만나는 곳은 [싱크넷(SincNet)](https://ratsgo.github.io/speechbook/docs/neuralfe/sincnet)입니다. 정확히는 싱크넷에서 `SincConv`에 해당합니다. 컨볼루션 필터가 싱크 함수(sinc function)인 1D conv 레이어를 가리킵니다. 이와 관련한 자세한 내용은 싱크넷 항목을 보시기 바랍니다.

## **그림3** Encoder
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1SRxsXd.png" width="250px" title="source: imgur.com" />

싱크넷 이후로는 7개의 1D conv 레이어를 쌓습니다. 원시 음성은 그 길이가 대단히 길기 때문에(Sample rate가 16KHz이고 2초짜리 음성일 경우 그 길이만 32000) 적절하게 줄여줄 필요가 있습니다. 컨볼루션 필터 길이와 stride 등을 잘 주면 꽤 효과적으로 그 길이를 줄일 수 있습니다. 7개의 컨볼루션 레이어는 바로 이런 역할을 수행합니다.

PASE+ 인코더에는 Quasi-Recurrent Neural Network(QRNN)이라는 특이한 구조가 존재합니다. 수식1을 보면 그 계산 과정이 LSTM과 유사하다는 걸 알 수 있습니다. 그런데 $f$, $o$ 따위를 만들 때 1D Conv를 사용하고 있습니다. 다시 말해 $h_t$, $c_t$를 계산할 때 해당 시점의 $f_t$, $o_t$를 그때그때 계산하는게 아니라, 이미 모두 계산이 마쳐진 행렬 형태의 $F$, $O$로부터 $t$ 시점의 벡터 부분을 참조한다는 것입니다. 

## **수식1** Quasi-Recurrent Neural Network
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bszsaeT.png" width="400px" title="source: imgur.com" />

다시 말해 QRNN은 현재 시점의 $h_t$, $c_t$를 계산할 때 직전 계산 결과에 의존하지 않습니다. 따라서 LSTM처럼 시퀀셜한 정보를 캐치하면서도 병렬 처리가 가능해집니다.

## **그림4** Quasi-Recurrent Neural Network
{: .no_toc .text-delta }
<img src="https://i.imgur.com/lccn34L.png" width="200px" title="source: imgur.com" />


PASE+ 인코더의 특징 가운데 하나는 1D conv 레이어에 관한 skip connection입니다. 1D conv 레이어 출력에 이같이 skip connection을 도입한 이유는 PASE+ 인코더의 최종 산출물에 다양한 추상화 수준의 음성 피처를 반영하기 위함입니다. 비유하자면 마을 사진을 찍을 때 1m, 10m, 100m 상공 각각에서 관찰한 모든 결과를 반영하는 셈입니다. skip connection에 Fully Connected Layer가 끼어있는 건 QRNN의 출력과 차원수를 맞춰주기 위해서입니다.

PASE+ 인코더 말단엔 256차원의 선형변환과 배치 정규화(batch normalization) 레이어가 있습니다.

---

## Speech Distortion

PASE+ 인코더 학습을 위해 여섯 가지 노이즈 추가 기법이 적용됐습니다. 인코더 피처 추출에 일반화(generalization) 성능을 높이기 위해서입니다. 다음과 같습니다.

- **Reverberation** : 잔향(殘響, Reverberation)은 소리를 내고 있다가 소리를 끊었을 때 해당 소리가 차츰 감쇄해 가는 현상입니다. 입력 음성에 잔향이 발생한 것처럼 노이즈를 줍니다.
- **Additive Noise** : 알람, 노크 등 노이즈를 더해(add) 입력 음성에 추가합니다.
- **Frequency Mask** : 입력 음성의 특정 주파수 밴드를 제거합니다.
- **Temporal Mask** : 입력 음성의 연속 샘플을 0으로 치환합니다(=특정 시간대 제거).
- **Clipping** : 입력 음성 샘플을 랜덤하게 제거합니다.
- **Overlap Speech** : 입력 음성과 다른 음성을 더해 노이즈를 줍니다.

---

## Worker

PASE+의 워커는 회귀(regression) 혹은 이진 분류(binary classification)를 수행합니다. 워커는 대부분 작은 크기의 피드포워드 뉴럴네트워크(feed-forward network)입니다. 워커는 인코더를 학습하기 위한 용도로, 학습이 끝난 후 제거하고 인코더만 음성 피처 추출기로 씁니다. 

PASE+엔 7가지의 워커가 있는데요. 우선 회귀 태스크를 수행하는 5가지 워커 각각이 수행하는 태스크와 학습 방법은 다음과 같습니다.

- **Wave** : 입력 음성으로 복원합니다. 워커의 출력과 입력 음성 사이의 Mean Squared Error(MSE)를 최소화하는 방향으로 모델을 업데이트합니다. 
- **Log Power Spectrum** : 입력 음성의 Log [Power Spectrum](https://ratsgo.github.io/speechbook/docs/fe/mfcc#power-spectrum)으로 변환합니다. 워커의 출력과 입력 음성의 Log Power Spectrum 사이의 MSE를 최소화하는 방향으로 모델을 업데이트합니다.
- **MFCC** : 입력 음성의 [MFCC](https://ratsgo.github.io/speechbook/docs/fe/mfcc#mfccs)로 변환합니다. 워커의 출력과 입력 음성의 MFCC 사이의 MSE를 최소화하는 방향으로 모델을 업데이트합니다.
- **FBANK** : 입력 음성의 [Filter Bank](https://ratsgo.github.io/speechbook/docs/fe/mfcc#filter-banks)로 변환합니다. 워커의 출력과 입력 음성의 Filter Bank 사이의 MSE를 최소화하는 방향으로 모델을 업데이트합니다.
- **GAMMA** : 입력 음성의 [Gammatone feature](https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram)로 변환합니다. 워커의 출력과 입력 음성의 Gammatone feature 사이의 MSE를 최소화하는 방향으로 모델을 업데이트합니다.
- **Prosody features** : 입력 음성의 prosody feature로 변환합니다. prosody feature는 음성의 강세, 억양 등에 영향을 주는 값들을 가리키는데요. 음성 프레임 단위로 prosody feature를 추출할 수 있습니다. prosody feature에는 해당 프레임 기본 주파수(fundamental frequency)의 로그값, voiced/unvoiced 확률, zero-crossing rate, 에너지(energy) 등이 있습니다. 기본 워커의 출력과 입력 음성 각각의 prosody feature 사이의 MSE를 최소화하는 방향으로 모델을 업데이트합니다.

이진 분류 태스크를 수행하는 워커는 2가지입니다. 두 개의 입력 쌍(pair)이 포지티브 샘플(positive sample)의 관계인지 네거티브 샘플(negative sample)의 관계인지 맞추는 과정에서 학습합니다. 우선 입력 쌍 가운데 하나(anchor sample)는 PASE+ 인코더 출력 프레임 벡터들 가운데 랜덤으로 추출하고요. 나머지 샘플(포지티브, 네거티브 입력)을 뽑는 방법은 다음과 같습니다.

- **Local Info Max** : 포지티브 샘플은 앵커 샘플과 같은 문장에서 랜덤으로 뽑습니다. 네거티브 샘플은 앵커 샘플과 다른 배치에서 랜덤으로 뽑습니다. 포지티브 샘플은 앵커 샘플과 동일한 화자(speaker), 네거티브 샘플은 다른 화자일 가능성이 높습니다. LIM 워커는 PASE+ 인코더로 하여금 화자를 구분하는 능력을 부여해 줍니다.
- **Global Info Max** : 앵커 샘플은 랜덤으로 뽑되 2초 간의 PASE+ 프레임 벡터의 평균을 취합니다. 포지티브 샘플은 앵커와 같은 문장에서 뽑되 역시 2초 간 평균을 취합니다. 네거티브 샘플은 앵커와 동일한 배치 내 다른 문장에서 뽑되 2초 간 평균입니다. GIM 워커는 문장의 경계를 인식하게 합니다. 


---

## Performances

PASE+의 음성 피처 성능은 표1, 표2와 같습니다. 표1, 표2 모두 Phone Error Rate로 낮을 수록 좋은 모델이라는 뜻입니다. 표1을 보면 전통의 강자인 MFCC나 Filter Bank 성능보다 좋습니다.

## **표1** PASE+
{: .no_toc .text-delta }
<img src="https://i.imgur.com/v3ExGNJ.png" width="400px" title="source: imgur.com" />

표2는 ablation study 결과입니다. 지금까지 언급한 Speech Distortion, QRNN, Skip Connection, 워커 등이 피처 품질 개선에 기여하고 있음을 확인하고 있습니다.

## **표2** PASE+
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Q2UJJJw.png" width="400px" title="source: imgur.com" />

---