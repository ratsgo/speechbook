---
layout: default
title: Neural Feature Extraction
nav_order: 6
has_children: true
has_toc: false
permalink: /docs/neuralfe
---

# Neural Acoustic Feature Extraction

이 챕터에서는 뉴럴 네트워크(Neural Network) 기반의 음성 피처 추출 기법들에 대해 살펴봅니다. 뉴럴 네트워크 기반의 추출 기법들은 [Mel-Frequency Cepstral Coefficients(MFCC)](https://ratsgo.gihub.io/speechbook/docs/fe/mfcc) 같은 추출 방식과는 구분됩니다. 

MFCC 같은 종류는 음성 도메인의 지식과 공식에 기반한 추출 방법이며 음성 입력이 주어지면 피처가 고정된(deterministic) 형태입니다. 하지만 이 글에서 소개할 뉴럴 네트워크 기반의 기법들은 뉴럴넷이 특정 목적을 수행하는 과정에서 음성 피처를 추출하는 방식으로 작동합니다. 음성 도메인 지식을 많이 필요로 하지 않으며 음성 입력이 동일하다고 해도 학습 과정에서 피처가 변화할 수 있습니다.

이 글에서는 [Wav2Vec](https://ratsgo.github.io/speechbook/docs/neuralfe/wav2vec)과 [SincNet](https://ratsgo.github.io/speechbook/docs/neuralfe/sincnet) 두 개를 중심으로 살펴봅니다. 전자는 현재 음성 프레임과 다음 음성 프레임의 유사도를 높이는 과정에서 학습되며 후자는 음성 입력을 좀 더 섬세하게 처리하기 위해 제안된 새로운 형태의 컨볼루션 뉴럴네트워크(Convolutional Neural Network)입니다. 마지막으로 [Problem-Agnostic Speech Encoder(PASE)](https://ratsgo.github.io/speechbook/docs/neuralfe/pase)는 음성 피처를 좀 더 잘 추출하기 위해 고안된 아키텍처로 기본적으로는 SincNet 구조를 따릅니다.


---