---
layout: default
title: Feature Extraction
nav_order: 4
has_children: true
has_toc: false
permalink: /docs/fe
---

# Acoustic Feature Extraction

이 챕터에서는 **Mel-Frequency Cepstral Coefficients(MFCC)**를 추출하는 방법을 살펴봅니다. MFCC는 음성 인식과 관련해 불필요한 정보는 버리고 중요한 특질만 남긴 것입니다. MFCC는 [기존 시스템](https://ratsgo.github.io/speechbook/docs/am)은 물론 최근 엔드투엔드(end-to-end) 기반 모델에 이르기까지 음성 인식 시스템에 널리 쓰이는 피처인데요. [뉴럴네트워크 기반 피처 추출](https://ratsgo.github.io/speechbook/docs/neuralfe) 방식과는 달리 음성 도메인의 지식과 공식에 기반한 추출 방법이며 음성 입력이 주어지면 피처가 고정된(deterministic) 형태입니다. 

이 챕터의 핵심은 [MFCCs](https://ratsgo.gihub.io/speechbook/docs/fe/mfcc)인데요. 이 아티클을 읽기 전에 그의 기반 지식이 되는 [푸리에 변환(Fourier Transform)](https://ratsgo.github.io/speechbook/docs/fe/ft) 항목을 먼저 읽어보시기를 권해 드립니다. 푸리에 변환은 시간과 주파수 도메인 관련 함수를 이어주는 수학적인 연산입니다. 음성 신호의 스펙트럼 분석 등에 필수적입니다. 


---