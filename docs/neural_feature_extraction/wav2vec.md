---
layout: default
title: Wav2Vec
nav_order: 1
parent: Neural Feature Extraction
permalink: /docs/neuralfe/wav2vec
---

# Wav2Vec
{: .no_toc }


뉴럴네트워크 기반 피처 추출 기법 가운데 하나인 [Wav2Vec](https://arxiv.org/pdf/1904.05862)/[VQ-Wav2Vec](https://arxiv.org/pdf/1910.05453) 모델을 살펴봅니다. 사람의 개입 없이 음성 특질을 추출하는 방법을 제안해 주목을 받았습니다. 다만 음성 특질의 품질이 [PASE](https://ratsgo.github.io/speechbook/docs/neuralfe/pase)보다는 낮은 경향이 있고 아직은 정립이 되지 않은 방법론이라 생각돼 그 핵심 아이디어만 간략하게 일별하는 방식으로 정리해 보겠습니다. 
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## Wav2Vec

[Wav2Vec](https://arxiv.org/pdf/1904.05862)의 전체 아키텍처는 그림1과 같습니다. Wav2Vec은 크게 encoder network $f$와 context network $g$ 두 개 파트로 구성돼 있습니다. 둘 모두 컨볼루션 뉴럴네트워크(convolutional neural network)입니다.

encoder network $f$는 음성 입력 $\mathcal{X}$를 hidden representaion $\mathcal{Z}$로 인코딩하는 역할을 수행합니다. context network $g$는 $\mathcal{Z}$를 context representation $\mathcal{C}$로 변환합니다. Wav2Vec 학습을 마치면 $\mathcal{C}$를 해당 음성의 피처로 사용합니다.

## **그림1** Wav2Vec
{: .no_toc .text-delta }
<img src="https://i.imgur.com/H9X1HiX.png" width="400px" title="source: imgur.com" />

Wav2Vec은 Word2Vec처럼 해당 입력이 포지티브 쌍인지 네거티브 쌍인지 이진 분류(binary classification)하는 과정에서 학습됩니다. 포지티브 쌍은 그림1처럼 입력 음성의 $i$번째 context representation $\mathcal{C_{i}}$와 $i+1$번째 hidden representaion $\mathcal{Z_{i+1}}$입니다. 네거티브 쌍은 입력 음성의 $i$번째 context representation $\mathcal{C_{i}}$와 현재 배치의 다른 음성의 hidden representation들 가운데 랜덤으로 추출해 만듭니다.

학습이 진행될 수록 포지티브 쌍 관계의 represenation은 벡터 공간에서 가까워지고, 네거티브 쌍은 멀어집니다. 다시 말해 encoder network $f$와 context network $g$는 입력 음성의 다음 시퀀스가 무엇일지에 관한 정보를 음성 피처에 잘 녹여낼 수 있게 됩니다.


---

## VQ-Wav2Vec

[VQ-Wav2Vec](https://arxiv.org/pdf/1910.05453)은 기본적으로는 Wav2Vec의 아키텍처와 같습니다. 다만 중간에 Vector Quantization 모듈이 추가됐습니다. 그림2와 같습니다. encoder network $f$는 음성 입력 $\mathcal{X}$를 hidden representaion $\mathcal{Z}$로 인코딩하는 역할을 수행합니다. Vector Quantization 모듈 $q$는 continous representaion $\mathcal{Z}$를 dicrete representaion $\hat{\mathcal{Z}}$로 변환합니다. context network $g$는 $\hat{\mathcal{Z}}$를 context representation $\mathcal{C}$로 변환합니다.

## **그림2** VQ-Wav2Vec
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ivviYL1.png" width="400px" title="source: imgur.com" />


---

## Vector Quantization

VQ-Wav2Vec의 핵심은 Vector Quantization 모듈입니다. Gumbel Softmax, K-means Clustering 두 가지 방식이 제시됐습니다. 그림3은 Gumbel Softmax를 도식화한 것입니다. 우선 $\mathcal{Z}$를 선형변환해 logit를 만듭니다. 여기에 Gumbel Softmax와 argmax를 취해 원핫 벡터를 만듭니다. 연속적인(continous) 변수 $\mathcal{Z}$가 이산(discrete) 변수로 변환됐습니다. 이것이 바로 Vector Quantization입니다. 

이후 Embedding matrix를 내적해 $\hat{\mathcal{Z}}$를 만듭니다. 결과적으로는 Vector Quantization으로 $V$개의 Embedding 가운데 $e_2$를 하나 선택한 셈이 되는 것입니다.

## **그림3** Gumbel Softmax
{: .no_toc .text-delta }
<img src="https://i.imgur.com/y15Qu5Z.png" width="400px" title="source: imgur.com" />

그런데 여기에서 하나 의문이 있습니다. argmax는 미분이 불가능한데 어떻게 이 방식으로 전체 네트워크를 학습할 수 있을까요? Gumbel Softmax를 파이토치로 구현한 코드를 보면 이해할 수 있습니다. 코드1을 볼까요?

## **코드1** Gumbel Softmax
{: .no_toc .text-delta }
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
```

`gumbel_softmax`의 리턴값, `(y_hard - y).detach() + y`에 주목합시다. 순전파(forward computation)에서는 argmax가 취해져 원핫 벡터가 된 `y_hard`만 다음 레이어 계산에 넘겨줍니다. 역전파(backward computation)에서는 소프트맥스가 취해진 확률 벡터인 `y`에만 그래디언트가 흘러갑니다.

VQ-Wav2Vec 저자들은 K-means Clustering 방식으로도 Vector Quantization을 수행했습니다. $\mathcal{Z}$와 Embedding Matrix 각각의 벡터와 유클리디안 거리를 계산합니다. 여기에서 가장 가까운 Embedding Vector를 하나 선택(그림4 예시에서는 $e_2$가 선택)해 다음 계산으로 넘깁니다. 

## **그림4** K-means Clustering
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nrY2IAx.png" width="400px" title="source: imgur.com" />

argmin 역시 미분이 불가능한데요. Gumbel Softmax 때처럼 순전파와 역전파 과정을 섬세하게 설계하면 미분 가능하게 만들 수 있습니다.

---

## References

- [Schneider, S., Baevski, A., Collobert, R., & Auli, M. (2019). wav2vec: Unsupervised pre-training for speech recognition. arXiv preprint arXiv:1904.05862.](https://arxiv.org/pdf/1904.05862)
- [Baevski, A., Schneider, S., & Auli, M. (2019). vq-wav2vec: Self-supervised learning of discrete speech representations. arXiv preprint arXiv:1910.05453.](https://arxiv.org/pdf/1910.05453)


---