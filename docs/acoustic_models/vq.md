---
layout: default
title: Vector Quantization
nav_order: 5
parent: Acoustic Models
permalink: /docs/am/vq
---

# Vector Quantization
{: .no_toc }

벡터 양자화(Vector Quantization)에 대해 살펴봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---


## 개요

[은닉 마코프 모델(Hidden Markov Model)](https://ratsgo.github.io/speechbook/docs/am/hmm)에서는 일반적으로 상태 전이에 대한 제약을 두지 않습니다. 하지만 은닉 마코프 모델을 음성 인식에 적용할 때는 `left-to-right` 제약을 둡니다. 다시 말해 현재 시점 기준으로 이후 상태로 전이만 가능하게 할 뿐, 이전 시점 상태로의 전이는 허용하지 않는다는 것입니다. 음성 인식을 위한 은닉 마코프 모델에서 은닉 상태(hidden state)는 음소(또는 단어)가 될텐데요. 음소(또는 단어)는 시간에 대해 불가역적이기 때문에 이렇게 모델링한 것 아닌가 싶습니다. 그림1을 봅시다.


## **그림1** 음성 인식을 위한 HMM 모델링
{: .no_toc .text-delta }
<img src="https://i.imgur.com/0xXAVXX.png" title="source: imgur.com" />



---


## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3)
- [Stanford CS224S - Spoken Language Processing](https://web.stanford.edu/class/cs224s)


---