---
layout: default
title: Viterbi Decoding
nav_order: 2
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

그림1은 은닉마코프모델 기반 음성 인식 모델의 비터비 디코딩 전반을 도식화한 그림입니다.


## **그림1** HMM 기반 디코딩
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bpoTguW.png" width="600px" title="source: imgur.com" />


---


## Forward Computation


## **그림2** Forward Computation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3LzVWZV.png" width="400px" title="source: imgur.com" />


## **표1** Forward Computation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/OgBywaq.png" width="600px" title="source: imgur.com" />


---


## Viterbi Search

그림3, 그림4는 Forward Computation으로부터 베스트 경로 하나만 남겨서 비터비 경로를 탐색(trellis)하는 과정을 개념적으로 나타낸 것입니다. 그림5는 이로부터 역추적하는 과정을 나타냅니다. 역추적을 하는 이유는 매번 순간의 최선의 선택이 전체 최적을 보장하지 못할 수도 있기 때문입니다.


## **그림3** Viterbi Trellis
{: .no_toc .text-delta }
<img src="https://i.imgur.com/J40M4Oh.png" width="400px" title="source: imgur.com" />


## **그림4** Viterbi Trellis
{: .no_toc .text-delta }
<img src="https://i.imgur.com/vENtXE3.png" width="500px" title="source: imgur.com" />


## **그림5** Viterbi Backtrace
{: .no_toc .text-delta }
<img src="https://i.imgur.com/JXM6Yua.png" width="500px" title="source: imgur.com" />


그림6은 바이그램 단위 디코딩을 개념적으로 나타낸 것입니다.

## **그림6** Bigram Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/sMwdIJd.png" width="500px" title="source: imgur.com" />


---


## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

---