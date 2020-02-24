---
layout: default
title: Multipass Decoding
nav_order: 3
parent: Decoding strategy
permalink: /docs/decoding/multipass
---

# Multipass Decoding
{: .no_toc }

1차를 대강 탐색하고 2차를 자세히 탐색해 디코딩 품질을 높이는 Multipass Decoding 기법을 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## N-best decoding

N-best decoding은 Multipass Decoding의 일종입니다. 그 컨셉은 그림1과 같습니다.


## **그림1** N-best decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/gcZtcTb.png" width="400px" title="source: imgur.com" />

## **그림2** N-best decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/YSOp36B.png" width="600px" title="source: imgur.com" />

---

## Word Lattice

Word Lattice는 N-best decoding의 단점을 극복한 기법입니다. 그 컨셉은 그림3과 같습니다.

## **그림3** Word Lattice
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LZ1Tnly.png" width="600px" title="source: imgur.com" />


---

## Word Graph / Finite-state Machine


## **그림4** Word Graph
{: .no_toc .text-delta }
<img src="https://i.imgur.com/AjirX9e.png" width="600px" title="source: imgur.com" />


---

## Confusion Network


## **그림5** Confusion Network
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pLlRqVv.png" width="600px" title="source: imgur.com" />


---

## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

---