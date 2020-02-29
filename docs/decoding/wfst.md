---
layout: default
title: Weighted Finite-State Transducers
nav_order: 5
parent: Decoding strategy
permalink: /docs/decoding/wfst
---

# Weighted Finite-State Transducers
{: .no_toc }

Weighted Finite-State Transducers를 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## Concept

Weighted Finite-State Transducers를 개념적으로 나타낸 그림은 그림1과 같습니다.


## **그림1** Motivation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/kN1n00y.png" width="500px" title="source: imgur.com" />

## **수식1** WFSA, WFST
{: .no_toc .text-delta }
<img src="https://i.imgur.com/wrgf0jz.jpg" width="500px" title="source: imgur.com" />

---

## Operations

### Composition


## **수식2** Composition
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1rWPyy8.jpg" width="500px" title="source: imgur.com" />


## **그림2** Composition (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/qvxcvSG.jpg" width="500px" title="source: imgur.com" />

## **그림3** Composition (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ZHc8WWO.jpg" width="500px" title="source: imgur.com" />

## **그림4** Composition (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/T9DhSKf.jpg" width="500px" title="source: imgur.com" />

## **그림5** Composition (4)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/cEHRv8H.png" width="500px" title="source: imgur.com" />


### Determinization

## **그림6** Determinization (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/DmuwcLA.png" width="500px" title="source: imgur.com" />

## **그림7** Determinization (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TaYdafG.jpg" width="500px" title="source: imgur.com" />


### Minimization

## **그림8** Minimization (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/xsLi1xN.png" width="500px" title="source: imgur.com" />


## **그림9** Minimization (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/DmuwcLA.png" width="500px" title="source: imgur.com" />



---

## WFST with ASR


## **표1** Transducers
{: .no_toc .text-delta }
<img src="https://i.imgur.com/O0e5AxA.jpg" width="500px" title="source: imgur.com" />

## **그림10** Concept (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/pO3AZle.jpg" width="500px" title="source: imgur.com" />

## **그림11** Concept (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/3ZhRfss.jpg" width="500px" title="source: imgur.com" />

## **수식3** H ◦ C ◦ L ◦ G composition
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Tgatro3.jpg" width="500px" title="source: imgur.com" />



### Grammar Transducer

## **그림11** Grammar Transducer (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/EbQtNZ5.jpg" width="500px" title="source: imgur.com" />

## **그림12** Grammar Transducer (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/GTBuFJo.jpg" width="500px" title="source: imgur.com" />


### Lexicon Transducer

## **그림13** Lexicon Transducer (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/gK3Fbml.jpg" width="500px" title="source: imgur.com" />

## **그림14** Lexicon Transducer (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/x6lOu8L.jpg" width="500px" title="source: imgur.com" />


### L ◦ G composition

## **그림15** L ◦ G composition
{: .no_toc .text-delta }
<img src="https://i.imgur.com/eokYfcL.jpg" title="source: imgur.com" />


### Context-dependent Transducer

## **그림16** Context-dependent Transducer (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/VP5ScQ1.jpg" width="500px" title="source: imgur.com" />

## **그림17** Context-dependent Transducer (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/6s1WbQm.png" width="500px" title="source: imgur.com" />


### HMM Transducer

## **그림18** HMM Transducer
{: .no_toc .text-delta }
<img src="https://i.imgur.com/xL4xxwM.jpg" width="500px" title="source: imgur.com" />


---


## Decoding


---

## References

- [Speech and Language Processing 3rd edition draft](https://web.stanford.edu/~jurafsky/slp3/9.pdf)
- [Speech Recognition — Weighted Finite-State Transducers (WFST)](https://medium.com/@jonathan_hui/speech-recognition-weighted-finite-state-transducers-wfst-a4ece08a89b7)

---