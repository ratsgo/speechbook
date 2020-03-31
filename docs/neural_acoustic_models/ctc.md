---
layout: default
title: Connectionist Temporal Classification
nav_order: 4
parent: Neural Acoustic Models
permalink: /docs/neuralam/ctc
---

# Connectionist Temporal Classification
{: .no_toc }

입력 음성 프레임 시퀀스와 타겟 단어/음소 시퀀스 간에 명시적인 얼라인먼트(alignment) 정보 없이도 음성 인식 모델을 학습할 수 있는 기법인 Connectionist Temporal Classification(CTC)를 살펴봅니다.
{: .fs-4 .ls-1 .code-example }

## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}


---

## Motivation

## **그림1** Motivation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LRjrS68.png" width="400px" title="source: imgur.com" />

## **그림2** Motivation
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nxoVdg5.png" width="400px" title="source: imgur.com" />



---


## All Possible Paths

## **그림3** All Possible Paths
{: .no_toc .text-delta }
<img src="https://i.imgur.com/tRNSz3q.png" width="400px" title="source: imgur.com" />

## **그림4** All Possible Paths
{: .no_toc .text-delta }
<img src="https://i.imgur.com/W6RDQxj.png" width="400px" title="source: imgur.com" />


## **수식1** Path Probability
{: .no_toc .text-delta }

$$p\left( \pi |\mathbf{x} \right) =\prod _{ t=1 }^{ T }{ { y }_{ { \pi  }_{ t } }^{ t } }$$


## **수식2** All Paths Probability
{: .no_toc .text-delta }

$$p( \mathbf{l} | \mathbf{x} )=\sum _{ \pi \in { \cal{B}  }^{ -1 } \left( \mathbf{l} \right) }^{  }{ p\left( \pi | \mathbf{x} \right)  } $$


---

## Forward Computation

## **수식3** Forward Probability (1)
{: .no_toc .text-delta }

$${ \alpha  }_{ t }\left( s \right) =\sum _{ \pi \in { N }^{ T }: \cal{B} \left( { \pi  }_{ 1:t } \right) ={ \mathbf{l} }_{ 1:s } }^{  }{ \prod _{ t'=1 }^{ t }{ { y }_{ { \pi  }_{ t' } }^{ t' } }  }$$


## **그림5** Forward Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/dpfr6Oi.png" width="400px" title="source: imgur.com" />


## **수식4** Forward Computation Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 4 \right) & = p\left( "-he" \right) +p\left( "hhe" \right) +p\left( "h-e" \right) +p\left( "hee" \right) \\ 
& = { y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 }\cdot { y }_{ e }^{ 3 }
\end{align*}
$$


## **수식5** Dynamic Programming Example
{: .no_toc .text-delta }

$${ \alpha  }_{ 3 }\left( 4 \right) =p\left( "-he" \right) +p\left( "hhe" \right) +p\left( "h-e" \right) +p\left( "hee" \right) \\ ={ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 }\cdot { y }_{ e }^{ 3 }\\ =\left( { y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }+{ y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 } \right) { y }_{ e }^{ 3 }\\ =\left\{ { \alpha  }_{ 2 }\left( 3 \right) +{ \alpha  }_{ 2 }\left( 2 \right) +{ \alpha  }_{ 2 }\left( 4 \right)  \right\} { y }_{ e }^{ 3 }$$


## **그림6** Case 1
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LImeAHA.png" width="400px" title="source: imgur.com" />


## **그림7** Case 2
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TdNSUy7.png" width="400px" title="source: imgur.com" />


## **그림8** Architecture
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1vbZ0gt.png" width="400px" title="source: imgur.com" />


## **수식3** Forward Probability (2)
{: .no_toc .text-delta }

$${ \alpha  }_{ t }\left( s \right) =\left\{ { \alpha  }_{ t-1 }\left( s \right) +{ \alpha  }_{ t-1 }\left( s-1 \right)  \right\} { y }_{ { l' }_{ s } }^{ t }$$

$${ \alpha  }_{ t }\left( s \right) =\left\{ { \alpha  }_{ t-1 }\left( s \right) +{ \alpha  }_{ t-1 }\left( s-1 \right) +{ \alpha  }_{ t-1 }\left( s-2 \right)  \right\} { y }_{ { l' }_{ s } }^{ t }$$


---


## Backward Computation

## **수식3** Backward Probability (1)
{: .no_toc .text-delta }

$${ \beta  }_{ t }\left( s \right) =\sum _{ \pi \in { N }^{ T }:B\left( { \pi  }_{ t:T } \right) ={ l }_{ s:|l| } }^{  }{ \prod _{ t'=t }^{ T }{ { y }_{ { \pi  }_{ t' } }^{ t' } }  } $$

## **그림9** Backward Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nq67yUG.png" width="400px" title="source: imgur.com" />

## **수식3** Backward Probability (2)
{: .no_toc .text-delta }

$$\beta _{ t }\left( s \right) =\left\{ { \beta  }_{ t+1 }\left( s \right) +\beta _{ t+1 }\left( s+1 \right)  \right\} { y }_{ { l' }_{ s } }^{ t }\\ { \beta  }_{ t }\left( s \right) =\left\{ { \beta  }_{ t+1 }\left( s \right) +{ \beta  }_{ t+1 }\left( s+1 \right) +\beta _{ t+1 }\left( s+2 \right)  \right\} { y }_{ { l' }_{ s } }^{ t }$$


---


## Complete Path Calculation

## **그림10** Complete Path Calculation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ctsh5p9.png" width="400px" title="source: imgur.com" />

## **수식3** Forward Probability Example
{: .no_toc .text-delta }

$${ \alpha  }_{ 3 }\left( 2 \right) =p\left( "--h" \right) +p\left( "-hh" \right) +p\left( "hhh" \right) \\ ={ y }_{ - }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ h }^{ 3 }+{ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }$$


## **수식4** Backward Probability Example
{: .no_toc .text-delta }

$$\beta _{ 3 }\left( 2 \right) =p\left( "hel-lo" \right) \\ ={ y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }$$



## **수식4** Forward/Backward Probability Example
{: .no_toc .text-delta }

$${ \alpha  }_{ 3 }\left( 2 \right) \cdot \beta _{ 3 }\left( 2 \right) =\left\{ p\left( "--hel-lo" \right) +p\left( "-hhel-lo" \right) +p\left( "hhhel-lo" \right)  \right\} { y }_{ h }^{ 3 }\\ ={ y }_{ - }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }+{ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }$$


## **수식4** Complete Path Probability Example
{: .no_toc .text-delta }

$$p\left( "--hel-lo" \right) +p\left( "-hhel-lo" \right) +p\left( "hhhel-lo" \right) =\frac { { \alpha  }_{ 3 }\left( 2 \right) \cdot \beta _{ 3 }\left( 2 \right)  }{ { y }_{ h }^{ 3 } }$$


---


## Likelihood Computation


## **수식11** Likelihood Computation
{: .no_toc .text-delta }

$$p\left( l|x \right) =\sum _{ s=1 }^{ |l'| }{ \frac { { \alpha  }_{ t }\left( s \right) \cdot \beta _{ t }\left( s \right)  }{ { y }_{ { l' }_{ s } }^{ t } }  } $$


## **그림11** Likelihood Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/IYSiNlx.png" width="400px" title="source: imgur.com" />


## **수식11** Likelihood Computation Example
{: .no_toc .text-delta }

$$p\left( "hello"|x \right) =\frac { { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right)  }{ { y }_{ h }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( e \right) \cdot \beta _{ 3 }\left( e \right)  }{ { y }_{ e }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( l \right) \cdot \beta _{ 3 }\left( l \right)  }{ { y }_{ l }^{ 3 } } $$


---


## Gradient Computation


## **수식11** Gradient Computation (1)
{: .no_toc .text-delta }

$$\frac { \partial \ln { \left( p\left( l|x \right)  \right)  }  }{ \partial { y }_{ k }^{ t } } =\frac { 1 }{ p\left( l|x \right)  } \frac { \partial p\left( l|x \right)  }{ \partial { y }_{ k }^{ t } } $$


## **수식11** Gradient Computation (2)
{: .no_toc .text-delta }

$$\frac { \partial p\left( l|x \right)  }{ \partial { y }_{ k }^{ t } } =\frac { 1 }{ { { y }_{ k }^{ t } }^{ 2 } } \sum _{ s\in lab\left( l,k \right)  }^{  }{ { \alpha  }_{ t }\left( s \right) \cdot \beta _{ t }\left( s \right)  }$$


## **그림12** Gradient Computation Example (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Zhscxzw.png" width="400px" title="source: imgur.com" />


## **수식12** Gradient Computation Example (1)
{: .no_toc .text-delta }

$$p\left( "hello"|x \right) =\frac { { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right)  }{ { y }_{ h }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( e \right) \cdot \beta _{ 3 }\left( e \right)  }{ { y }_{ e }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( l \right) \cdot \beta _{ 3 }\left( l \right)  }{ { y }_{ l }^{ 3 } } $$

## **수식12** Gradient Computation Example (1)
{: .no_toc .text-delta }


$$\frac { \partial p\left( "hello"|x \right)  }{ \partial { y }_{ h }^{ 3 } } =\frac { 1 }{ { { y }_{ h }^{ 3 } }^{ 2 } } \frac { { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right)  }{ { y }_{ h }^{ 3 } } $$


## **그림13** Gradient Computation Example (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HgVRXpa.png" width="400px" title="source: imgur.com" />

## **수식12** Gradient Computation Example (2)
{: .no_toc .text-delta }

$$p\left( "hello"|x \right) =\frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } } +\frac { { \alpha  }_{ 5 }\left( - \right) \cdot \beta _{ 5 }\left( - \right)  }{ { y }_{ - }^{ 5 } } +\frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } } $$

## **수식12** Gradient Computation Example (2)
{: .no_toc .text-delta }

$$\frac { \partial p\left( "hello"|x \right)  }{ \partial { y }_{ l }^{ 5 } } =\frac { 1 }{ { { y }_{ l }^{ 5 } }^{ 2 } } \left( \frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } } +\frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } }  \right) $$

---


## Normalization

---


## Decoding


## **그림14** Best Path Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TWnm2Vi.png" width="400px" title="source: imgur.com" />


## **그림15** Prefix Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bjbfVAV.png" width="400px" title="source: imgur.com" />


## **그림16** Beam Search (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HTIszHk.png" width="400px" title="source: imgur.com" />


## **그림17** Beam Search (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bB1pi0h.png" width="400px" title="source: imgur.com" />


## **그림18** Beam Search (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TGba0fl.png" width="400px" title="source: imgur.com" />


---


## Properties

## **그림19** Conditional Indepedence
{: .no_toc .text-delta }
<img src="https://i.imgur.com/leR0ren.png" width="400px" title="source: imgur.com" />


## **그림20** Activations
{: .no_toc .text-delta }
<img src="https://i.imgur.com/jQwZHyh.png" title="source: imgur.com" />

---