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
<img src="https://i.imgur.com/LRjrS68.png" width="550px" title="source: imgur.com" />

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

수식3에서 $N^T$는 $N$개의 범주를 가지는 음소 시퀀스의 길이가 $T$라는 의미입니다.

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
{ \alpha  }_{ 3 }\left( 4 \right) & = p\left( \text{"-he"} \right) +p\left( \text{"hhe"} \right) +p\left( \text{"h-e"} \right) +p\left( \text{"hee"} \right) \\
& = { y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 }\cdot { y }_{ e }^{ 3 }
\end{align*}
$$

$s=2$는 `h`, $s=3$은 `-`, $s=4$는 `e`입니다.


## **수식5** Dynamic Programming Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 4 \right) &=p\left( \text{"-he"} \right) +p\left( \text{"hhe"} \right) +p\left( \text{"h-e"} \right) +p\left( \text{"hee"} \right) \\ 
&={ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ e }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 }\cdot { y }_{ e }^{ 3 }\\ 
&=\left\{ \left( { y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 } \right) +\left( { y }_{ h }^{ 1 }\cdot { y }_{ - }^{ 2 } \right) +\left( { y }_{ h }^{ 1 }\cdot { y }_{ e }^{ 2 } \right)  \right\} { y }_{ e }^{ 3 }\\ 
&=\left\{ { \alpha  }_{ 2 }\left( 2 \right) +{ \alpha  }_{ 2 }\left( 3 \right) +{ \alpha  }_{ 2 }\left( 4 \right)  \right\} { y }_{ e }^{ 3 }
\end{align*}
$$


## **그림6** Case 1
{: .no_toc .text-delta }
<img src="https://i.imgur.com/LImeAHA.png" width="200px" title="source: imgur.com" />


## **그림7** Case 2
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TdNSUy7.png" width="200px" title="source: imgur.com" />


## **그림8** Architecture
{: .no_toc .text-delta }
<img src="https://i.imgur.com/1vbZ0gt.png" width="500px" title="source: imgur.com" />


## **수식6** Forward Probability (2)
{: .no_toc .text-delta }

$$
\require{ams}
\begin{equation*}
    { \alpha  }_{ t }\left( s \right) =
    \begin{cases}
      \left\{ { \alpha  }_{ t-1 }\left( s \right) +{ \alpha  }_{ t-1 }\left( s-1 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{if}\ { \mathbf{l}' }_{ s }=b \text{ or } { \mathbf{l}' }_{ s - 2 }={ \mathbf{l}' }_{ s }\\
      \left\{ { \alpha  }_{ t-1 }\left( s \right) +{ \alpha  }_{ t-1 }\left( s-1 \right) +{ \alpha  }_{ t-1 }\left( s-2 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{otherwise}
    \end{cases}
\end{equation*}
$$

---


## Backward Computation

## **수식7** Backward Probability (1)
{: .no_toc .text-delta }

$${ \beta  }_{ t }\left( s \right) =\sum _{ \pi \in { N }^{ T }: \cal{B} \left( { \pi  }_{ t:T } \right) ={ \mathbf{l} }_{ s:| \mathbf{l} | } }^{  }{ \prod _{ t'=t }^{ T }{ { y }_{ { \pi  }_{ t' } }^{ t' } }  } $$


## **그림9** Backward Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/nq67yUG.png" width="400px" title="source: imgur.com" />

## **수식8** Backward Probability (2)
{: .no_toc .text-delta }

$$
\require{ams}
\begin{equation*}
    \beta _{ t }\left( s \right) =
    \begin{cases}
      \left\{ { \beta  }_{ t+1 }\left( s \right) +\beta _{ t+1 }\left( s+1 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{if}\ { \mathbf{l}' }_{ s }=b \text{ or } { \mathbf{l}' }_{ s - 2 }={ \mathbf{l}' }_{ s }\\
      \left\{ { \beta  }_{ t+1 }\left( s \right) +{ \beta  }_{ t+1 }\left( s+1 \right) +\beta _{ t+1 }\left( s+2 \right)  \right\} { y }_{ { \mathbf{l}' }_{ s } }^{ t }, & \text{otherwise}
    \end{cases}
\end{equation*}
$$


---


## Complete Path Calculation

## **그림10** Complete Path Calculation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/ctsh5p9.png" width="400px" title="source: imgur.com" />

## **수식9** Forward Probability Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 2 \right) &=p\left( \text{"--h"} \right) +p\left( \text{"-hh"} \right) +p\left( \text{"hhh"} \right) \\ 
&={ y }_{ - }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ h }^{ 3 }+{ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }
\end{align*}
$$


## **수식10** Backward Probability Example
{: .no_toc .text-delta }

$$
\begin{align*}
\beta _{ 3 }\left( 2 \right) &=p\left( \text{"hel-lo"} \right) \\ 
&={ y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }
\end{align*}
$$



## **수식11** Forward/Backward Probability Example
{: .no_toc .text-delta }

$$
\begin{align*}
{ \alpha  }_{ 3 }\left( 2 \right) \cdot \beta _{ 3 }\left( 2 \right) =&{ y }_{ - }^{ 1 }\cdot { y }_{ - }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }\\
&+{ y }_{ - }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }\\
&+{ y }_{ h }^{ 1 }\cdot { y }_{ h }^{ 2 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ h }^{ 3 }\cdot { y }_{ e }^{ 4 }\cdot { y }_{ l }^{ 5 }\cdot { y }_{ - }^{ 6 }\cdot { y }_{ l }^{ 7 }\cdot { y }_{ o }^{ 8 }\\
=&\left\{ p\left( \text{"--hel-lo"} \right) +p\left( \text{"-hhel-lo"} \right) +p\left( \text{"hhhel-lo"} \right)  \right\} { y }_{ h }^{ 3 }\\ 
\end{align*}
$$


## **수식12** Complete Path Probability Example
{: .no_toc .text-delta }

$$p\left( \text{"--hel-lo"} \right) +p\left( \text{"-hhel-lo"} \right) +p\left( \text{"hhhel-lo"} \right) =\frac { { \alpha  }_{ 3 }\left( 2 \right) \cdot \beta _{ 3 }\left( 2 \right)  }{ { y }_{ h }^{ 3 } }$$


---

## Likelihood Computation


수식12에서 구한 계산 결과는 $t=3$ 시점에 상태 $h$가 관측될 확률입니다. 그렇다면 상태에 관계없이 $t=3$ 시점에 $p(\text{hello})$, 즉 우도는 얼마일까요? 수식13과 같습니다. 수식12에서는 given $\mathbf{x}$ 생략.


## **수식13** Likelihood Computation
{: .no_toc .text-delta }

$$p\left( \mathbf{l} | \mathbf{x} \right) =\sum _{ s=1 }^{ | \mathbf{l}' | }{ \frac { { \alpha  }_{ t }\left( s \right) \cdot \beta _{ t }\left( s \right)  }{ { y }_{ { \mathbf{l}' }_{ s } }^{ t } }  } $$


## **그림11** Likelihood Computation Example
{: .no_toc .text-delta }
<img src="https://i.imgur.com/IYSiNlx.png" width="400px" title="source: imgur.com" />


## **수식14** Likelihood Computation Example
{: .no_toc .text-delta }

$$
\begin{align*}
p\left( \text{"hello"} | x \right) =&\frac { { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right)  }{ { y }_{ h }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } }\\ 
&+\frac { { \alpha  }_{ 3 }\left( e \right) \cdot \beta _{ 3 }\left( e \right)  }{ { y }_{ e }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( l \right) \cdot \beta _{ 3 }\left( l \right)  }{ { y }_{ l }^{ 3 } }
\end{align*}
$$


---


## Gradient Computation

우리는 우도, 즉 $p(\mathbf{l}\|\mathbf{x})$를 최대화하는 파라메터를 찾고자 합니다(Maximum Likelihood Estimation). 


## **수식15** Gradient Computation (1)
{: .no_toc .text-delta }

$$\frac { \partial \ln { \left( p\left( \mathbf{l} | \mathbf{x} \right)  \right)  }  }{ \partial { y }_{ k }^{ t } } =\frac { 1 }{ p\left( \mathbf{l} | \mathbf{x} \right)  } \frac { \partial p\left( \mathbf{l}| \mathbf{x} \right)  }{ \partial { y }_{ k }^{ t } } $$


## **수식16** Gradient Computation (2)
{: .no_toc .text-delta }

$$\frac { \partial p\left( \mathbf{l} | \mathbf{x} \right)  }{ \partial { y }_{ k }^{ t } } =\frac { 1 }{ { { y }_{ k }^{ t } }^{ 2 } } \sum _{ s\in \text{lab} \left( \mathbf{l},k \right)  }^{  }{ { \alpha  }_{ t }\left( s \right) \cdot \beta _{ t }\left( s \right)  }$$


## **그림12** Gradient Computation Example (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Zhscxzw.png" width="400px" title="source: imgur.com" />


## **수식17** Gradient Computation Example (1)
{: .no_toc .text-delta }

$$
\begin{align*}

p\left( "hello"|x \right) =&\frac { { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right)  }{ { y }_{ h }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } }\\
&+\frac { { \alpha  }_{ 3 }\left( e \right) \cdot \beta _{ 3 }\left( e \right)  }{ { y }_{ e }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( - \right) \cdot \beta _{ 3 }\left( - \right)  }{ { y }_{ - }^{ 3 } } +\frac { { \alpha  }_{ 3 }\left( l \right) \cdot \beta _{ 3 }\left( l \right)  }{ { y }_{ l }^{ 3 } }
\end{align*}
$$

## **수식18** Gradient Computation Example (1)
{: .no_toc .text-delta }

$$\frac { \partial p\left( \text{"hello"} | \mathbf{x} \right)  }{ \partial { y }_{ h }^{ 3 } } =\frac { 1 }{ { { y }_{ h }^{ 3 } }^{ 2 } } { \alpha  }_{ 3 }\left( h \right) \cdot \beta _{ 3 }\left( h \right) $$


## **그림13** Gradient Computation Example (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HgVRXpa.png" width="400px" title="source: imgur.com" />

## **수식19** Gradient Computation Example (2)
{: .no_toc .text-delta }

$$p\left( \text{"hello"} | \mathbf{x} \right) =\frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } } +\frac { { \alpha  }_{ 5 }\left( - \right) \cdot \beta _{ 5 }\left( - \right)  }{ { y }_{ - }^{ 5 } } +\frac { { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  }{ { y }_{ l }^{ 5 } } $$

## **수식20** Gradient Computation Example (2)
{: .no_toc .text-delta }

$$\frac { \partial p\left( \text{"hello"} | \mathbf{x} \right)  }{ \partial { y }_{ l }^{ 5 } } =\frac { 1 }{ { { y }_{ l }^{ 5 } }^{ 2 } } \left\{ { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right) + { \alpha  }_{ 5 }\left( l \right) \cdot \beta _{ 5 }\left( l \right)  \right\} $$

---


## Normalization

---


## Decoding


## **그림14** Best Path Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TWnm2Vi.png" title="source: imgur.com" />


## **그림15** Prefix Decoding
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bjbfVAV.png" width="200px" title="source: imgur.com" />


## **그림16** Beam Search (1)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/HTIszHk.png" width="500px" title="source: imgur.com" />


## **그림17** Beam Search (2)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/bB1pi0h.png" width="600px" title="source: imgur.com" />


## **그림18** Beam Search (3)
{: .no_toc .text-delta }
<img src="https://i.imgur.com/TGba0fl.png" width="600px" title="source: imgur.com" />


---


## Properties

## **그림19** Conditional Indepedence
{: .no_toc .text-delta }
<img src="https://i.imgur.com/leR0ren.png" width="150px" title="source: imgur.com" />


## **그림20** Activations
{: .no_toc .text-delta }
<img src="https://i.imgur.com/jQwZHyh.png" title="source: imgur.com" />

---


## References

- [모두의 연구소, 음성인식 풀잎스쿨]()
- [Connectionist Temporal Classification, Labelling Unsegmented Sequence Data with RNN](https://www.youtube.com/watch?v=UMxvZ9qHwJs&t=2379s)
- [Sequence Modeling
With CTC](https://distill.pub/2017/ctc/)

---