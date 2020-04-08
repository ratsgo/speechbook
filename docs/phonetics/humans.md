---
layout: default
title: Recognition by Humans
nav_order: 3
parent: Phonetics
permalink: /docs/phonetics/humans
---

# 인간의 음성 인식
{: .no_toc }

본 블로그는 자동 음성 인식(Automatic Speech Recognition)을 위한 기법들을 정리한 것입니다. 다시 말해 컴퓨터로 하여금 사람 말 소리를 알아듣게 하려는 것인데요. 그러면 당연히 사람의 음성 인식에 대해 알아볼 필요가 있습니다. 이 글에서는 사람의 말소리 인식 능력을 알아보고자 합니다. 이 글은 [Speech and Language Processing 2nd Edition](https://www.amazon.com/Speech-Language-Processing-Daniel-Jurafsky/dp/0131873210)을 정리한 것입니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---


## lexical access

사람은 음성을 단어 단위로 인식한다고 합니다(**lexical access**). 말소리를 단어 단위로 인식하는 이러한 특성은 자동 음성 인식(Automatic Speech Recognition, ASR)에도 도입되었습니다. 인간의 단어 단위 말소리 인식은 구체적으로 다음 세 가지 특성으로 나눠 생각해볼 수 있습니다.

1. **frequency** : 사람은 빈도 높은 단어를 빠르게 인식합니다. 고빈도 단어는 노이즈가 심한 환경에서도, 작은 말소리로도 정확하게 알아차릴 수 있습니다. 
2. **parallelism** : 여러 단어(예컨대 두 명 이상의 화자가 발화)를 한번에 알아들을 수 있습니다. 
3. **cue-based processing** : 인간의 음성 인식은 다양한 단서(cue)에 기반합니다. 다음 챕터에서 차례로 살펴보겠습니다.

---

## cue-based processing

사람이 말소리를 알아듣기 위해 사용하는 단서 가운데 하나는 **음성적 특징(acoustic cues)**입니다. 포만트(formant)나 성대진동 개시 시간(voice onset time) 등이 여기에 해당합니다. **포만트**란 [스펙트럼](https://ratsgo.github.io/speechbook/docs/fe/mfcc#framework)에서 음향 에너지가 몰려 있는 봉우리를 가리킵니다. 포만트가 어떤 주파수 영역대에서 형성되어 있는지에 따라 사람은 말소리를 다르게 인식합니다. **성대진동 개시 시간**은 무성폐쇄음(예: `ㅍ`)의 개방 단계 후에 후행하는 모음을 위해 성대가 진동하는 시간 사이의 기간을 의미합니다. 성대 진동 개시 시간은 말소리에서 유성자음(예: `ㅂ`)과 무성자음을 식별하는 중요한 단서라고 합니다.

어휘 그 자체도 말소리 인식에 중요한 단서가 될 수 있습니다(**lexical cues**). 이와 관련해 Warren이라는 학자는 1970년 **음소 복원 현상(Phonemic restoration effect)**라는 개념을 제시했습니다. 예컨대 단어를 이루는 음소(phoneme) 가운데 하나를 기침 소리로 대체하더라도 해당 음소를 들은 것으로 인식한다는 것입니다. 이는 청자가 해당 어휘를 이미 알고 있는 덕분입니다.

입모양 같은 **시각적 단서(visual cues)** 역시 상당한 영향을 미칠 수 있습니다. 발견자의 이름을 딴 **맥거크 효과(McGurk effect)**라는 것이 있는데요. 입모양 또는 기타 다른 감각 정보의 영향으로 실제와는 다른 소리로 지각되는 현상을 가리킵니다. 예컨대 `ga`라는 음절(syllable)을 발음하는 영상을 보여주면서도 `ba`라는 소리를 들려주면 `da`라고 알아듣는 식입니다.

인간의 말소리 인식은 최근 들었던 단어에 영향을 받기도 합니다. 의미론적 단어 연상(semantic word association)이나 반복 점화(repetition priming) 등이 바로 그것입니다. 의미론적 단어 연상은 사람이 최근에 들었던 단어 가운데 의미상 유사한 단어를 더 빨리 알아듣는 현상을 의미합니다. 반복 점화는 어떤 자극이 반복돼 해당 자극의 이후 경험이 뇌에서 빨리 처리되는 걸 가리킵니다. 이전에 어떤 단어를 들었는데 그것이 반복된다면 더 빨리 알아채는 식입니다. 이와 관련해서는 자연어 처리 연구자들이 **cashe language model**이라는 개념으로 모델링한 바 있습니다.


---

## on-line processing

인간의 말소리 인식은 그때그때 실시간으로 진행됩니다(**on-line processing**). Marslen-Wilson의 1973년 연구에 따르면 사람은 다른 사람의 말을 듣고서 250ms 내에 바로 따로 말할 수 있는 것으로 나타났습니다. 다시 말해 단어 세그먼트(word segmentation), 구문 분석(parsing), 그리고 해당 문장에 대한 해석(interpretation)에 이르기까지 전 과정을 250ms 안에 처리한다는 이야기입니다. 기계가 따라하기 어려울 정도로 대단한 능력입니다.


---


## References

- [Speech and Language Processing 2nd Edition](https://www.amazon.com/Speech-Language-Processing-Daniel-Jurafsky/dp/0131873210)

---