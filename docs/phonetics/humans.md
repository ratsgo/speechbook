---
layout: default
title: Recognition by Humans
nav_order: 3
parent: Phonetics
permalink: /docs/phonetics/humans
---

# Speech Recognition by Humans

사람과 기계의 음성 인식을 비교 검토합니다.



- 당연한 말이지만 사람은 기계보다 음성 인식을 잘 한다.
    - (그 당시) clean speech에 대해 에러률이 5배가 차이 나고, noisy speech에 대해선 차이가 더 심해진다.
- ASR은 사람 음성 인식과 몇가지 특성을 공유한다. (몇가지는 영감을 받았다.)
    - lexical access (the process of retrieving a word from the mental lexicon)
        - frequency
            - 사람은 빈도 높은 단어에 더 빨리 접근한다.
            - 빈도 높은 단어는 노이즈가 심해도 더 잘 알아듣는다.
            - 빈도 높은 단어는 자극이 크지 않아도 알아 듣는다.
        - parallelism
            - multiple words are active at the same time
        - cue-based processing
            - 음성에 대한 사람의 인식은 여러가지 cue에 영향을 받는다.
            - acoustic cues
                - formant structure or voicing timing
            - visual cues
                - lip movement
            - lexical cues
                - phoneme restoration effect.
                    - 단어를 이루는 phone 중에 하나를 기침 소리로 바꿔도 그 phone을 들은 줄 안다.
            - McGurk effect
                - visual input이 phone perception을 방해할 수 있다.
                - ga를 말하는 영상을 보여주며 ba 소리를 들려주면 da를 들었는 줄 안다.
            - 추가 cue들
                - semantic word association
                    - words are accessed more quickly if a semantically related word has been heard recently
                - repetition priming
                    - words are accessed more quickly if they themselves have just been heard
                - → 여러 연구자가 cache language model로 모델링 했다.
- 인간과 기계의 차이
    - on-line processing
        - ASR 모델이 전체 발화를 다 듣고 처리하는 데 반해
        - 사람은 즉각 처리한다.
        - close shadowers
            - 사람은 듣고서 250ms 내에 바로 따라 말할 수 있다.
            - 즉, 사람은 word segmentation, parsing, and interpretation을 250ms 안에 처리한다.
    - neighborhood effects
        - the neighborhood of a word is the set of words that closely resemble it
        - large frequency-weighted neighborhoods are accessed more slowly than words with fewer neighbor
    - prosodic knowledge를 써서 단어 인식하기