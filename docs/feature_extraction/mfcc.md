---
layout: default
title: MFCCs
nav_order: 2
parent: Feature Extraction
permalink: /docs/fe/mfcc
---

# Mel-Frequency Cepstral Coefficients
{: .no_toc }

MFCC 피처를 추출하는 방법을 알아봅니다.
{: .fs-4 .ls-1 .code-example }


## Table of contents
{: .no_toc .text-delta .mt-6}

1. TOC
{:toc}

---

## codes

## **그림1** framework
{: .no_toc .text-delta }
<img src="https://i.imgur.com/Pn5LGTk.png" width="600px" title="source: imgur.com" />

## **음성1** 예시 음성
{: .no_toc .text-delta }
<audio class="audio" controls preload="none"><source src="https://github.com/ratsgo/speechbook/blob/master/docs/phonetics/a_p1.wav?raw=true"></audio>

## **코드1** 예시 코드
{: .no_toc .text-delta }
```python
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct

sample_rate, signal = scipy.io.wavfile.read('/Users/david/Downloads/OSR_us_000_0010_8k.wav')

signal = signal[0:int(3.5 * sample_rate)]

pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))

num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))

pad_signal = np.append(emphasized_signal, z)

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

frames = pad_signal[indices.astype(np.int32, copy=False)]

frames *= np.hamming(frame_length)

NFFT = 512

mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate
```

---

## References

- [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
- [김형석, Mel-Frequency Cepstral Coefficients, 모두의연구소 음성인식 부트캠프, 2020. 2. 1.](https://home.modulabs.co.kr/product/%ec%9d%8c%ec%84%b1-%ec%9d%b8%ec%8b%9d-%eb%b6%80%ed%8a%b8%ec%ba%a0%ed%94%84)

---