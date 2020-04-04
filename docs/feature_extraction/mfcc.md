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
<audio class="audio" controls preload="none"><source src="https://github.com/ratsgo/speechbook/blob/master/docs/feature_extraction/example.wav?raw=true"></audio>

> transcript : 그래가지고 그거 연습해야 된다고 이제 빡씨게 모여야 되는데 내일 한 두 시나 네 시에 모여서 저녁 여덟시까지 교회에 있을 거 같애

## **코드1** Wave file 읽기
{: .no_toc .text-delta }
```python
import scipy.io.wavfile
sample_rate, signal = scipy.io.wavfile.read('example.wav')
```

```
>>> signal
array([36, 37, 60, ...,  7,  9,  8], dtype=int16)
>>> len(signal)
183280
>>> len(signal) / sample_rate
11.455
```


## **코드2** Preemphasis
{: .no_toc .text-delta }
```python
signal = signal[0:int(3.5 * sample_rate)]

import numpy as np
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
```

## **코드3** Framing
{: .no_toc .text-delta }
```python
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
```


## **코드4** Windowing
{: .no_toc .text-delta }
```python
frames *= np.hamming(frame_length)
# frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))
```


## **코드5** Fourer-Transform & Power Spectrum
{: .no_toc .text-delta }
```python
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
```

## **코드6** Filter Banks
{: .no_toc .text-delta }
```python
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
```

## **코드6** Log-Filter Banks
{: .no_toc .text-delta }
```python
filter_banks = 20 * np.log10(filter_banks)  # dB
```

## **코드8** Mel-frequency Cepstral Coefficients
{: .no_toc .text-delta }
```python
from scipy.fftpack import dct
num_ceps = 12
cep_lifter = 22
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift
```

## **코드9** Mean Normalization
{: .no_toc .text-delta }
```python
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
```

---

## References

- [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
- [김형석, Mel-Frequency Cepstral Coefficients, 모두의연구소 음성인식 부트캠프, 2020. 2. 1.](https://home.modulabs.co.kr/product/%ec%9d%8c%ec%84%b1-%ec%9d%b8%ec%8b%9d-%eb%b6%80%ed%8a%b8%ec%ba%a0%ed%94%84)

---