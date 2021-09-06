# Attention-OCR

This repository is my simple re-implement of AttentionOCR in Tensorflow 2.0

Original paper: https://arxiv.org/pdf/1704.03549.pdf and it's official code can be found at Tensorflow models repo

One more paper this repo based on: https://arxiv.org/abs/1904.01906 and it's code: https://github.com/clovaai/deep-text-recognition-benchmark

Different from these:

- No Spartial Tranformer Network (STN)
- No Position encoder
- Add 1 Bilstm layer between CNN and lstm output layer
- Using EfficientNetB0 as backbone
