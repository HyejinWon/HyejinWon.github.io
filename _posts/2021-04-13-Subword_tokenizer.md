---
published: true
layout: single
title: "Subword Tokenizer"
category: post
tags: [NLP, Transformer, tokenizer, GPT]
use_math: true
---

### Tokenizer

문장을 보다 작은 단위로 분절하기 위한 방법으로, 공백 (띄어쓰기)를 사용하는 나라에서는 공백을 기준으로 분절하기도 한다. 하지만 중국어, 일본어와 같은 경우에는 공백으로 token의 단위를 분절하기 어렵기 때문에 subword tokenizer의 필요성이 대두 되었다.

### subword tokenizer

'빈도가 잦은 단어는 유지하고, 저빈도의 단어는 의미있는 단위로 분절한다'라는 원칙을 기반으로 한다. 특히 교착어에서 유용하다. 

ex. '학교' - '학교를', '학교에', '학교가' ...

위의 예에서 '학교'를 조사와 분리하게 되면 vocab의 size는 엄청나게 줄어들 수 있게 된다.

### BPE(byte pair encoing)

BPE는 말그대로 자주 등장하는 byte를 특정 byte로 치환하는 방법이다. 

일반적으로 훈련 데이터를 단어 단위로 분절하는 **pre-tokenize** 과정을 거쳐야 한다. pre-tokenize는 단어 단위로 나누는 방법도 있지만, 일련의 규칙을 기반으로 나눌 수도 있다.

1. pre-tokenize to word , 'a apple is good'
2. word to char, 'a':10, 'a p p l e':15, 'i s':4, 'g o o d':2
3. find **maximum** byte pair
4. add dictionary
5. merge char to pair

사전의 크기는 일반적으로 `기본 단어 개수 + 합쳐진 서브워드의 개수` 가 되고, 사용자가 서브워드의 개수를 지정할 수 있음. gpt에서는 40,478개로 vocab을 설정했고, 이는 478은 base char이고, 40,000만큼 merge되면 학습을 멈춘다.

### Byte-level BPE

GPT-2에서 사용하는 tokenizer로 byte를 기반으로 한다. 이것은 모든 기본 문자가 어휘에 포함되도록 하면서 기본 어휘의 크기가 256이 되도록 강요하는 영리한 방법이다. (1byte는 256개의 고유한 값으로 구성된다)  문장기호를 위해서는 몇 가지의 규칙이 필요하지만, gpt-2 tokenizer는 <unk> 를 만들어내지 않는다. gpt-2의 vocab은 50,257이며, 256byte 는 기본 token이고, end-of-text 토큰과 symbols가 50,000로 구성된다. 

특이한 점은, 공백도 토큰의 일부로 판단한다는 것인데, 아래의 코드를 보자

```python
>>> from transformers import GPT2Tokenizer
>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> tokenizer("Hello world")['input_ids']
[15496, 995]
>>> tokenizer(" Hello world")['input_ids']
[18435, 995]
```

### WorldPiece

wordpiece는 BERT에 활용된 subword tokenizer이다. 이 역시 BPE와 같이 **pre-tokenize**를 진행해야 한다. 그리고, word를 char로 변환한다.  또한, 사용자가 지정한 subword개수만큼 char를 합치는 과정을 진행한다.  BPE와 다른점은 가장 많이 나온 pair를 합치는 것이 아니라, 병합하였을 때 **코퍼스의 Likelihood**를 가장 높이는 쌍과 병합하게 된다.

즉, wordpiece는 병합 후보에 오른 쌍을 미리 병합해보고, 잃게 되는 것은 무엇인지, 쌍을 병합할 가치가 충분한지 등을 판단한 후에 병합을 수행한다.

### Unigram

BPE, WordPiece와 같이 기본 char에서 subword를 점진적으로 병합해 나아가는 것이 아니라, 모든 **Pre-tokenized** 토큰과 서브 워드에서 시작해 점차 사전을 줄여나가는 방식으로 진행된다.

매 step마다 unigram은 주어진 코퍼스와 현재 사전에 대한 loss값을 측정한다. 이후 각각의 subword에 대해 해당 subword가 corpus에서 제거 되었을 때, loss가 얼마나 증가하는지를 측정한다. 이때, loss를 가장 적게 증가시키는 p개의 토큰을 제거한다. (일반적으로 p는 전체 사전 크기의 10-20%) unigram은 해당 과정을 사용자가 지정한 사전의 크기를 지니게 될 때까지 반복하게 된다. 이때, 기존 char는 삭제되지 않고 유지된다. 

만약 'goods'라는 단어에서 어떻게 분절할 것이냐?라고 한다면, unigram의 사전은 token의 확률값도 지니고 있음으로, token의 확률값이 가장 큰 것을 선택하여 단어를 분절하게 된다.

### SentencePiece

위의 subword tokenizer는 pre-tokenize가 base로 진행되고 알고리즘들이 작동되는 방식이다. 모든 언어가 다 공백을 가지고 있는 것은 아니기때문에 이를 해결하고자 나타난 알고리즘이다. (pre-tokenize X)

문장에 공백이 존재한다면 “▁” (U+2581) 토큰으로 대치시켜준다. 이로써 문장을 word의 집합으로 보지 않게 되고, 문장에 포함된 공백을 포함하여 BPE와 unigram을 적용하여 사전을 구축하게 된다.

sentencepiece는 ALBERT, XLNet 등에 사용되어 훈련되었다.

ex. Hello World → Hello▁World

---

Reference

[https://huffon.github.io/2020/07/05/tokenizers/](https://huffon.github.io/2020/07/05/tokenizers/)

[https://github.com/google/sentencepiece#comparisons-with-other-implementations](https://github.com/google/sentencepiece#comparisons-with-other-implementations)

[https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer)
