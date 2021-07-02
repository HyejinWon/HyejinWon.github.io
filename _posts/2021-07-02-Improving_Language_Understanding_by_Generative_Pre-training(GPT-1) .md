# Improving Language Understanding by Generative Pre-training(GPT-1)

---

### Abstract

Unlabeled data는 충분히 많지만, 특정 task를 학습하기 위한 labeled data의 양은 그에 못 미친다. 논문에서는 generative pre-training LM을 방대한 양의 unlabeled data로 학습하고, discriminative fine-tuning을 통해 특정 task에 fine-tuning하는 것이 효과있음을 주장한다.

### Introduction

supervised learning에 있어서 raw data를 효율적으로 사용하는 것이 labeled data 에 대한 의존도를 낮추는 방법이다. 대부분의 딥러닝 방법은 어느정도의 labeled data를 요구하며, labeled data의 부족(결핍)은 여러 도메인에 있어서 그들의 모델 능력을 제한하게 한다. 

unlabeled data에서 좋은 표현 능력을 학습하는 것도 분명히 성능을 올릴 수 있다. 그에 대한 증명은 다양한 nlp task에서 모델에 사용되는 임베딩을 unlabeled data로 학습한 벡터를 사용하는 것으로 증명된다. 

unlabeled data에서 word-level 정보를 이끌어내는 것은 2가지 이유때문에 어렵다.

1. transfer하기에 적합한 text representation을 학습하는데 있어 optimization objective가 불분명하다. 최근에 language model, machine translation, discourse coherence같은 다양한 objective가 연구되어왔다.
2. 학습된 representation이 target task로 트랜스퍼하기에 효과적인 방법에 대한 의견일치가 되지 않았다. 그래서 모델 아키텍처를 테스크에 맞게 수정하거나, 복잡한 learning scheme을 거치거나, 추가적인 learning objective를 더하는 방법들이 존재하였다. 이러한 불확실성은 semi-supervised learning을 어렵게 한다.

논문에서는 language understanding task에 있어서 unsupervised pre-training 과 supervised fine-tuning을 결합한 **semi-supervised** 접근을 탐구한다. 

또한, target task와 같은 도메인의 unlabeled data를 필요로 하지 않는 방법이다.

학습하는데는 2가지 방법을 거친다.

1. unlabeled data로 모델을 pre-training 한다.
2. 이 모델을 기반으로 target task를 fine-tuning 한다.

모델은 Transformer를 기반으로 한다. Transformer는 기존 RNN계열에 비해 Long-term dependencies을 잘 다룰수 있고, 결과적으로 다양한 task에 대해  강건한 transfer성능을 보인다. transfer에서 task에 적합한 input sequence를 입력으로 주었다. 실험에서 증명한 바와 같이 이러한 적용은 fine-tuning에서 작은 변화만으로도 좋은 성능을 내도록 해줬다.

### Related work

**semi-supervised learning for NLP**

이전 연구들은 unlabeled data를 단어 레벨, 구 레벨로 하여 사용하였다. 몇년 전에는 연구자들은 unlabeled data로 단어 임베딩을 만들어 냈고, 이 방법은 여러 task에 좋은 성능을 보였다. 그러나 이런 연구들은 '단어'레벨에 국한되었다. 최근의 연구는 단어레벨보다 더 나아가는 방법인 phrase나  문장 레벨 임베딩을 사용한다.

**Unsupervised pre-training**

Unsupervised pre-training의 목적은 최적의 초기값을 찾는 것이다.  최근 연구는 이미지 분류, 음서이인식, 엔티티 분류, 기계번역에서 많이 사용된다. GPT와 유사한 연구로 언어모델을 pre-training하고, target task에 맞게 fine-tuning하는 것을 포함한다. 앞선 연구자들은 pre-training phase가 언어적 정보를 포착하는데 도움이 된다고 하지만, 그들은 LSTM모델을 사용했고, LSTM모델은 짧은 길이만 사용가능하다는 제약이 있다. 이와 반대로, transformer모델을 사용하게 되면 긴 시퀀스도 가능하다. 

또다른 연구에서는 supervised learning에서 pre-training 언어모델로 부터 얻은 hidden representation을 부가정보로 사용한다. 이 방법은 각 task마다 별도로 많은 양의 파라미터를 포함한다. 그러나 GPT는 아주 작은 변화로도 transfer learning이 가능하다.

**Auxiliary training objectives**

보조 학습 목적함수를 사용하는 것은 semi-supervised learning의 대안이다. 이전 연구에서는 semantic role labeling의 성능을 올리기 위해서 품사태깅, 청킨, 개체명인식, 그리고 언어모델링을 부가로 사용했다. GPT 역시 보조 목적함수를 사용한다. 그러나 unsupervised pre-training이 이미 몇몇의 target task 정보에 대하여 학습했다는 것을 보인다.

### Framework

1. 큰 말뭉치에 대해서 high capacity LM를 가지는 모델을 만드는 것
2. Label data에 대해서도 적용하는것

**Unsupervised pre-training**

목적 함수는 SGD로 학습되었고, 식은 다음과 같다. k는 context window크기, P는 조건부 확률, $\theta$는 파라미터이다.

$$L_1(U) = \sum_i logP(u_i|u_{i-k}, ...,u_{i-1};\theta)$$

GPT-1의 구조는 Transformer decoder를 여러개 쌓아서 LM으로 사용하였다. ~~Multi head self attention은 입력 context 범위를 넘어서도 볼 수 있음.(즉, 원래 방법에서 제제 안했다.)~~

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled.png)

h0은 임베딩만 진행하고, h1부터 hn까지는 transformer decoder를 통과하는 것을 확인할 수 있다. 그리고 가장 마지막 transformer decoder의 output을 softmax 해서 조건부 확률P를 구해낸다.

**Supervised fine-tuning**

labeled dataset은 x는 토큰들의 시퀀스라 보면 되고, y는 x에 대한 라벨이다. 위의 조건부 확률을 구하는 부분에 있어서 labeled dataset은 가장 마지막 단에 Feedforward Layer를 추가해서 x에 대한 y를 구해낸다.

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%201.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%201.png)

위의 목적함수를 supervised learning dataset(labeled dataset)에도 적용하면 다음과 같다. C는 labeled dataset을 의미한다.

$$L_2(C) = \sum_{(x,y)} logP(y|x_{1}, ...,x_{m};\theta)$$

보조 학습 목적함수를 LM에 추가하는 것이 fine-tuning함에 있어서 (a) supervised model의 generalization이 향상되고, (b) 빠르게 수렴하게 만들어 학습에 도움이 되는 것을 발견했다고 한다. 아래의 목적함수를 최적화 한다.

$$L_3(C) = L_2(C)+L_1(C)$$

**Task-specific input transformations**

각 task마다 input의 형식을 다르게 주었는데, 아래 그림에서 직관적으로 확인할 수 있다. 

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%202.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%202.png)

### Experiments

**setup (unsupervised)**

Data : BooksCorpus 데이터(문장 순서 그대로 지켜서 학습 → long-range 정보 얻음) + Word Benchmark 데이터 (문장단위로 셔플링)

Model : Transformer Decoder 12개 사용 + masked self-attention, Adam, BPE vocab 40,000, activation으로 GELU 사용

**Supervised fine-tuning**

다양한 task의 supervised dataset으로 실험을 진행해본 결과 가장 좋은 성능을 보이는 것을 확인할 수 있었다.

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%203.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%203.png)

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%204.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%204.png)

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%205.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%205.png)

### Analysis

**Impact of number of layers transferred**

Unsupervised data로 학습한 모델을 fine-tuning 에 적용할 때, 각각의 transformer layer는 labeled data 문제를 해결하는데 있어서 매우 유용한 도움을 주는 것으로 확인가능하다.

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%206.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%206.png)

**Zero-shot Behaviors**

연구자들은 근본적인 generative model이 LM capability를 향상시키기 위해 많은 task를 수행하는 법을 배울 수 있고, LSTM과 비교해서 transformer의 attentional memory가 transfer에 도움이 된다고 가정하였다. 이 모델의 또 다른 중요한 성과는 다양한 작업에서 상당한 제로샷 성능이다. 이 논문은 모델이 pre-training으로 인해 질문-응답, 스키마 해상도, 감정 분석 등과 같은 다양한 NLP 작업에서 제로샷 성능으로 진화했음을 입증했다.

![Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%207.png](Improving%20Language%20Understanding%20by%20Generative%20Pre%20616ba60f8d814894880a0065e10679c5/Untitled%207.png)

### Conclusion

논문에서는 generative pre-training과 discriminative fine-tuning을 통해 좋은 성능을 내는 single task-agnostic model을 만들어 낼 수 있음을 보여주었다. 길고 연이어진 말뭉치(unsupervised data)들로 학습하여 모델이 12개 task중 9개 분야에서 최고 기록을 달성하였다. 

---

reference

- [https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)
- [https://vanche.github.io/NLP_Pretrained_Model_GPT/](https://vanche.github.io/NLP_Pretrained_Model_GPT/)