# Deep Learning, CNNs, Transformers, VLMs and Generative AI — A Unified Notebook

---

# Encoder–Decoder CNNs (VERY IMPORTANT)

Classification CNNs like AlexNet only solved:

```text
image → class
```

But many vision tasks require:

```text
image → image
```

Examples:

* segmentation
* denoising
* image generation
* depth estimation
* super-resolution

This led to encoder–decoder architectures.

---

## Core Idea

```text
image
↓
encoder
↓
latent representation
↓
decoder
↓
output image
```

---

## Encoder

The encoder compresses the image into semantic features.

Example:

```text
256×256
↓
128×128
↓
64×64
↓
32×32
```

Spatial size decreases while semantic abstraction increases.

The encoder learns:

* edges
* textures
* objects
* high-level semantics

---

## Decoder

The decoder reconstructs spatial information.

Example:

```text
32×32
↓
64×64
↓
128×128
↓
256×256
```

The decoder restores:

* boundaries
* textures
* image structure

---

## Why This Was Revolutionary

AlexNet asked:

```text
“What is this image?”
```

Encoder–decoder networks asked:

```text
“How do I transform this image?”
```

This became the foundation for:

* segmentation
* VAEs
* GANs
* diffusion models
* Stable Diffusion

---

# U-Net (2015)

U-Net became one of the most important encoder–decoder architectures.

Architecture:

```text
Encoder ↓
latent bottleneck
Decoder ↑
```

with skip connections.

---

## Why Skip Connections Matter

Deep encoders lose fine details.

The decoder may know:

* “there is a cat”

but lose:

* exact edges
* object boundaries

Skip connections directly pass early features to decoder layers.

This preserves:

* localization
* sharpness
* edges

---

## U-Net Became Foundation For

* medical segmentation
* diffusion models
* Stable Diffusion
* image restoration

---

# Variational Autoencoders (VAE)

Ordinary autoencoder:

```text
image
↓
encoder
↓
latent vector
↓
decoder
↓
reconstructed image
```

Problem:

* latent space becomes disorganized
* random sampling fails

---

## VAE Core Idea

Instead of encoding image into a single point:

```math
z = E(x)
```

VAE predicts:

* mean
* variance

```math
μ(x), σ(x)
```

Then samples:

```math
z ~ N(μ, σ²)
```

This creates a smooth latent space.

---

## VAE Loss Function

```math
L =
||x - x̂||²
+
D_KL(q(z|x) || p(z))
```

Two components:

### Reconstruction Loss

Preserves image quality.

### KL Divergence

Forces latent distributions toward:

```math
p(z)=N(0,I)
```

This allows meaningful sampling.

---

# Why VAE Was Important

VAE introduced:

# latent semantic compression

Instead of storing raw pixels:

```text
store semantic meaning/features
```

This became foundational for:

* latent diffusion
* Stable Diffusion
* multimodal latent spaces

---

# Attention Mechanism

RNNs struggled with:

* long-range dependencies
* sequential bottlenecks

Attention solved this.

---

## Scaled Dot Product Attention

```math
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

Where:

* Q = queries
* K = keys
* V = values

---

## Intuition

Attention asks:

```text
“Which other tokens matter for this token?”
```

Every token can interact with every other token.

---

# Transformer (2017)

Transformers removed recurrence entirely.

Architecture:

```text
input embeddings
↓
self-attention
↓
MLP
↓
stacked transformer blocks
```

Advantages:

* parallelizable
* scalable
* captures long-range dependencies

---

# Self-Attention Mathematics

Suppose:

```math
X ∈ R^{N×D}
```

Compute:

```math
Q = XW_Q
K = XW_K
V = XW_V
```

Attention matrix:

```math
A = softmax(QKᵀ / √dₖ)
```

Output:

```math
AV
```

Every token attends to every other token.

---

# BERT vs GPT

## BERT

Encoder-only transformer.

Learns:

* masked token prediction

Models:

```math
P(token | surrounding context)
```

Good for:

* understanding
* QA
* classification

---

## GPT

Decoder-only transformer.

Learns:

* next token prediction

```math
P(x₁,x₂,...,xₙ)
=
∏ P(xᵢ | x<i)
```

Good for:

* generation
* reasoning
* chat

---

# Vision Transformer (ViT)

ViT introduced the idea:

```text
image = sequence of patches
```

Pipeline:

```text
image
↓
split into patches
↓
flatten patches
↓
linear embeddings
↓
transformer
↓
classification token
↓
class prediction
```

---

## Patchification Mathematics

Suppose image:

```math
224×224×3
```

Patch size:

```math
16×16
```

Number of patches:

```math
N = HW/P²
```

```math
N = 224×224 / 16² = 196
```

Each patch becomes a token.

---

# Why ViT Was Revolutionary

CNN philosophy:

```text
vision needs inductive biases
```

ViT philosophy:

```text
large-scale data can learn biases automatically
```

ViT showed:

> Convolutions are not fundamentally necessary for vision.

---

# CLIP

CLIP learned a shared image-text latent space.

Architecture:

```text
image → image encoder
text → text encoder
```

Both embeddings mapped into same semantic space.

---

## CLIP Objective

Given:

* image embedding v_i
* text embedding v_t

maximize:

```math
cosine(v_i,v_t)
```

Correct image-text pairs:

* pulled together

Incorrect pairs:

* pushed apart

---

# What CLIP Actually Does

CLIP is primarily:

* alignment model
* semantic understanding system
* retrieval system

NOT inherently an image generator.

---

# DALL·E

DALL·E solved:

```text
text → image generation
```

---

## DALL·E Pipeline

```text
text
↓
text tokens
↓
transformer
↓
image tokens
↓
image decoder
```

Images represented using VQ-VAE tokens.

Transformer predicts image token sequences autoregressively.

---

# CLIP vs DALL·E

## CLIP

Learns:

```math
f(image) ≈ g(text)
```

Alignment.

---

## DALL·E

Learns:

```math
P(image|text)
```

Generation.

---

# Diffusion Models

Diffusion models generate by iterative denoising.

---

## Forward Process

Add noise gradually:

```math
x_t =
√(1-β_t)x_{t-1}
+
√β_t ε
```

Eventually:

* pure Gaussian noise

---

## Reverse Process

Learn:

```math
P(x_{t-1}|x_t)
```

Meaning:

* denoise step-by-step

---

# Stable Diffusion

Stable Diffusion combines:

* VAE
* diffusion
* CLIP embeddings
* attention

---

## Stable Diffusion Pipeline

```text
image
↓
VAE encoder
↓
latent representation
↓
diffusion process
↓
UNet / Transformer denoising
↓
VAE decoder
↓
final image
```

This is called:

# Latent Diffusion

---

# Why Latent Diffusion Was Revolutionary

Instead of diffusing:

```math
512×512×3
```

Stable Diffusion diffuses:

```math
64×64×4
```

Much cheaper and faster.

---

# Vision Language Models (VLMs)

VLMs combine:

* vision encoders
* language models

Pipeline:

```text
image
↓
vision encoder
↓
visual embeddings
↓
LLM
↓
language output
```

Examples:

* GPT-4V
* Gemini
* LLaVA

---

# Are VLMs Image Generators?

Usually:

* understanding systems
* reasoning systems
* multimodal alignment systems

They primarily model:

```math
P(text|image,text)
```

not:

```math
P(image|text)
```

However modern systems increasingly combine:

* VLMs
* diffusion generators
* multimodal transformers

into unified architectures.

---

# Final Unified Evolution Chain

```text
Classical CV
↓
AlexNet
↓
VGG / GoogLeNet
↓
ResNet
↓
Encoder–Decoder CNNs
↓
VAEs / GANs
↓
Transformers
↓
BERT / GPT
↓
Vision Transformers
↓
CLIP
↓
DALL·E
↓
Diffusion Models
↓
Stable Diffusion
↓
VLMs
↓
Unified Multimodal AI
```


# Chronological List of Seminal Papers

| Year | Paper (Link) & Citation                                     | Authors                  | Venue                      |
|------|-------------------------------------------------------------|--------------------------|----------------------------|
| 2012 | [**ImageNet Classification with Deep CNNs**](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Krizhevsky *et al.*, **NeurIPS 2012**【84†L13-L21】)     | A. Krizhevsky *et al.*    | NeurIPS 2012 (oral)        |
| 2014 | [**Very Deep CNNs for Large-Scale Image Recognition**](https://arxiv.org/pdf/1409.1556.pdf) (Simonyan & Zisserman, **arXiv 2014**【130†L52-L60】)      | K. Simonyan, A. Zisserman | ICLR 2015 (submitted)      |
| 2014 | [**Going Deeper with Convolutions (GoogLeNet)**](https://arxiv.org/pdf/1409.4842.pdf) (Szegedy *et al.*, **arXiv 2014**【133†L55-L63】)   | C. Szegedy *et al.*       | CVPR 2015 (oral)           |
| 2014 | [**Rich Feature Hierarchies (R-CNN)**](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) (Girshick *et al.*, **CVPR 2014**【137†L147-L154】) | R. Girshick *et al.*      | CVPR 2014                  |
| 2014 | [**Sequence to Sequence Learning**](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (Sutskever *et al.*, **NeurIPS 2014**【145†L1-L9】【145†L19-L28】)            | I. Sutskever *et al.*     | NeurIPS 2014               |
| 2015 | [**Deep Residual Learning (ResNet)**](https://arxiv.org/pdf/1512.03385.pdf) (He *et al.*, **arXiv 2015**【138†L39-L44】)           | K. He *et al.*            | CVPR 2016 (oral)           |
| 2015 | [**Fast R-CNN**](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) (Girshick, **ICCV 2015**【138†L11-L19】【138†L39-L44】)               | R. Girshick              | ICCV 2015 (oral)           |
| 2016 | [**Faster R-CNN**](https://arxiv.org/pdf/1506.01497.pdf) (Ren *et al.*, **NeurIPS 2015**【140†L53-L62】【140†L65-L70】)            | S. Ren *et al.*           | NeurIPS 2015 (oral)        |
| 2016 | [**YOLO: Real-Time Object Detection**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) (Redmon *et al.*, **CVPR 2016**【142†L8-L17】【142†L17-L24】)     | J. Redmon *et al.*        | CVPR 2016                  |
| 2017 | [**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762.pdf) (Vaswani *et al.*, **NeurIPS 2017**【149†L54-L62】)        | A. Vaswani *et al.*       | NeurIPS 2017               |
| 2018 | [**Deep Contextualized Word Representations (ELMo)**](https://arxiv.org/pdf/1802.05365.pdf) (Peters *et al.*, **NAACL 2018**【151†L54-L62】) | M. E. Peters *et al.*     | NAACL 2018                 |
| 2018 | [**BERT: Deep Bidirectional Transformers**](https://arxiv.org/pdf/1810.04805.pdf) (Devlin *et al.*, **NAACL 2019**【153†L55-L64】)      | J. Devlin *et al.*        | NAACL 2019 (oral)          |
| 2019 | [**Language Models are Unsupervised Multitask Learners (GPT-2)**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford *et al.*, **OpenAI Blog 2019**【157†L19-L27】) | A. Radford *et al.*       | OpenAI blog (arXiv 2019)   |
| 2020 | [**Language Models are Few-Shot Learners (GPT-3)**](https://arxiv.org/pdf/2005.14165.pdf) (Brown *et al.*, **NeurIPS 2020**【159†L69-L77】)      | T. B. Brown *et al.*      | NeurIPS 2020 (oral)        |
| 2021 | [**Vision Transformer (ViT)**](https://arxiv.org/pdf/2010.11929.pdf) (Dosovitskiy *et al.*, **ICLR 2021**【144†L59-L67】)            | A. Dosovitskiy *et al.*   | ICLR 2021 (oral)           |
| 2021 | [**CLIP: Vision-Language Pretraining**](https://arxiv.org/pdf/2103.00020.pdf) (Radford *et al.*, **ICML 2021**【165†L59-L68】【165†L69-L74】)      | A. Radford *et al.*       | ICML 2021 (oral)           |
| 2021 | [**DALL·E: Zero-Shot Text-to-Image**](https://arxiv.org/pdf/2102.12092.pdf) (Ramesh *et al.*, **arXiv 2021**【167†L51-L58】)  | A. Ramesh *et al.*        | arXiv 2021                 |
| 2022 | [**PaLM: 540B Transformer**](https://arxiv.org/pdf/2204.02311.pdf) (Chowdhery *et al.*, **arXiv 2022**【161†L69-L78】)    | A. Chowdhery *et al.*     | arXiv 2022                 |
| 2022 | [**DALL·E 2: Text-to-Image with CLIP Latents**](https://arxiv.org/pdf/2204.06125.pdf) (Ramesh *et al.*, **arXiv 2022**【169†L49-L58】) | A. Ramesh *et al.*        | arXiv 2022                 |
| 2022 | [**Stable Diffusion (LDM)**](https://arxiv.org/pdf/2112.10752.pdf) (Rombach *et al.*, **CVPR 2022**【171†L51-L60】【171†L63-L70】)  | R. Rombach *et al.*       | CVPR 2022                  |
| 2023 | [**LLaMA: Open Language Models**](https://arxiv.org/pdf/2302.13971.pdf) (Touvron *et al.*, **arXiv 2023**【163†L55-L61】) | H. Touvron *et al.*       | arXiv 2023                 |

Each paper’s summary and resources appear in the sections below. Click a title for the full PDF.

## AlexNet (Krizhevsky *et al.*, 2012)【84†L13-L21】

**Problem & Impact:** Introduced the first large-scale deep convolutional neural network (CNN) that dramatically improved image classification on ImageNet. AlexNet (5 convolutional layers, 3 fully-connected layers) set a new record, achieving 37.5% top-1 error (17.0% top-5) on ImageNet’s 1.2M-image ILSVRC-2012 task【84†L13-L21】. Its success revived deep learning in CV.

**Key Ideas:** Used multiple GPUs to train a deeper network with *ReLU* activations (speeding up convergence over tanh) and *dropout* in FC layers to reduce overfitting【84†L13-L21】. The network architecture: convolution+ReLU+max-pooling layers stacked, followed by fully-connected layers and a 1000-way softmax. Crucially, AlexNet split the model across two GPUs and used data augmentation (random crops, lighting, PCA color perturbations). It introduced *Local Response Normalization* (LRN) – biologically inspired competition between neuron activations – although this is now less used.

**Dataset & Training:** Trained on ILSVRC2012 (1.2M images, 1000 classes). The large dataset allowed training 60M-parameter network without severe overfitting. Used *stochastic gradient descent (SGD)* with momentum, weight decay. Training took about a week on two Nvidia GTX 580 GPUs【84†L13-L21】.

**Results:** AlexNet’s error beat the previous state-of-the-art (e.g., SIFT+coding methods) by a large margin: top-5 error dropped from ~26% to 17%【84†L13-L21】. It won ILSVRC2012 by a significant margin and spurred widespread interest in deep CNNs for vision.

**Prerequisites:** Understand CNNs (convolutions, pooling), SGD, ReLU and overfitting (dropout)【84†L13-L21】. Background: Goodfellow *et al.* (Ch. 6,7), Stanford’s CS231n lectures on CNNs, Alex Krizhevsky’s GPU programming tips. Hands-on: re-implement AlexNet on CIFAR-10, experiment with dropout.

## VGG & GoogLeNet (Simonyan & Zisserman; Szegedy *et al.*, 2014)【130†L52-L60】【133†L55-L63】

**Problem & Impact:** AlexNet showed deep CNNs’ power, but how to optimize architecture? *VGG* and *GoogLeNet (Inception v1)*, both 2014, explored depth and width trade-offs. VGG demonstrated that simply **increasing depth** with small filters (3×3) yields better features. GoogLeNet showed *Inception modules* that increase both depth and width efficiently.

**VGG (Very Deep ConvNets):** Simonyan & Zisserman trained networks up to 19 layers deep, with 3×3 conv filters stacked. Key finding: increasing depth from 8 to 16–19 layers gave substantial accuracy gains【130†L52-L60】. The simplicity (all convs 3×3, pooling 2×2, 2 FC layers) showed depth alone was valuable. They achieved first/second places in ILSVRC2014 classification/localization【130†L52-L60】. VGG weights learned on ImageNet generalized well to other tasks. However, VGG had 138M parameters, requiring large compute and memory. 

**GoogLeNet (Inception v1):** Introduced *Inception modules* that compute 1×1, 3×3, 5×5 convs in parallel (plus 3×3 maxpool) then concatenate. A 1×1 bottleneck before costly filters reduces dimensions. This design *widens* the network but keeps compute roughly constant【133†L55-L63】. GoogLeNet is 22 layers deep (27 including pooling) yet only ~6.8M parameters. It achieved SOTA in ILSVRC2014【133†L55-L63】. The intuition: capture multi-scale features and “Hebbian” design principles. In practice, Inception could go deeper while fitting GPU memory.  

**Results:** VGG with 19 layers reached 92.6% top-5 on ImageNet (first place)【130†L52-L60】. GoogLeNet matched or surpassed AlexNet with far fewer parameters; it won ILSVRC2014【133†L55-L63】. Both models enabled transfer learning: their pretrained weights were widely adopted.

**Prerequisites:** CNNs and backprop basics, model depth vs width trade-offs. Text: Goodfellow *et al.* Ch. 9, Stanford CS231n on network architecture design. For Inception, study *Network-in-Network* (Lin *et al.*, 2013) which inspired 1×1 bottlenecks. Code: Try reimplementing VGG block modules, and building an Inception block. Datasets: ImageNet / smaller datasets like CIFAR10 for practice.

## R-CNN Series (Girshick *et al.*, 2014–2015)【137†L147-L154】【138†L11-L19】

**Problem & Impact:** Moving from image classification to object detection requires localizing multiple objects. Traditional methods (sliding windows, DPM) were slow or low-accuracy. R-CNN (2014) introduced a CNN-based pipeline combining region proposals and classification, drastically improving detection accuracy【137†L147-L154】. Its descendants (Fast R-CNN, Faster R-CNN) iteratively improved speed and simplicity.

**R-CNN (2014):** *Rich feature hierarchies* proposed using a selective search to generate ~2000 region proposals per image, warping each to fixed size, extracting CNN features (from AlexNet), then classifying with class-specific SVMs【137†L147-L154】. It also trained a bounding-box regressor to refine localization. Importantly, the CNN was **pre-trained on ImageNet** (large data) and then fine-tuned on the smaller Pascal VOC detection set – showing transfer learning was effective【137†L147-L154】. *Results:* R-CNN achieved ~54% mAP on VOC (versus ~33% for DPM)【137†L147-L154】. Drawbacks: Very slow (CNN run per proposal, ~47s/image) and multi-stage training.

**Fast R-CNN (2015):** Built on R-CNN by sharing convolutional computation. The *entire image* is forwarded through CNN once; then a **Region of Interest (RoI) pooling** layer extracts fixed-size features for each proposal. A single network has two outputs per RoI: softmax class probabilities and bounding-box regression deltas. It uses a single multi-task loss【134†L25-L33】【134†L67-L71】. Advantages: one-stage training (single backprop for all tasks), no SVM stage, and much faster. *Results:* Trains VGG16 9× faster than R-CNN, tests 213× faster【138†L11-L19】. Achieved ~66% mAP on VOC2012【138†L39-L44】 (vs 62% R-CNN), and removed the need for huge disk storage of features【134†L25-L33】【138†L11-L19】.

**Faster R-CNN (2015):** Introduced a *Region Proposal Network* (RPN) that generates box proposals using the CNN’s conv feature maps【140†L53-L62】. The RPN is a small fully-conv network over each feature map location, predicting objectness scores and box offsets for a set of anchor boxes. Crucially, RPN **shares features** with Fast R-CNN, yielding a unified network end-to-end【140†L53-L62】. This nearly eliminates the computational bottleneck of proposals. *Results:* The unified Faster R-CNN (with VGG) runs at ~5 FPS on GPU (with 300 proposals)【140†L65-L70】 and reached SOTA on PASCAL VOC and COCO. It became the basis for many detection systems (winning ILSVRC/COCO in 2015)【140†L65-L70】.

**Prerequisites:** CNNs and region proposal concepts. Background: “Selective Search” algorithm. Tutorials: PyImageSearch R-CNN series. Hands-on: implement Fast R-CNN training on VOC (using PyTorch tutorials), experiment with RoI pooling layers. Read up on multi-task loss (classification + regression)【134†L25-L33】.

## YOLO (Redmon *et al.*, 2016)【142†L8-L17】【142†L17-L24】

**Problem & Impact:** R-CNN family improved accuracy but remained relatively slow (hundreds of ms per image). YOLO reframed detection as a **single-step regression** problem【142†L8-L17】. This yielded dramatically faster detection, enabling real-time performance.

**Key Ideas:** YOLO divides the input image into an S×S grid. Each grid cell predicts B bounding boxes (with coordinates and confidence) and C class probabilities. A single convolutional network (24 conv layers + 2 FC layers) processes the full image and directly outputs these predictions【142†L8-L17】. Because the network sees the whole image, it reasons globally and reduces false positives on background. Training uses sum-squared error on bounding box coordinates and class confidences. Unlike R-CNN’s pipeline, YOLO is trained end-to-end for detection performance【142†L8-L17】.

**Performance:** YOLO’s base model runs at **45 frames/sec** on Titan X, Fast YOLO (smaller) achieves **155 FPS**【142†L17-L24】. It attained over twice the mAP of other real-time systems of the time【142†L17-L24】. Localization errors (imprecise boxes) were higher than slower detectors, but YOLO excelled at speed and generalization (e.g. fewer false detections on new domains). It matched or outperformed then-SOTA methods when transferred to other datasets without retraining【142†L17-L24】.

**Prerequisites:** Convolutional networks, regression and coordinate parameterization. Read Goodfellow *et al.* Ch. 11 on multi-task learning. Tutorials: official YOLO GitHub/code, PyTorch YOLOv3 colabs. Experiment: train TinyYOLO on VOC, compare speed/accuracy trade-offs.  

## Transformer (Vaswani *et al.*, 2017)【149†L54-L62】

**Problem & Impact:** In sequence transduction (e.g., translation), RNNs (LSTMs) with attention had dominated, but were sequential and hard to parallelize. *Transformer* introduced a purely *attention-based* model, discarding recurrence entirely【149†L54-L62】. This enabled **massive parallelism** and faster training, revolutionizing NLP (it became the basis for BERT, GPT, etc.).

**Key Ideas:** Transformer uses *Multi-Head Self-Attention (MHSA)* and position-wise feedforward layers in both encoder and decoder stacks. The core is **scaled dot-product attention**: for query $Q$, key $K$, value $V$, 
$$\text{Attention}(Q,K,V) = \mathrm{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr)V.$$ 
Each layer has multiple heads (different linear projections) to capture diverse relations. Encoder layers have *self-attention* (keys, queries, values all from previous layer) and feedforward; decoder layers add *encoder–decoder attention*. Positional encodings (sinusoids) inject sequence order【149†L54-L62】.

**Advantages & Results:** The model achieved new SOTA on WMT 2014 EN–DE (28.4 BLEU, +2 over previous best) and EN–FR (41.8 BLEU)【149†L54-L62】. It trained much faster (3.5 days on 8 GPUs vs ~8 days) due to parallelization. The decoder–encoder attention can be seen as a soft alignment mechanism. Transformer generalized to many tasks (e.g. constituency parsing) and became the backbone of nearly all subsequent language models.

**Prerequisites:** Understanding of RNN encoder-decoder and Bahdanau attention. Mathematics of matrix dot-products and softmax. Goodfellow *et al.* Ch. 10 (seq2seq basics). Background on attention from Bahdanau *et al.* (2015) to appreciate differences. Hands-on: implement a simplified multi-head attention layer, experiment on toy translation or parsing data.  

## ELMo (Peters *et al.*, 2018)【151†L54-L62】

**Problem & Impact:** Traditional word embeddings (word2vec, GloVe) assign a single vector per word type, failing to capture context (polysemy). ELMo introduced **deep contextualized embeddings** by using the internal states of a pre-trained **bi-directional LSTM language model**【151†L54-L62】. This improved a wide range of NLP tasks and showed that pre-trained representations (not just word-level) are broadly useful.

**Key Ideas:** ELMo trains a large BiLSTM on a large corpus (WMT14 or Wikipedia) as a language model. Given a sentence, each word’s representation is a weighted sum of the hidden layers of this biLM (both forward and backward). Thus, the embedding for “bank” in “river bank” vs “bank loan” will differ. The key formula (simplified) is: 
$$\text{ELMo}_i = \gamma \sum_{j=1}^L s_j h_{i,j}$$ 
where $h_{i,j}$ are the hidden states at layer $j$ for word $i$, and $s_j$, $\gamma$ are learned scalar weights per task. 

**Results:** Adding ELMo embeddings to existing models significantly improved SOTA on six benchmark tasks (e.g. QA, entailment, sentiment)【151†L54-L62】. The authors also showed that exposing the network’s deep representations (rather than a single vector) was important for flexibility【151†L54-L62】.

**Prerequisites:** RNN language models, transfer learning in NLP. Background: Jurafsky & Martin (ch. 9 on LMs), Michael Collins NIPS ‘13 tutorial on neural LMs. Practical: Use AllenNLP’s pretrained ELMo in a downstream task (e.g. sentiment classifier) to see improvements. 

## BERT (Devlin *et al.*, 2018)【153†L55-L64】

**Problem & Impact:** Pre-Transformer word-piece LMs were typically unidirectional or shallowly bidirectional. BERT introduced **Deep Bidirectional Transformers** with masked language modeling (MLM) and next-sentence prediction (NSP) as pre-training objectives【153†L55-L64】. It achieved new SOTA on a host of tasks (GLUE, SQuAD, etc.) and popularized fine-tuning Transformer encoders for NLP.

**Key Ideas:** BERT’s architecture is a multi-layer Transformer encoder (12 or 24 layers). During pre-training, 15% of input tokens are randomly masked (replaced with `[MASK]`) and the model learns to predict them (MLM). Additionally, given two sentences A and B, BERT predicts if B is the “next sentence” (NSP task). These tasks allow learning deep contextual encodings. Fine-tuning is done by adding a simple output layer (e.g. classification head) and training on the downstream task data.

**Equation (MLM Loss):** For masked positions $m$,  
$$\mathcal{L}_{MLM} = -\sum_{m \in \text{masked}} \log p(x_m \mid \text{context}).$$ 
Similarly, NSP is binary cross-entropy on sentence order. 

**Results:** BERT set new records on 11 NLP tasks. For example, on GLUE it achieved 80.5% (7.7 pts above previous), SQuAD v1.1 F1 93.2%【153†L64-L69】. Its representations proved universally useful: a single pre-trained BERT model can be fine-tuned for varied tasks with minimal modification. BERT sparked countless variants (RoBERTa, ALBERT, etc.) and established Transformer encoders as standard.

**Prerequisites:** Transformer encoder details, masked LM concept. Knowledge of evaluation metrics (GLUE, SQuAD). Background: Radford *et al.* (2018) GPT (for generative LMs) and Radford *et al.* (2019) GPT-2 (for context of transfer learning approaches). Hands-on: Fine-tune BERT on a classification (e.g., SST-2) using Hugging Face Transformers tutorial. 

## GPT-1 (Radford *et al.*, 2018)【155†L33-L41】

**Problem & Impact:** While BERT-like models focused on understanding tasks, GPT showed that a **generative Transformer language model** can also be fine-tuned for many tasks. It popularized the paradigm of *pretrain on large unlabeled text + fine-tune on tasks*. 

**Key Ideas:** GPT-1 is a Transformer decoder (12-layer, 768-d) trained with a left-to-right language modeling objective on the BooksCorpus (7,000 books, 800M words) and Wikipedia【155†L16-L24】. To use it for classification or QA, the authors “format” the task as text: e.g. for sentiment, prepend “Review: [text] \n Sentiment: ?”. They fine-tune the entire model (all weights) on each task’s dataset. This is in contrast to BERT’s masked objective. GPT uses unsupervised pre-training and then **discriminative fine-tuning** on labeled examples (with minimal architectural change).

**Results:** The pre-trained GPT significantly outperformed task-specific models on 9 of 12 benchmarks tested【155†L33-L41】. It gave absolute improvements of +8.9% on commonsense reasoning, +5.7% on RACE QA, and +1.5% on entailment【155†L37-L41】, all without task-specific architectures. GPT-1 showed that language modeling transfers broadly.

**Prerequisites:** Left-to-right language modeling. Concept of fine-tuning vs feature extraction. Compare to BERT’s bidirectional masked pretraining. Pretraining corpora (BooksCorpus/Wikipedia). See OpenAI blog for code. Experiment: Use Hugging Face `GPT2LMHeadModel` to generate text, and fine-tune for sentiment analysis or text classification. 

## GPT-2 (Radford *et al.*, 2019)【157†L19-L27】

**Problem & Impact:** GPT-2 scaled up GPT-1 in model size and data to study *zero-shot learning*. It demonstrated that very large LMs learn to perform many tasks from natural language prompts, without fine-tuning.

**Key Ideas:** GPT-2 is a Transformer decoder with 1.5B parameters (12→48 layers depending on size), trained on a new WebText corpus (~8M documents, 40GB) curated from Reddit links【157†L9-L17】. The key novelty: *zero-shot task adaptation*. Instead of fine-tuning, the model is given a task description/prompt plus examples in the input context, and generates answers (few-shot or even zero-shot). For instance: “Translate English to French:…” or “Q: [QA question] A:”. GPT-2 is not given special tokens for tasks; tasks are encoded in plain text.

**Results:** GPT-2 achieved SOTA zero-shot performance on many tasks (language modeling benchmarks, QA, reading comprehension). It beat or matched supervised baselines on 7 of 8 LM benchmarks, and generated coherent paragraphs virtually indistinguishable from human text【157†L19-L27】. It showed performance grew roughly logarithmically with model size【157†L19-L27】.

**Prerequisites:** Transformers, zero-shot/few-shot learning idea. Understand language modeling perplexity vs evaluation metrics. Familiarize with OpenAI’s dataset creation. Lab: use the `transformers` library to load GPT-2 (small or medium) and test its zero-shot QA by prompt formatting. 

## GPT-3 (Brown *et al.*, 2020)【159†L69-L77】

**Problem & Impact:** GPT-3 tested extreme scale (175B parameters) to push few-shot learning. It showed that beyond a point, increasing size greatly enhances *in-context learning*, and generates human-level text. It further popularized the idea that giant LMs can act as general problem-solvers with no fine-tuning.

**Key Ideas:** Same architecture as GPT-2 (decoder-only), but with 175B parameters (96-layer Transformer)【159†L69-L77】. Trained on Common Crawl + filtered web + books (570B tokens). *No fine-tuning at all*: all tasks are presented as text prompts (few-shot or zero-shot). GPT-3 can do translation, question-answering, arithmetic, even code, by understanding instructions/demos in the prompt.

**Results:** GPT-3 achieved strong performance on a wide range of NLP tasks **without gradient updates**【159†L69-L77】. In some cases it was competitive with fine-tuned SOTA. For example, it solved 3-digit arithmetic puzzles and generated news articles that humans struggled to distinguish from real【159†L69-L77】. GPT-3 revealed limitations (e.g. bias, needing massive compute) but sparked enormous interest (e.g. as ChatGPT’s foundation).

**Prerequisites:** Large-scale LM training (billions of tokens), attention to computational requirements (TPU pods). Familiarity with few-shot prompting and evaluation on benchmarks like LAMBADA or arithmetic tests. Practice: use the OpenAI API (GPT-3) to experiment with prompt engineering and few-shot learning tasks.

## Vision Transformer (ViT) (Dosovitskiy *et al.*, 2020)【144†L59-L67】

**Problem & Impact:** Could Transformers (without convolutions) handle image data? ViT showed yes: by splitting an image into patches, flattening them, and feeding them as tokens to a Transformer encoder. This challenged the dominance of CNNs in vision and opened research into pure-attention image models.

**Key Ideas:** An input image (e.g. 224×224) is split into $16\times 16$ patches. Each patch is flattened and linearly projected into an embedding (this is like patch-level “words”). A learnable *[CLS]* token summarises the image. Position embeddings are added. The sequence of patch embeddings is processed by standard Transformer encoders. The classification output uses the [CLS] token embedding after the final layer.  

The model is **pure Transformer** – no conv at all【144†L59-L67】. To achieve high accuracy, ViT relies on large pre-training datasets (e.g. JFT-300M or ImageNet21k) since unlike CNNs, it lacks strong inductive biases for locality. 

**Results:** When pre-trained on large data and fine-tuned, ViT matched or exceeded state-of-the-art CNNs. For example, ViT-Large trained on ImageNet21k reached 88.55% top-1 on ImageNet. It achieved comparable or better results on CIFAR, VTAB, etc.【144†L59-L67】. ViT requires fewer computation steps to reach comparable accuracy (fewer FLOPs due to patching and no conv overhead). This introduced a new class of vision models and influenced many hybrids (Vision Transformers, DeiT, etc.).

**Prerequisites:** Transformer encoder (as in NLP). Basic image processing (patch extraction). Prior CNN knowledge for comparison. Recommended: Carion *et al.* (DETR) for understanding attention in vision, and Google’s “An Image is Worth 16x16” (the ViT paper). Lab: use Hugging Face `ViTForImageClassification` on CIFAR-10, compare with a CNN baseline.

## CLIP (Radford *et al.*, 2021)【165†L59-L68】【165†L69-L74】

**Problem & Impact:** Traditional vision models are trained for fixed classes. CLIP (Contrastive Language–Image Pre-training) instead learns from **raw (image, text) pairs**. It aligned image and text embeddings so that zero-shot classification or captioning can be done by natural language prompts. This achieved *language-level flexibility* for vision without supervised labels for each task.

**Key Ideas:** CLIP trains two encoders (an image CNN or Vision Transformer, and a text Transformer) such that their output embeddings are comparable. Given $N$ image-text pairs $(I_i, T_i)$, it optimizes a symmetric contrastive loss: each image should match its paired text more than other texts, and vice versa. Formally, using embeddings $z_i^I, z_i^T$, it minimizes an InfoNCE-like loss: for image $i$:
$$-\log \frac{e^{\cos(z_i^I,z_i^T)/\tau}}{\sum_{j=1}^N e^{\cos(z_i^I,z_j^T)/\tau}}$$
(and similarly for each text). Here $\tau$ is a temperature.

**Results:** Trained on 400M image-text pairs, CLIP learned *open-vocabulary* visual concepts. In zero-shot ImageNet classification (without seeing any ImageNet labels), CLIP matched the original ResNet-50 accuracy by simply using class names as text prompts【165†L69-L72】. It generalized to many tasks (OCR, fine-grained, etc.)【165†L59-L68】. CLIP’s learned features transfer non-trivially: on over 30 datasets, it often rivals fully supervised models【165†L59-L68】. This work was foundational for later VLMs (DALL·E, etc.) and multimodal research.

**Prerequisites:** Contrastive learning concepts. Familiarity with softmax/probability and cross-modal embedding. Recommended: review word2vec’s skip-gram for contrastive reasoning. Implement: use OpenAI’s CLIP repository or Hugging Face `clip` to do zero-shot classification (e.g. use prompts “a photo of a {label}”). 

## DALL·E 1 (Ramesh *et al.*, 2021)【167†L51-L58】

**Problem & Impact:** Showed that transformer LMs can generate high-quality images from text *zero-shot*, bridging vision and language generation. DALL·E was a 12B-parameter Transformer that proved end-to-end text-to-image synthesis is possible without special modules beyond the Transformer itself.

**Key Ideas:** DALL·E first encodes images as discrete tokens via a **VQ-VAE (Discrete Variational Autoencoder)**: an image is compressed into a sequence of quantized codebook entries. It then treats image tokens and text tokens as one long sequence and trains an autoregressive Transformer to model the joint distribution. Given a text prompt, it generates image tokens sequentially. 

Unlike GANs or diffusion, DALL·E needed **no adversarial training** or separate text encoders – it learns to map language tokens to image tokens purely by sequence modeling【167†L51-L58】. With enough data (250M text-image pairs) and parameters (12B), it learned to produce diverse, often coherent images from natural language descriptions.

**Results:** DALL·E generated novel, creative images (e.g. “a two-story pink house shaped like a shoe”). In zero-shot setting it was competitive with prior domain-specific models on text-to-image tasks【167†L51-L58】. Its samples contained coherent object composition and rough adherence to complex descriptions, albeit with some typical image LM artifacts (slightly blurry, mistakes). DALL·E’s demonstration sparked broad interest and research into text-conditional generation.

**Prerequisites:** Transformers, VQ-VAE (see van den Oord *et al.* (2017) for vector quantization), and autoregressive generative modeling. Code: OpenAI’s DALL·E (if available) or public reimplementations (e.g. minDALL·E). Try generating images from captions in a Jupyter environment (e.g. use a pre-trained DALL·E mini model).

## PaLM (Chowdhery *et al.*, 2022)【161†L69-L78】

**Problem & Impact:** Studied the effect of *extreme scale* (540B parameters) on few-shot learning for language understanding and reasoning. PaLM (Pathways Language Model) set new records, especially on challenging reasoning benchmarks (GSM8K arithmetic, BIG-bench) and code generation tasks.

**Key Ideas:** PaLM is a 540B-parameter dense (non-sparse) Transformer (Decoder-only), trained on 6144 TPUv4 chips. Key technical enabler: Google’s Pathways system allowed training such a big model efficiently. They applied standard language modeling with minimal novel changes. The focus was on empirical scaling laws: they observed that larger models improved not only language tasks but also *multi-step reasoning*.

**Results:** PaLM achieved SOTA few-shot results on hundreds of tasks【161†L69-L78】. Notably, it outperformed existing fine-tuned models on reasoning tasks (e.g. 62.5% on GSM8K arithmetic, prior ~56%), and exceeded average human performance on BIG-bench (a suite of tough tasks)【161†L73-L81】. Some tasks saw dramatic leaps (“discontinuous improvements”) as size grew【161†L73-L81】. It also showed strong multilingual and code-generation ability (tested on open-ended code benchmarks). PaLM opened the door for very large public models (e.g. Chinchilla, LLaMA) and cautioned on societal impacts (bias, toxicity) of such scale.

**Prerequisites:** Understanding of scaled training infrastructure (TPU/parallelism), language model evaluation. Familiarity with few-shot benchmarks (e.g. BIG-bench). While not much new architecture, it’s helpful to know about “Chinchilla scaling laws” (optimal data/model tradeoff) which were contemporaneous. Study: use Google Colab TPUs or Hugging Face `transformers` to try PaLM via the API (if accessible) or experiment with smaller analogs.

## DALL·E 2 (Ramesh *et al.*, 2022)【169†L49-L58】

**Problem & Impact:** DALL·E 1 produced creative images, but had limitations in resolution and realism. DALL·E 2 improved quality and diversity by introducing a **two-stage diffusion-based approach** leveraging CLIP embeddings【169†L49-L58】. It demonstrated that combining contrastive vision-language models (CLIP) with diffusion results in more photorealistic outputs and image editing capabilities.

**Key Ideas:** DALL·E 2 splits generation into two parts: 
1. A *prior* model maps a text prompt to a CLIP image embedding. This can be autoregressive or diffusion-based. 
2. A *decoder* then generates an image conditioned on that CLIP embedding using a diffusion model【169†L49-L58】.

By generating an intermediate continuous representation (CLIP latent), DALL·E 2 can produce images that better preserve semantics (CLIP ensures alignment) and improve diversity. Diffusion models (denoising sequential refinement) yield higher fidelity details. The architecture uses cross-attention (transformer U-Net for diffusion).

**Results:** DALL·E 2 produces high-resolution, highly realistic images from text, often photorealistic. It can also perform image editing (inpainting, style transfer) by manipulating the CLIP embedding or diffusion process【169†L49-L58】. Compared to DALL·E 1, it greatly improved image clarity and caption consistency. The system advanced text-to-image research and inspired open-source projects (e.g. Imagen, Stable Diffusion).

**Prerequisites:** Variational autoencoders, CLIP embeddings. Diffusion models basics (see Sohl-Dickstein *et al.*, 2015 or Ho *et al.*, 2020). Background on conditional image synthesis. Lab: try the open-source implementation (OpenAI not open-sourcing, but similar ideas in [Stable Diffusion](#stable-diffusion)) to generate images. Study the cross-attention mechanism for conditioning.

## Stable Diffusion (Rombach *et al.*, 2022)【171†L51-L60】【171†L63-L70】

**Problem & Impact:** Diffusion models achieve SOTA in image generation but are costly in pixel space. Stable Diffusion introduced **Latent Diffusion Models (LDMs)**: perform diffusion in the latent space of an autoencoder, greatly reducing computation while maintaining high quality【171†L51-L60】. This made large-scale text-to-image open models practical (leading to the widely-used Stable Diffusion).

**Key Ideas:** First, train a powerful **autoencoder** (an encoder + decoder) that compresses images to a lower-dimensional latent space. Then train a diffusion model on this latent space (rather than raw pixels). The diffusion U-Net is augmented with cross-attention layers to condition on text embeddings (from CLIP). Training in latent space drastically cuts memory/GPU needs: the paper reports near-optimal tradeoff of complexity vs detail【171†L51-L60】. 

**Results:** LDMs achieve SOTA or near-SOTA on several tasks (unconditional generation, inpainting, super-resolution) with much less compute【171†L51-L60】. By March 2022, this led to the open-source *Stable Diffusion* model (checkpoint released by Stability AI) that can generate high-resolution, photorealistic images from text prompts in seconds on consumer GPUs. The architecture and code are public, democratizing text-to-image. 

**Prerequisites:** Understanding of diffusion models (score-based SDEs, denoising). Latent variable models (VQ-VAE or autoencoders). Familiarity with CLIP for text encoding. Tutorials: see CompVis GitHub and Hugging Face Diffusers library to experiment. Try generating images locally and compare runtime/memory vs pixel diffusion. 

