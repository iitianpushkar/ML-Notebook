# Deep Learning, CNNs, Transformers, VLMs and Generative AI — A Unified Notebook

---

# 1. Classical Computer Vision Era

Before deep learning, computer vision relied heavily on:

* SIFT
* HOG
* handcrafted edge detectors
* manually engineered features

Problems:

* brittle pipelines
* poor scalability
* weak generalization

The major revolution came when deep neural networks started learning features automatically from raw images.

---

# 2. AlexNet (2012)

AlexNet started the deep learning revolution in computer vision.

Core pipeline:

```text
image
↓
convolutions
↓
pooling
↓
fully connected layers
↓
class prediction
```

---

## AlexNet Architecture

```text
Input
↓
Conv1
↓
MaxPool
↓
Conv2
↓
MaxPool
↓
Conv3
↓
Conv4
↓
Conv5
↓
MaxPool
↓
FC6
↓
FC7
↓
FC8
↓
Softmax
```

5 convolution layers + 3 fully connected layers.

---

# 3. Convolution Layer vs Fully Connected Layer

---

## Fully Connected Layer

Every neuron connects to ALL input values.

Mathematically:

```math
y = Wx+b
```

Problem for images:

Huge parameter count.

Example:

```math
224\times224\times3 = 150,528
```

A fully connected layer with 1000 neurons:

```math
150,528 \times 1000
```

~150 million parameters.

---

## Convolution Layer

Instead of global connectivity:

* local receptive fields
* small kernels
* weight sharing

Example kernel:

```math
K=
\begin{bmatrix}
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix}
```

This detects vertical edges.

---

# 4. Convolution Mathematics

At location:

```math
(i,j)
```

convolution computes:

```math
(X*K)(i,j)
=
\sum_m \sum_n X(i-m,j-n)K(m,n)
```

Meaning:

1. take local image patch
2. element-wise multiply with kernel
3. sum values

The multiplication step is essentially a local Hadamard product.

---

# 5. Why CNNs Worked So Well

CNNs have strong inductive biases:

* locality
* translation equivariance
* weight sharing
* hierarchical feature learning

---

## Feature Hierarchy

Early layers:

* edges
* gradients
* colors

Middle layers:

* textures
* patterns
* object parts

Deep layers:

* faces
* animals
* cars
* objects

---

# 6. VGG Networks

VGG showed:

> deeper CNNs improve performance.

Key idea:

* use many small 3×3 convolutions

instead of large kernels.

---

# 7. GoogLeNet / Inception

Introduced:

# Inception Modules

Multiple convolutions performed in parallel:

* 1×1
* 3×3
* 5×5

Outputs concatenated together.

This improved:

* multi-scale feature extraction
* computational efficiency

---

# 8. Residual Networks (ResNet)

As CNNs became deeper, training became difficult.

Problem:

* vanishing gradients
* degradation problem

Surprisingly:

* deeper networks performed worse.

---

## Core ResNet Idea

Instead of learning:

```math
H(x)
```

learn:

```math
F(x)=H(x)-x
```

So output becomes:

```math
y = F(x)+x
```

This is called:

# residual connection

or

# skip connection

---

## Why Residual Connections Help

Gradient can directly flow through:

```text
skip path
```

This prevents:

* vanishing gradients
* optimization collapse

---

## ResNet Block

```text
input
↓
Conv
↓
ReLU
↓
Conv
↓
+ skip connection
↓
output
```

---

# 9. Encoder–Decoder CNNs

Classification CNNs solve:

```text
image → class
```

But many tasks require:

```text
image → image
```

Examples:

* segmentation
* denoising
* super-resolution
* image generation

---

# 10. Core Encoder–Decoder Pipeline

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

Compresses image into semantic features.

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

---

## Decoder

Restores spatial structure.

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

---

# 11. U-Net

U-Net became one of the most important encoder–decoder architectures.

---

## U-Net Structure

```text
Encoder ↓
latent bottleneck
Decoder ↑
```

with:

* skip connections

---

## Why Skip Connections Matter

Encoder loses:

* spatial precision
* edges
* localization

Skip connections restore:

* boundaries
* details
* segmentation quality

---

# 12. Variational Autoencoders (VAE)

Ordinary autoencoders compress images into latent vectors.

Problem:

* latent space becomes irregular

---

# 13. VAE Core Idea

Instead of predicting single vector:

```math
z = E(x)
```

predict distribution:

```math
\mu(x), \sigma(x)
```

Then sample:

```math
z \sim \mathcal{N}(\mu,\sigma^2)
```

---

# 14. VAE Loss

```math
L=
||x-\hat{x}||^2
+
D_{KL}(q(z|x)||p(z))
```

---

# 15. Why VAE Was Important

VAEs introduced:

* smooth latent spaces
* semantic compression
* latent generative modeling

Foundation for:

* latent diffusion
* Stable Diffusion

---

# 16. Generative Adversarial Networks (GANs)

GANs introduced adversarial learning.

Two networks compete:

---

## Generator

```text
noise z → fake image
```

---

## Discriminator

```text
image → real/fake probability
```

---

# 17. GAN Objective

Generator tries to fool discriminator.

Discriminator tries to distinguish:

* real images
* fake images

---

## GAN Loss Function

```math
\min_G\max_D
\mathbb{E}_{x\sim p_{data}}[\log D(x)]
+
\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
```

---

# 18. Deep Convolutional GANs (DCGAN)

DCGAN introduced stable convolutional GAN training.

---

## Generator Architecture

```text
noise z
↓
project + reshape
↓
transposed convolution
↓
transposed convolution
↓
transposed convolution
↓
image
```

---

## Discriminator Architecture

```text
image
↓
convolution
↓
convolution
↓
convolution
↓
real/fake
```

---

# 19. Important DCGAN Innovations

---

## 1. Remove Pooling

Use:

* strided convolutions
* learned downsampling

---

## 2. Use BatchNorm

Stabilized GAN training.

---

## 3. Use ReLU / LeakyReLU

Generator:

* ReLU

Discriminator:

* LeakyReLU

---

## 4. Fully Convolutional Design

No fully connected hidden layers.

---

# 20. Why GANs Were Important

GANs introduced:

* adversarial learning
* realistic image synthesis
* semantic latent spaces

GANs dominated image generation before diffusion models.

---

# 21. GAN Latent Space

GAN learns:

```math
z \rightarrow image
```

Nearby latent vectors generate:

* similar images
* smooth semantic transitions

Example:

```text
smiling woman
− neutral woman
+ neutral man
=
smiling man
```

---

# 22. Sequence-to-Sequence Models

Before transformers:

```text
encoder RNN → decoder RNN
```

used for:

* translation
* summarization

Problems:

* sequential bottlenecks
* poor long-range memory

---

# 23. Attention Mechanism

Attention asks:

```text
“What parts of the sequence matter?”
```

---

## Scaled Dot Product Attention

```math
Attention(Q,K,V)
=
softmax(QK^T/\sqrt{d_k})V
```

Where:

* Q = queries
* K = keys
* V = values

---

# 24. Transformer (2017)

Transformers removed recurrence entirely.

---

## Transformer Pipeline

```text
embeddings
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

# 25. Self Attention Mathematics

Suppose:

```math
X \in \mathbb{R}^{N\times D}
```

Compute:

```math
Q=XW_Q
K=XW_K
V=XW_V
```

Attention matrix:

```math
A=softmax(QK^T/\sqrt{d_k})
```

Output:

```math
AV
```

Every token attends to every other token.

---

# 26. BERT vs GPT

---

## BERT

Encoder-only transformer.

Learns:

```math
P(token|context)
```

Good for:

* understanding
* QA
* classification

---

## GPT

Decoder-only transformer.

Good for:
- generation
- reasoning
- chat

---

# 27. Vision Transformer (ViT)

ViT treats image patches as tokens.

---

# 28. ViT Pipeline

```text
image
↓
patchify
↓
flatten patches
↓
linear embeddings
↓
transformer
↓
classification
```

---

# 29. Patchification Mathematics

Suppose:

```math
224\times224\times3
```

Patch size:

```math
16\times16
```

Number of patches:

```math
N = HW/P^2
```

```math
N = 196
```

---

# 30. Why ViT Was Revolutionary

CNN philosophy:

```text
vision requires inductive biases
```

ViT philosophy:

```text
large-scale data can learn biases automatically
```

ViT showed:

* convolutions are not fundamentally necessary for vision

---

# 31. CLIP

CLIP learns shared image-text embedding space.

---

## CLIP Architecture

```text
image → image encoder
text → text encoder
```

Both embeddings mapped into same semantic space.

---

# 32. CLIP Objective

```math
cosine(v_i,v_t)
```

maximize similarity for:

* correct image-text pairs

minimize for:

* incorrect pairs

---

# 33. What CLIP Actually Does

CLIP performs:

* alignment
* semantic understanding
* retrieval
* zero-shot classification

NOT image generation.

---

# 34. DALL·E

DALL·E performs:

```text
text → image generation
```

---

# 35. DALL·E Pipeline

```text
text
↓
transformer
↓
image tokens
↓
decoder
↓
image
```

Images represented using:

* VQ-VAE tokens

Transformer predicts image tokens autoregressively.

---

# 36. CLIP vs DALL·E

---

## CLIP

Learns:

```math
f(image)\approx g(text)
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

# 37. Diffusion Models

Diffusion models generate images via iterative denoising.

---

# 38. Forward Diffusion

```math
x_t=
\sqrt{1-\beta_t}x_{t-1}
+
\sqrt{\beta_t}\epsilon
```

Gradually:

* destroys image
* adds Gaussian noise

Eventually:

* pure Gaussian noise

---

# 39. Reverse Diffusion

Learn:

```math
P(x_{t-1}|x_t)
```

Meaning:

* denoise step-by-step

---

# 40. Stable Diffusion

Stable Diffusion combines:

* VAE
* diffusion
* CLIP
* attention
* U-Net

---

# 41. Stable Diffusion Pipeline

```text
image
↓
VAE encoder
↓
latent space
↓
diffusion
↓
UNet denoising
↓
VAE decoder
↓
image
```

---

# 42. Why Latent Diffusion Was Revolutionary

Instead of diffusing:

```math
512\times512\times3
```

diffuse:

```math
64\times64\times4
```

Much cheaper and faster.

---

# 43. Vision Language Models (VLMs)

VLMs combine:

* vision encoder
* language model

---

# 44. VLM Pipeline

```text
image
↓
vision encoder
↓
visual embeddings
↓
LLM
↓
text output
```

Examples:

* GPT-4V
* Gemini
* LLaVA

---

# 45. Are VLMs Image Generators?

Usually no.

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

# 46. Unified Evolution Chain

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
U-Net
↓
VAEs
↓
GANs / DCGAN
↓
Attention
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

---

# Chronological List of Seminal Papers

| Year | Model / Paper | Category | Main Contribution | Why It Was Important | Paper |
|------|----------------|----------|-------------------|----------------------|--------|
| 2012 | AlexNet | CNN | Deep CNN breakthrough using GPUs + ReLU | Started the deep learning revolution in computer vision | [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) |
| 2014 | VGG | CNN | Very deep CNNs using stacked 3×3 convolutions | Showed depth alone greatly improves representation learning | [Paper](https://arxiv.org/abs/1409.1556) |
| 2014 | GoogLeNet (Inception) | CNN | Inception modules for multi-scale feature extraction | Made networks deeper and computationally efficient | [Paper](https://arxiv.org/abs/1409.4842) |
| 2014 | GAN | Generative Models | Adversarial image generation via generator vs discriminator | Introduced modern generative modeling | [Paper](https://arxiv.org/abs/1406.2661) |
| 2015 | ResNet | CNN | Residual skip connections | Solved vanishing gradients and enabled ultra-deep networks | [Paper](https://arxiv.org/abs/1512.03385) |
| 2015 | U-Net | Segmentation | Encoder–decoder segmentation architecture | Became standard for medical imaging and segmentation tasks | [Paper](https://arxiv.org/abs/1505.04597) |
| 2015 | DCGAN | GAN | Stable convolutional GAN architecture | Made GAN training significantly more stable | [Paper](https://arxiv.org/abs/1511.06434) |
| 2015 | Seq2Seq | NLP | Encoder–decoder sequence learning | Foundation for translation and modern text generation | [Paper](https://arxiv.org/abs/1409.3215) |
| 2016 | WaveNet | Audio Generation | Autoregressive raw audio generation | Major breakthrough in speech synthesis | [Paper](https://arxiv.org/abs/1609.03499) |
| 2017 | Transformer | NLP | Attention-only architecture | Replaced RNNs and became foundation of all modern LLMs | [Paper](https://arxiv.org/abs/1706.03762) |
| 2017 | Capsule Networks | CNN | Dynamic routing between capsules | Attempted to preserve spatial hierarchies better than CNNs | [Paper](https://arxiv.org/abs/1710.09829) |
| 2018 | BERT | NLP | Bidirectional transformers | Revolutionized language understanding and embeddings | [Paper](https://arxiv.org/abs/1810.04805) |
| 2018 | GPT | LLM | Generative pretraining with transformers | Started autoregressive LLM scaling trend | [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 2018 | StyleGAN | GAN | Style-based image generation | Produced highly realistic controllable faces/images | [Paper](https://arxiv.org/abs/1812.04948) |
| 2019 | XLNet | NLP | Permutation-based autoregressive training | Combined strengths of autoregressive and bidirectional models | [Paper](https://arxiv.org/abs/1906.08237) |
| 2019 | T5 | NLP | Text-to-text transfer transformer | Unified all NLP tasks into one framework | [Paper](https://arxiv.org/abs/1910.10683) |
| 2020 | GPT-3 | LLM | Large-scale in-context learning | Emergent few-shot learning from scaling | [Paper](https://arxiv.org/abs/2005.14165) |
| 2020 | ViT | Vision Transformer | Transformers for vision | Showed transformers can outperform CNNs in vision | [Paper](https://arxiv.org/abs/2010.11929) |
| 2020 | DDPM | Diffusion Models | Denoising diffusion probabilistic models | Foundation of modern diffusion image generation | [Paper](https://arxiv.org/abs/2006.11239) |
| 2021 | CLIP | Vision-Language | Vision-language alignment using contrastive learning | Enabled zero-shot image classification | [Paper](https://arxiv.org/abs/2103.00020) |
| 2021 | DALL·E | Multimodal | Text-to-image generation | First major transformer-based image generation system | [Paper](https://arxiv.org/abs/2102.12092) |
| 2021 | Switch Transformer | MoE | Sparse Mixture-of-Experts transformers | Enabled trillion-parameter scaling efficiently | [Paper](https://arxiv.org/abs/2101.03961) |
| 2022 | Stable Diffusion | Diffusion | Latent diffusion image generation | Open-source explosion of AI image generation | [Paper](https://arxiv.org/abs/2112.10752) |
| 2022 | PaLM | LLM | Large Pathways Language Model | Demonstrated scaling laws and reasoning emergence | [Paper](https://arxiv.org/abs/2204.02311) |
| 2022 | Chinchilla | Scaling Laws | Compute-optimal scaling laws | Changed how LLMs are trained efficiently | [Paper](https://arxiv.org/abs/2203.15556) |
| 2022 | Flamingo | Vision-Language | Few-shot multimodal learning | Important step toward general multimodal agents | [Paper](https://arxiv.org/abs/2204.14198) |
| 2023 | LLaMA | Open LLM | Efficient open-weight language models | Triggered open-source LLM ecosystem | [Paper](https://arxiv.org/abs/2302.13971) |
| 2023 | Segment Anything (SAM) | Vision | Universal segmentation model | General-purpose segmentation foundation model | [Paper](https://arxiv.org/abs/2304.02643) |
| 2023 | GPT-4V | Multimodal LLM | Vision-language understanding | Combined image and text reasoning at scale | [Technical Report](https://cdn.openai.com/papers/gpt-4.pdf) |
| 2023 | Gemini | Multimodal LLM | Native multimodal transformer | Unified multimodal reasoning across text/image/audio/video | [Paper](https://arxiv.org/abs/2312.11805) |
| 2024 | Sora | Video Generation | Diffusion transformer video generation | Major leap in realistic world simulation and video synthesis | [Report](https://openai.com/research/video-generation-models-as-world-simulators) |                 |

