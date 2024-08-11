---
title: '🤖 Accelerating Transformers via Conditional Computation: As Aspect of Mixture-of-Depths'
summary: ' '
date: 2024-05-22  
authors:
  - admin
  - Minjae Park
tags:
  - Paper review
image:
  caption: 'Image credit: [**DALL·E 3**](https://openai.com/index/dall-e-3/)'
---
This post is cross-posted at [EffL@POSTECH](https://effml-postech.github.io/docs/spring24/16_/)

## **Introduction**
“Choice and concentration” is an effective strategy for achieving success in problems. Sometimes, it is not necessary to put the same amount of effort and time into all problems. Expending energy on trivial issues may fail to concentrate on what truly matters. Similarly, in language models, there is a technique that does not focus equally on all tokens but allocates less budget to non-essential tokens. This technique is called conditional computation.

In this post, We will explain conditional computation strategies for Transformers, focusing on a technology announced this year called **Mixture-of-Depths.**

paper: <U><a href="https://arxiv.org/abs/2404.02258" target="_blank"> Mixture-of-Depths: Dynamically allocating compute in transformer-based language models </a></U>


Let's dive in!



## **Understanding the problem: Uniform computation in Transformers**
These days, most language models are based on Transformers, and we stack these blocks to make big models. When given an input sequence, tokens pass through these blocks to predict the next token. The problem is that the models spread computations uniformly across input sequences. Transformers use the same amount of computation for essential tokens as for non-essential ones. For instance, predicting a token within a sentence is cheaper than predicting the first token of the next sentence. Researchers want to address this issue by making Transformers focus on important tokens by allocating unimportant tokens with fewer computing resources.

## **Conditional computation for Transformers**
- **Early exiting**
  <p align="center">
    <img src=./Early_Exiting.png> 
  </p>
  Instead of passing through all layers, the model can stop early if it is confident enough about its prediction. This saves computation time and resources. Large pre-trained models like BERT can use early exiting to maintain performance while reducing computing resources.
  
- **CoLT5**
  <p align="center">
    <img src=./colt1.png width = "50%" height = "50%">
  </p>
  CoLT5 is an architecture allowing unnecessary tokens pass through light attention and light MLP. Light attention refers to a local attention layer that just calculates attention value between a few nearby tokens. Conversely, heavy Attention refers to a global attention layer that calculates some chosen token (chosen by router) and calculates attention values with all input tokens. It uses top-k routing mechanism that performs well (will be discussed in a later section).
  <p align="center">
    <img src=./colt2.png width = "40%" height = "40%">
  </p>
  The figure above is the attention map in CoLT5. Light-colored ones indicate light attention(local attention) and bold ones indicate heavy attention. The model chooses 1/16 of query tokens and 1/8 of key tokens for heavy attention calculation.
- **Mixture of Experts (MoE)**
  <p align="center">
    <img src=./moe1.png>
  </p>
  MoE is a model consisting of parallel expert models which is fitted to certain domains. Token-level routing decisions are made across the network depths. Routing decision of the model determines which expert it will be sent to.
  
## **Overview to Mixture-of-Depths (MoD)**
Our goal is to reduce the overall FLOPs by focusing on essential tokens and relatively fewer non-essential tokens. The router is responsible for determining the path each token should take. A trained router evaluates whether the token is necessary. If the token is deemed essential, it passes through self-attention and the subsequent MLP (requiring FLOPs). Otherwise, it bypasses these stages via a residual connection (saving FLOPs).
<p align="center">
    <img src=./Mixture-of-Depths.png> 
</p>
 The above image depicts the path of the model with an input sequence length of 64. The purple color shows the computation performed by that layer and the orange color shows the path taken by the residual connection.

## **Routing schemes**
Routing implementation is the most crucial part of MoD. Authors compare three routing schemes, demonstrating that MoD is the most efficient approach.

<p align="center">
    <img src=./Routing_Schemes.png> 
</p>

### **Token-choice routing**
Token-choice routing is a method where each token selects the path it will follow. The router produces probability distributions for each token across the computational paths. Based on this distribution, each token chooses its preferred path at each layer.
  
In this scheme, tokens have the flexibility to select their path, allowing for dynamic processing. However, this can lead to path-balancing issues as all tokens might prefer on the same path. It causes potential overloads on specific paths. To mitigate it, auxiliary loss is used to ensure that most tokens do not prefer a single path.
  
### **Expert-choice routing**
Expert-choice routing is the reverse version of token-choice routing. Similar to token-choice routing, the router produces a probability distribution for each token. In this scheme, instead of tokens selecting their paths, each path selects the top-{{< math >}}$k${{< /math >}} tokens based on the experts' preferences.

Using this method ensures that each path receives k tokens, maintaining balance among the paths. However, some tokens may not be selected because there might be common tokens that multiple paths prefer.

### **Expert-choice MoD**
This method is advantageous as it reduces the overall FLOPs in the model's forward pass. When k is smaller than the input sequence length, some tokens do not need to undergo self-attention and MLP computations. For the left and middle approaches in the figure, selecting the top-k tokens may result in increased FLOPs since multiple experts need to perform computations.

For the following reasons, the authors decided to use expert-choice routing and utilize only single path:
- **Efficiency of computation**
  There is no need for an auxiliary balancing loss.
- **Simplicity of implementation**
  Tokens can be chosen with the highest output value of router in order.
- **Clear criteria**
  Top-k strategy can guarantee that the most important token is calculated since the top-{{< math >}}$k${{< /math >}} tokens are independent of the magnitude of router weights. Since tokens are divided into two sets, one passing through self-attention and MLP, and the other passing through residual connections, a strategy is needed to partition tokens into these two sets.

## **Routing**
- {{< math >}}$l${{< /math >}} is a given layer.
- {{< math >}}$S${{< /math >}} is a sequence length.
- {{< math >}}$\beta=1-C/S${{< /math >}} is an user-defined capacity per batch element.
- {{< math >}}$f${{< /math >}} comprises self-attention and subsequent MLP.

{{< math display=true >}}
$$
x^{l+1}_i=\begin{cases}r^{l}_i f_i(\tilde{X}^l)+x^{l}_i, &    \text{if } r^{l}_i >  P_\beta(R^l)\\x^{l}_i, & \text{if }r^{l}_i <  P_\beta(R^l)\end{cases}
$$
{{< /math >}}

Find the {{< math >}}$\beta${{< /math >}}-th percentile ({{< math >}}$P_\beta(R^l)${{< /math >}}) of the set of router weights {{< math >}}$R^l${{< /math >}}. If the router weight {{< math >}}$r^l${{< /math >}} is greater than {{< math >}}$P_\beta(R^l)${{< /math >}}, perform self-attention and subsequent MLP computations. If it is less than {{< math >}}$P_\beta(R^l)${{< /math >}}, pass through the token residual connection.

## **Implementation**
### **Capacity**
In this paper, capacity-based routing is employed. Token *capacity* is the total proportion of tokens composing the input for a given operation. For instance, if the input sequence length is 100 and the capacity is 20%, each layer operates on the top-20 tokens determined by router weights.

By lowering the capacity of the computations, a smaller compute budget can be utilized per forward pass compared to the vanilla Transformers. In MoD, capacity is utilized as a hyperparameter to determine the proportion of tokens processed per layer operation.
### **Autoregressively sampling**
We're looking to implement expert-choice routing, but there is one distinct problem: top-k operations rely on future tokens! Our goal is for each token to determine if it belongs to the top-k using routers. To do this, every token needs access to the router weights of future tokens. Unfortunately, we cannot predict the future router weights and cannot employ autoregressive sampling. To solve this problem, the authors propose two methods.

- **Simple auxiliary loss**
  <p align="center">
    <img src=./Routing_Analysis.png width = "30%" height = "30%"> 
  </p>
  Designing an additional binary cross-entropy loss function at the router's output can resolve this issue. By incorporating this, the value of tokens in the top-k is guided to be greater than 0.5, while the value of tokens not in the top-k is guided to be less than 0.5. As token passes through the router, they are categorized into top-k set if their value exceeds 0.5. Then it passes through self-attention and subsequent MLP. Conversely, tokens with values below 0.5 pass through the residual connection. Integrating such a function impacts the primary language modeling objective by approximately 0.2-0.3%. We believe this likely refers to the extent to which performance and inference time are affected.
    
- **Small auxiliary MLP predictor**
  
  The second method does not affect the primary language modeling objective at all. The authors design a new MLP layer that functions as a binary classifier to determine whether a token is in top-k during the training process. This classifier is trained to make these determinations, and it is used in real time during the autoregressive sampling process.

These methods, in autoregressive situation, could predict whether given token is important or not in real-time. They provide empirical results that auxiliary tasks achieved 99% accuracy.

## **Open source MoD** (not official)
The following is an implementation of MoD that supports various LMs such as Mixtral, LLama3, and BLOOM. It implements MoD using PyTorch and Hugging Face Transformers library.

LINK: https://github.com/astramind-ai/Mixture-of-depths

## **Results**
### **Hyperparameter tuning**
<p align="center">
    <img src=./result2.png>
</p>

The authors first trained the model with a limited FLOPs budget (6e18) to determine the optimal hyperparameters. Through training the MoD Transformer with routing blocks and self-attention blocks arranged alternately, they found the optimal parameters. The two top-middle graphs show the actual training loss graphs for the points plotted in the left graph. Among them, MoD with 12.5% capacity generally results in lower loss values than the baseline.

- **Computation efficiency**: In the right graph, the points #1, #3 and  #2, #4 pairs are MoD models of the same parameter size. Not only does it have a lower loss value, but it also runs approximately 66% faster than the baseline.

### **isoFLOP analysis**

<p align="center">
    <img src=./result1.png>
</p>

In this figure, the training FLOPs budget is limited to 6e18, 2e19, and 1e20 comparing isoFLOP baseline and 12.5% capacity MoD.

- **Total loss**: The graph in the top-left corner shows that the isoFLOP baseline has a slightly better loss when the number of parameters is small (Note that there is a crossing point!).
- **Normalized loss**: When the x-axis is converted from parameters to FLOPs per FFW (Forward Pass) as shown in the top-right graph, MoD is better than the baseline in all cases.

### **Auto-regressive evaluation**

<p align="center">
    <img src=./result3.png>
</p>

MoD variants were evaluated during auto-regressive sampling. Each model was tested on data comprising 256,000 sequences.

- **Predictor accuracy**: Using predictor-based methods is cheaper than top-k but not more accurate. In the left graph, the performance of the predictor strategy is almost indistinguishable from the top-k strategy. Authors attribute this to the ease of learning this prediction problem.

### **Mixture-of-Depths-and-Experts (MoDE)**
<p align="center">
    <img src=./result4.png>
</p>
This figure shows the performance of MoDE and its two proposed structures. The top-left graph demonstrates that the performance of MoDE is better than both the Baseline and MoE. The right side explains the structures of Staged MoDE and Integrated MoDE.

- **Staged MoDE**: Two routers are deployed to first for determine the depth (MoD) and second for the expert (MoE).
- **Integrated MoDE**: The MoD router and MoE router are integrated into one single Router that can simultaneously decide whether to select an expert or the residual path (depth).

The paper mentions that the former is computationally efficient as it can skip self-attention operations through the MoD router, and the latter has better performance as the router mechanism is unified and self-attention operations are always performed.

## **Conclusion and discussion**

This paper insists that using MoD with a capacity 12.5% is better than the baseline transformer model.

However, there are some unresolved limitations not discussed in the paper.
- **Only loss values**: We believe this approach only indicates if parameters converge to the training dataset, not the model's performance. To ensure MoD's superiority over the baseline model, additional evaluation methods such as perplexity (WikiText-2, Lambada) and specific tasks (BoolQ, Hellaswag, etc.) should be included.

- **More experiments are needed**: The paper only compares loss values for 12.5% and 50% capacity. They also applied MoD in one of two layers, but there are no comments on why applying this method. Further studies about using one of three or four should be done.  

- **More baselines are needed**: Further studies should provide validation of MoD method by comparing other methods like COLT5 or MoE and proof of optimal hyperparameters.

## **References**
Arian et.al.,<U><a href="https://arxiv.org/abs/2105.09121" target="_blank"> Single-Layer Vision Transformers for More Accurate Early Exits with Less Overhead </a></U>, arXiv, 2021.  
Joshua et.al.,<U><a href="https://arxiv.org/abs/2303.09752" target="_blank"> COLT5: Faster Long-Range Transformers with Conditional Computation </a></U>, EMNLP, 2023.   
Noam et.al.,<U><a href="https://arxiv.org/abs/1701.06538" target="_blank"> OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER </a></U>, ICLR, 2017.
AstraMind AI (2024). Unofficial implementation for the paper "Mixture-of-Depths". https://github.com/astramind-ai/Mixture-of-depths.