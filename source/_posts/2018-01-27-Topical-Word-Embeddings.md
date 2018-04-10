---
title: Topical Word Embeddings
date: 2018-01-27
tags:
- Paper Report
- nlp

---


**Topical Word Embeddings** is an interesting paper from AAAI Conference on Artificial Intelligence, 2015, which employs LDA to assign topics for each word in the text corpus, and learns topical word embeddings (TWE) based on both words and their topics

This artical is a simple review/note I did when learning this paper

<!--more-->

---

> - This artical is just a simple review/note from my report for this paper
> -For more detailed and straightforward information with images, refer to
the report **at the bottom of this page**

---

## Introduction

### Problem / Motivation
- One word <--> One single vector
    - indiscriminative for ubiquitous **homonymy** & **polysemy**
    - It is **problematic** that one word owns a unique vector for tasks like **Text Classification**
    - Because many words have multiple senses
    - To conceive a model that can **discriminate** word senses and generate multi-embeddings for each word

- Topical word embeddings
    - based on words&their topics
    - obtain contextual word embeddings

### Related work

#### Multi-prototype vector space

##### Process
- Cluster contexts of word into groups
- Generate a distinct prototype vector for each cluster
- **TWO STEPS:**
    - Train single prototype word representation first
    - Identify multi word embeddings for each polysemous word by **clustering** all its **context window features**
        - Computed as the **average** of single prototype **embeddings** of its neighboring words in the context window

##### Challenges
- **Generate** multi-prototype vectors for each word **in isolation**
    - Ignore complicated correlations among words & their contexts
- Contexts of a word are divided into clusters with **no overlaps**
    - A word's several senses may correlate with each other
    - No clear sematic boundary between them
- other paper
    - Restriction of scalability when facing exploding training text corpus
    - The model is sensitive to the clustering algorithm and require much effort in clustering implementation and parameter tuning
    - The universality of the clustering algorithm?

#### Skip-Gram

- Main idea :
    - A well-known framework for learning **word vectors**
    - Represent each word as a **low-dimensional, dense vector** via its context words
    - If two words co-occur more frequently, then their word vectors are more **similar**, which is estimated by the cosine similarity of their word vectors


## Our Model -- TWE (Topical Word Embeddings)

### Main Process
- Employ LDA to assign each word a topic
- Form a word-topic pair <$w_i,z_i$>
- Window size is 1



### Skip-Gram

- Objective Function:

$$\mathcal{L}(D) = \frac{1}{M} \sum_{i=1}^{M}\sum_{-k\le c\le k,c\ne 0} \log{Pr(w_{i+c}|w_i)}$$

- The Probability:

$$Pr(w_c|w_i) = \frac{e^{w_c\cdot w_i}}{\sum_{w_i\in W}{e^{w_c\cdot w_i}}}$$

### TWE-1

 1. Learn word embeddings using Skip-gram
 2. Initialize each topic vector with the **average** over all words assigned to this topics
 3. Learn the < topic embeddings> while keeping < word embeddings> **unchanged**


$$\mathcal{L}(D) = \frac{1}{M} \sum_{i=1}^{M}\sum_{-k\le c\le k,c\ne 0} \log{Pr(w_{i+c}|w_i)}+\log{Pr(w_{i+c}|z_i)}$$

$$w^z=w\oplus z$$


### TWE-2

Initialize <$w_i,z_i$> with < word embeddings> from Skip-gram and learn TWE models

$$
\mathcal{L}(D) = \frac{1}{M} \sum_{i=1}^{M}\sum_{-k\le c\le k,c\ne 0} \log{Pr(<w_{i+c},z_{i+c}>|<w_i,z_i>)}$$


$$Pr(w_c|w_i) = \frac{e^{w_c^{z_c}\cdot w_i^{z_i}}}{\sum_{<w_c,z_c>\in <W,T>}{e^{w_c^{z_c}\cdot w_i^{z_i}}}}$$

### TWE-3

 1. Initialize word vectors using Skip-gram
 2. Initialize topic vectors using those from TWE-1
 3. Concatenate these two vector to form a new vector to learn TWE models using the objective function in TWE2

-----

    
### Contextual Word Embedding

- Topical Distribution of a word **w** under a specific context **c** :
$$Pr(z|w, c)\propto Pr(w|z)Pr(z|c)$$

- The contextual word embedding of the word **w** under the context **c** :
$$\vec w^c = \sum_{z\in T} Pr(z|w,c)\vec w^z$$


- Contextual word embedding will be used for computing contextual word similarity
- TWO methods for computing **word similarity**:
    - AvgSimC method: $$\sum_{z\in T}\sum_{z'\in T}Pr(z|w_i,c_i)Pr(z'|w_j,c_j)S(\vec w^z, \vec w^{z'})$$
    - MaxSimC method:
        - $\vec w^c = \vec w^z, z=arg\max_z Pr(z|w,c)$
        - $S(w_i,c_i,w_j,c_j) = \vec w_i^z\cdot \vec w_j^{z'}$

### Document Embedding

$$d=\sum_{w\in d}Pr(w|d)\vec w^z$$

-----


Dictionary

> - **Homonymy** :
the relation between two words that are spelled the same way but differ in meaning or the relation between two words that are pronounced the same way but differ in meaning
> - **Polysemy** : 
When a symbol, word, or phrase means many different things, that's called polysemy. The verb "get" is a good example of polysemy — it can mean "procure," "become," or "understand."

----

{% pdf /pdf/TWE.pdf %}
