# Attention Mechanism - Deep Dive with Examples

## One-Sentence Summary

**Attention lets each token learn which previous tokens are most important for predicting the next token.**

---

## The Intuition

### Without Attention (Bad)

```
To predict next token after: "The quick brown fox jumps over the lazy"

Input to MLP: [embed(the), embed(quick), ..., embed(lazy)]
             ↓
             Fixed processing - treats all positions equally
             ↓
             MLP doesn't know which words matter most!
             
Q: "What should come next?"
A: "I don't know, I processed all words the same way"

Result: Poor prediction
```

### With Attention (Good)

```
To predict next token after: "The quick brown fox jumps over the lazy"

For position 9 (after "lazy"):
  - Is "the" (position 0) relevant? Weak signal
  - Is "lazy" (position 8, just before) relevant? STRONG signal
  - Is "jumps" (position 4) relevant? Moderate signal
  
Attention learns: "Position 8 matters most for this prediction"

Now network can focus on relevant context → Better prediction

Result: High attention on "lazy" + "the" → Predicts "dog"
```

---

## Mathematical Definition

### Attention Score

```
For position i, compute score for attending to position j:

score[i,j] = (q[i] · k[j]) / sqrt(d_k)

Where:
  q[i] = Query vector for position i (what I'm looking for)
  k[j] = Key vector for position j (what I have)
  d_k = dimension of key (for scaling)

Example:
  q[i] = [0.1, -0.2, 0.3]  (3-dim query)
  k[j] = [0.2, 0.1, -0.1]  (3-dim key)
  
  score = (0.1×0.2 + (-0.2)×0.1 + 0.3×(-0.1)) / sqrt(3)
        = (0.02 - 0.02 - 0.03) / 1.732
        = -0.03 / 1.732
        = -0.017  ← Weak match (negative = low attention)

If score is:
  +1.0 → Strong match (attend more)
  +0.0 → Neutral match (attend equally)
  -1.0 → Bad match (attend less, but negative attention exists!)
```

### Attention Weights

```
Convert scores to probabilities:

weights[i] = softmax(scores[i])

Example (position i, can attend to j ∈ [0,1,2]):
  scores = [0.5, -0.2, 1.3]
  
  exp(scores) = [exp(0.5), exp(-0.2), exp(1.3)]
              = [1.65, 0.82, 3.67]
  
  sum = 1.65 + 0.82 + 3.67 = 6.14
  
  weights = [1.65/6.14, 0.82/6.14, 3.67/6.14]
          = [0.27, 0.13, 0.60]
  
Properties:
  - Sum to 1.0 (valid probability distribution)
  - Softmax amplifies differences (score 1.3 gets 60% vs 5% for score -0.2)
  - Differentiable (can backprop!)
```

### Attention Output

```
Weighted sum of values:

output[i] = sum_j(weights[i,j] × v[j])

Example (3 value vectors, each 4-dim):
  weights = [0.27, 0.13, 0.60]
  v[0] = [0.1, 0.2, 0.3, 0.4]
  v[1] = [0.5, 0.1, 0.2, 0.3]
  v[2] = [0.2, 0.3, 0.1, 0.5]
  
  output = 0.27×[0.1, 0.2, 0.3, 0.4]
         + 0.13×[0.5, 0.1, 0.2, 0.3]
         + 0.60×[0.2, 0.3, 0.1, 0.5]
  
  output = [0.027, 0.054, 0.081, 0.108]
         + [0.065, 0.013, 0.026, 0.039]
         + [0.120, 0.180, 0.060, 0.300]
         
         = [0.212, 0.247, 0.167, 0.447]
         
(Blend of all values, weighted by attention scores)
```

---

## Complete Example: Token-by-Token Prediction

### Setup

```
Vocabulary: {'.', ' ', 'a', 'b', 'c', 'd', 'e', 'h', 'i', 'l', 'o', 's', 't', 'w'}
Position 0: 'h'  (token_id = 7)
Position 1: 'e'  (token_id = 5)
Position 2: 'l'  (token_id = 9)
Position 3: 'l'  (token_id = 9)
Position 4: 'o'  (token_id = 10)

Task: Predict token at position 5 (should be ' ')

Model params:
  d_model = 8 (hidden dimension, normally 512)
  head_dim = 2 (d_model / n_head where n_head=4)
```

### Step 1: Token Embeddings

```
Vocabulary embeddings (each token → 8-dim vector):
  'h'  (7)  → [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8]
  'e'  (5)  → [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
  'l'  (9)  → [ 0.2,  0.1,  0.2,  0.1,  0.2,  0.1,  0.2,  0.1]
  'o'  (10) → [-0.2, -0.1, -0.2, -0.1, -0.2, -0.1, -0.2, -0.1]
  ' '  (1)  → [ 0.4, -0.4,  0.3, -0.3,  0.2, -0.2,  0.1, -0.1]

Sequence embedding:
  pos[0]: 'h'  → [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8]
  pos[1]: 'e'  → [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
  pos[2]: 'l'  → [ 0.2,  0.1,  0.2,  0.1,  0.2,  0.1,  0.2,  0.1]
  pos[3]: 'l'  → [ 0.2,  0.1,  0.2,  0.1,  0.2,  0.1,  0.2,  0.1]
  pos[4]: 'o'  → [-0.2, -0.1, -0.2, -0.1, -0.2, -0.1, -0.2, -0.1]

Shape: (5, 8)
```

### Step 2: Position Embeddings

```
Position embeddings (8-dim):
  pos[0] → [ 0.01, -0.01,  0.01, -0.01,  0.01, -0.01,  0.01, -0.01]
  pos[1] → [ 0.02, -0.02,  0.02, -0.02,  0.02, -0.02,  0.02, -0.02]
  pos[2] → [ 0.03, -0.03,  0.03, -0.03,  0.03, -0.03,  0.03, -0.03]
  pos[3] → [ 0.04, -0.04,  0.04, -0.04,  0.04, -0.04,  0.04, -0.04]
  pos[4] → [ 0.05, -0.05,  0.05, -0.05,  0.05, -0.05,  0.05, -0.05]

Combined (token + position):
  x[0] = [ 0.11,  0.19,  0.31,  0.39,  0.51,  0.59,  0.71,  0.79]
  x[1] = [-0.08, -0.22, -0.28, -0.42, -0.48, -0.62, -0.68, -0.82]
  x[2] = [ 0.23,  0.07,  0.23,  0.07,  0.23,  0.07,  0.23,  0.07]
  x[3] = [ 0.24,  0.06,  0.24,  0.06,  0.24,  0.06,  0.24,  0.06]
  x[4] = [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15]

Shape: (5, 8)
```

### Step 3: Query, Key, Value Projections

```
Q projection matrix (8×8):
  [0.2, 0.1, ..., 0.3]
  [0.1, 0.3, ..., 0.2]
  ...
  [0.3, 0.1, ..., 0.1]

K projection matrix (8×8): (similar)
V projection matrix (8×8): (similar)

For position 4 (want to predict next token):
  q[4] = Q @ x[4]
       = [matrix] @ [-0.15, -0.15, ...]
       = [ 0.2, -0.1,  0.3, -0.2, ...]  (8-dim)
  
  k[0] = K @ x[0] = [ 0.1,  0.2,  0.1,  0.3, ...]  (8-dim)
  k[1] = K @ x[1] = [-0.2, -0.1, -0.3, -0.2, ...]  (8-dim)
  k[2] = K @ x[2] = [ 0.2,  0.1,  0.2,  0.3, ...]  (8-dim)
  k[3] = K @ x[3] = [ 0.2,  0.1,  0.2,  0.3, ...]  (8-dim)
  k[4] = K @ x[4] = [-0.1, -0.2, -0.1, -0.2, ...]  (8-dim)
  
  v[0] = V @ x[0] = [ 0.3,  0.1,  0.2,  0.4, ...]  (8-dim)
  v[1] = V @ x[1] = [-0.1, -0.3, -0.1, -0.2, ...]  (8-dim)
  v[2] = V @ x[2] = [ 0.1,  0.2,  0.3,  0.1, ...]  (8-dim)
  v[3] = V @ x[3] = [ 0.1,  0.2,  0.3,  0.1, ...]  (8-dim)
  v[4] = V @ x[4] = [-0.2, -0.1, -0.2, -0.1, ...]  (8-dim)
```

### Step 4: Split into Heads (4 heads, 2-dim each)

```
q[4] = [ 0.2, -0.1,  0.3, -0.2,  0.4, -0.3,  0.5, -0.4]

Head 0 (dims 0-1): q_h0 = [ 0.2, -0.1]
Head 1 (dims 2-3): q_h1 = [ 0.3, -0.2]
Head 2 (dims 4-5): q_h2 = [ 0.4, -0.3]
Head 3 (dims 6-7): q_h3 = [ 0.5, -0.4]

Similarly for k and v across all positions.

Now process each head independently.
```

### Step 5: Attention Scores (Head 0 as Example)

```
Position 4, Head 0 (2-dim):
  q_h0[4] = [0.2, -0.1]

Scores for attending to each previous position:
  score[4,0] = q_h0[4] · k_h0[0] / sqrt(2)
             = ([0.2, -0.1] · [0.1, 0.2]) / 1.414
             = (0.02 - 0.02) / 1.414
             = 0.0
  
  score[4,1] = q_h0[4] · k_h0[1] / sqrt(2)
             = ([0.2, -0.1] · [-0.2, -0.1]) / 1.414
             = (-0.04 + 0.01) / 1.414
             = -0.030 / 1.414
             = -0.021
  
  score[4,2] = q_h0[4] · k_h0[2] / sqrt(2)
             = ([0.2, -0.1] · [0.2, 0.1]) / 1.414
             = (0.04 - 0.01) / 1.414
             = 0.030 / 1.414
             = +0.021
  
  score[4,3] = q_h0[4] · k_h0[3] / sqrt(2)
             = ([0.2, -0.1] · [0.2, 0.1]) / 1.414
             = (0.04 - 0.01) / 1.414
             = +0.021
  
  score[4,4] = q_h0[4] · k_h0[4] / sqrt(2)
             = ([0.2, -0.1] · [-0.1, -0.2]) / 1.414
             = (-0.02 + 0.02) / 1.414
             = 0.0

Scores: [0.0, -0.021, +0.021, +0.021, 0.0]
         ↑     ↑      ↑      ↑      ↑
       pos0  pos1   pos2   pos3   pos4
```

### Step 6: Causal Mask (Cannot See Future!)

```
Before mask:
  scores[4] = [0.0, -0.021, +0.021, +0.021, 0.0]

Causal mask: Set future positions to -inf
  Position 4 can only look at positions [0, 1, 2, 3, 4]
  (All are in past, so no masking needed for this example)

After mask:
  scores[4] = [0.0, -0.021, +0.021, +0.021, 0.0]  (unchanged)

But if predicting position 2:
  Before: [0.0, -0.021, +0.021, +0.021, 0.0]
  Mask:   Position 2 can see [0,1,2] only
  After:  [0.0, -0.021, +0.021, -inf, -inf]
```

### Step 7: Softmax (Convert to Probabilities)

```
Scores: [0.0, -0.021, +0.021, +0.021, 0.0]

exp(scores) = [exp(0.0), exp(-0.021), exp(0.021), exp(0.021), exp(0.0)]
            = [1.0, 0.979, 1.021, 1.021, 1.0]

sum = 1.0 + 0.979 + 1.021 + 1.021 + 1.0 = 5.021

weights = [1.0/5.021, 0.979/5.021, 1.021/5.021, 1.021/5.021, 1.0/5.021]
        = [0.199, 0.195, 0.203, 0.203, 0.199]
        ≈ [0.20, 0.20, 0.20, 0.20, 0.20]

Interpretation: Head 0 attends almost equally to all positions!
(Small differences due to the dot products being small)
```

### Step 8: Apply Attention (Head 0)

```
Weighted sum of values:
  output_h0[4] = 0.20×v_h0[0] + 0.20×v_h0[1] + 0.20×v_h0[2] + 0.20×v_h0[3] + 0.20×v_h0[4]

Where each v_h0[i] is a 2-dim vector:
  v_h0[0] = [0.3, 0.1]
  v_h0[1] = [-0.1, -0.3]
  v_h0[2] = [0.1, 0.2]
  v_h0[3] = [0.1, 0.2]
  v_h0[4] = [-0.2, -0.1]

  output_h0[4] = 0.20×[0.3, 0.1] + 0.20×[-0.1, -0.3] + 0.20×[0.1, 0.2] + 0.20×[0.1, 0.2] + 0.20×[-0.2, -0.1]
               = [0.06, 0.02] + [-0.02, -0.06] + [0.02, 0.04] + [0.02, 0.04] + [-0.04, -0.02]
               = [0.04, 0.02]  (2-dim)
```

### Step 9: Repeat for Heads 1, 2, 3

```
Head 1 (different query/key/value):
  → Learns different attention pattern
  → output_h1[4] = [some 2-dim vector]

Head 2:
  → output_h2[4] = [some 2-dim vector]

Head 3:
  → output_h3[4] = [some 2-dim vector]
```

### Step 10: Concatenate Heads

```
output[4] = [output_h0[4] || output_h1[4] || output_h2[4] || output_h3[4]]
          = [[0.04, 0.02] || [??, ??] || [??, ??] || [??, ??]]
          = [0.04, 0.02, ??, ??, ??, ??, ??, ??]  (8-dim)

This is the attention output for position 4!
```

### Step 11: Output Projection

```
O projection matrix (8×8):
  [0.1, 0.2, ..., 0.3]
  ...

output_proj[4] = O @ output[4]
               = [matrix] @ [0.04, 0.02, ??, ...]
               = [attn_final_0, attn_final_1, ..., attn_final_7]  (8-dim)

This becomes the attention output for position 4.
It will be added (residual) to the input and fed to MLP.
```

---

## Multi-Head Attention: Why Multiple Heads?

### Single Head (Bad)

```
Position 4 must attend to:
  - "l" at position 2 (repeat letter)
  - "l" at position 3 (repeat letter)
  - "o" at position 4 (last letter)

But single head computes ONE set of weights:
  weights = [0.20, 0.20, 0.20, 0.20, 0.20]  ← Same for all!

Cannot learn:
  - Pattern 1: "Last letter matters" (position 3-4)
  - Pattern 2: "Repeated letters matter" (positions 2-3)
  - Pattern 3: "Vowels matter" (positions 1, 4)

Single head tries to do all at once → Compromised representation
```

### Multiple Heads (Good)

```
Head 0:
  weights = [0.1, 0.1, 0.4, 0.3, 0.1]  ← Focuses on pos 2-3 (repeated)

Head 1:
  weights = [0.05, 0.05, 0.05, 0.05, 0.8]  ← Focuses on pos 4 (current)

Head 2:
  weights = [0.0, 0.4, 0.0, 0.0, 0.6]  ← Focuses on pos 1 and 4 (vowels)

Head 3:
  weights = [0.25, 0.25, 0.25, 0.25, 0.0]  ← Averages first 4

Concatenate outputs → Network gets all patterns:
  head0_output + head1_output + head2_output + head3_output
  = [pattern_of_repeats, pattern_of_current, pattern_of_vowels, pattern_of_average]
```

**Why it works:** Each head specializes in different patterns. Concatenating gives the network all perspectives.

---

## Causal Masking Visualized

### Attention Mask Matrix

```
Position:    0   1   2   3   4

    0: [  1   0   0   0   0 ]  Can attend to: [0]
    1: [  1   1   0   0   0 ]  Can attend to: [0, 1]
    2: [  1   1   1   0   0 ]  Can attend to: [0, 1, 2]
    3: [  1   1   1   1   0 ]  Can attend to: [0, 1, 2, 3]
    4: [  1   1   1   1   1 ]  Can attend to: [0, 1, 2, 3, 4]

Where 1 = allowed, 0 = masked (set to -inf)

This ensures:
  - Autoregressive (predict one token at a time)
  - Cannot cheat by looking at future
  - At inference, position T sees positions [0:T]
```

### What Happens Without Causal Mask

```
"The quick brown fox" → model sees full context → predicts "jumps" (correct)

But at test time:
"The quick brown fox [?]" → model should predict "jumps"

Without causal mask (trained with cheating):
  Model learned to look at position T+1 to predict position T
  At test, can't do that → Predicts garbage!

With causal mask (trained correctly):
  Model learned to predict from [0:T] only
  At test, works perfectly!
```

---

## Backpropagation Through Attention

### Forward Pass Recap

```
q = x @ W_q
k = x @ W_k
v = x @ W_v
scores = q @ k.T / sqrt(d_k)
weights = softmax(scores)  ← Differentiable!
output = weights @ v
```

### Backward Pass

```
loss.backward()

∂loss/∂output = [gradients from next layer]

∂loss/∂weights = ∂loss/∂output @ v.T  (chain rule)
∂loss/∂v = weights.T @ ∂loss/∂output

∂loss/∂softmax → ∂loss/∂scores
  (softmax derivative is complex: depends on all scores)

∂loss/∂q = ∂loss/∂scores @ k
∂loss/∂k = ∂loss/∂scores.T @ q

∂loss/∂W_q = x.T @ ∂loss/∂q  ← Update query weights
∂loss/∂W_k = x.T @ ∂loss/∂k  ← Update key weights
∂loss/∂W_v = x.T @ ∂loss/∂v  ← Update value weights
```

**Key insight:** Softmax creates dependencies between all score positions. Gradient of one score affects others!

---

## Common Patterns Attention Learns

### Pattern 1: Recent Context (Local Attention)

```
Position 100, Head learns:
  weights[100] = [~0, ~0, ..., 0.01, 0.05, 0.20, 0.50, 0.24]
                                 [95]  [96]  [97]  [98]  [99]  [100]

Learns: "Recent tokens most important"
Common in: Language modeling, translation
```

### Pattern 2: Bigram Copying

```
Position 100, Head learns:
  weights[100] = [0.01, 0.01, ..., 0.02, 0.80, ~0, ~0, 0.17]
                  [?]   [?]         [?]   [98]      [?]  [99]

Learns: "Copy bigram pattern" (position 98 and 99 had this pattern before)
Common in: Memorizing frequent sequences
```

### Pattern 3: All Attention (Uniform)

```
Position 100, Head learns:
  weights[100] = [1/101, 1/101, ..., 1/101]  (approximately equal)

Learns: "Average everything"
Common in: Some heads act as "fallback" in early layers
```

### Pattern 4: Syntactic Structure

```
Position 100 (noun), Head learns:
  weights[100] = [high for other nouns, low for verbs/adjectives]

Learns: "Nouns attention to nouns"
Common in: Higher layers, semantic grouping
```

---

## Why Attention is Better Than RNN

### RNN Problem

```
RNN processes sequentially:
  h[0] = f(x[0], 0)
  h[1] = f(x[1], h[0])
  h[2] = f(x[2], h[1])
  h[3] = f(x[3], h[2])
  h[4] = f(x[4], h[3])

To compute h[4], must compute h[0], h[1], h[2], h[3] first
Cannot parallelize! O(T) sequential operations

Information must flow through:
  x[0] → h[0] → h[1] → h[2] → h[3] → h[4]

If h[0] and h[4] need to interact:
  Information travels 4 hops
  May be lost/diluted (vanishing gradient)
```

### Attention Solution

```
All positions computed in parallel:
  q[0], q[1], q[2], q[3], q[4] computed simultaneously
  k[0], k[1], k[2], k[3], k[4] computed simultaneously
  v[0], v[1], v[2], v[3], v[4] computed simultaneously

Then attention connects all pairs directly:
  output[0] directly attends to [0]
  output[4] directly attends to [0, 1, 2, 3, 4]  ← Direct connection!

Benefits:
  1. Parallelizable (O(1) time, O(T) hardware parallelism)
  2. Direct connections (vanishing gradient solved)
  3. Explicit attention patterns (interpretable)
```

---

## Complexity Analysis

### Time Complexity

```
Attention on sequence of length T:

  q, k, v projection: O(T × d_model²)
  
  scores = q @ k.T: O(T² × d_model)  ← Quadratic in sequence length!
  
  softmax: O(T²)
  
  weights @ v: O(T² × d_model)

Total: O(T² × d_model)

For T=256, d_model=512:
  256² × 512 = 33,554,432 operations per attention head
  × 8 heads = 268 million operations (manageable)

For T=2048, d_model=512:
  2048² × 512 = 2,147,483,648 operations (becomes expensive!)

For T=32K, d_model=512:
  (32K)² × 512 = not feasible! (hence sparse/local attention variants)
```

### Space Complexity

```
Memory for attention:
  
  q, k, v: 3 × T × d_model
  scores matrix: T × T  ← Quadratic in sequence length!
  weights: T × T
  
For T=256, d_model=512:
  3 × 256 × 512 + 256² + 256² = 393K + 131K ≈ 0.5 MB per attention head
  × 8 heads ≈ 4 MB (tiny)

For T=2048, d_model=512:
  scores/weights: 2048² = 4.2M floats = 17 MB per head
  × 8 heads = 136 MB (significant)

For T=32K, d_model=512:
  scores/weights: (32K)² = 1B floats = 4 GB per head
  × 8 heads = 32 GB ← Infeasible on single GPU!

(This is why long-context is hard!)
```

---

## Key Takeaways

| Concept | Why It Matters |
|---------|----------------|
| **Query/Key/Value** | Different projections enable selective attention |
| **Softmax** | Converts scores to probabilities, enables backprop |
| **Causal Mask** | Ensures autoregressive property (can't peek forward) |
| **Multiple Heads** | Each head learns different patterns, concatenate = richer representation |
| **Scaled Dot-Product** | Dividing by sqrt(d_k) prevents softmax collapse |
| **Parallel Computation** | All positions computed at once (unlike RNN) |
| **Quadratic Complexity** | O(T²) is bottleneck for long sequences |
| **Direct Connections** | Vanishing gradient problem solved |

---

## Attention in Transformer Block

### Complete Block

```
Input: x (batch, seq_len, d_model)
  ↓
LayerNorm
  ↓
Self-Attention (8 heads):
  q = x @ W_q
  k = x @ W_k
  v = x @ W_v
  attn_out = attention_heads(q, k, v)
  attn_out = concat(head_0, ..., head_7) @ W_o
  ↓
Residual connection
  x = x + attn_out
  ↓
LayerNorm
  ↓
MLP (Feed-Forward):
  x = mlp(x)
  ↓
Residual connection
  x = x + mlp_out
  
Output: x (same shape as input)
```

### Why Residual Connections Matter

```
Without residual:
  x → attn → mlp → x'
  
  At layer 7: information from layer 0 very diluted
  Gradients: backprop struggles through 7 layers

With residual:
  x → (attn + x) → (mlp + x) → x'
  
  At layer 7: direct shortcut to layer 0!
  Gradients: can flow directly backward through shortcuts
  
In 8-layer model: each layer only needs to learn the "change" (residual)
                  not the full representation
```

---

## Attention in Lab 3 (Tensor Parallel)

### Single-GPU Attention

```
q = Q @ x  (32, 256, 512) → (32, 256, 512)
k = K @ x  (32, 256, 512) → (32, 256, 512)
v = V @ x  (32, 256, 512) → (32, 256, 512)

scores = (q @ k.T) / sqrt(64)  (32, 256, 256)
weights = softmax(scores)      (32, 256, 256)
output = weights @ v           (32, 256, 512)
```

### Tensor-Parallel Attention (Split Q, K, V Columns)

```
GPU 0: W_q[0:128, :], W_k[0:128, :], W_v[0:128, :]
GPU 1: W_q[128:256, :], W_k[128:256, :], W_v[128:256, :]
GPU 2: W_q[256:384, :], W_k[256:384, :], W_v[256:384, :]
GPU 3: W_q[384:512, :], W_k[384:512, :], W_v[384:512, :]

GPU 0: q_0 = Q_0 @ x  (32, 256, 128)  ← Only my part
       k_0 = K_0 @ x  (32, 256, 128)
       v_0 = V_0 @ x  (32, 256, 128)

GPU 1, 2, 3: Similar for their slices

Problem: scores = q @ k.T requires full k!
  scores_full = (concat[q_0, q_1, q_2, q_3]) @ (concat[k_0, k_1, k_2, k_3]).T
             = (32, 256, 512) @ (512, 256)
             = (32, 256, 256)  ← Everyone needs this!

Solution: All-gather k and v
  all_gather([k_0, k_1, k_2, k_3]) → k_full on all GPUs
  
  GPU 0: scores = q_0 @ k_full.T + q_1 @ k_full.T + q_2 @ k_full.T + q_3 @ k_full.T
  ← But wait, only q_0 is on GPU 0!
  
Actually:
  GPU 0: my_scores = q_0 @ k_full.T  (32, 256, 128) @ (512, 256) = (32, 256, 256)
  GPU 1: my_scores = q_1 @ k_full.T  (32, 256, 128) @ (512, 256) = (32, 256, 256)
  ...
  
  Full scores = my_scores[0] + my_scores[1] + ... (reduce!)
  
Or simpler:
  All-gather q, k, v first
  Compute attention normally
  Each GPU computes full attention
  Then split output back to GPUs
```

This is the complexity that Lab 3 teaches!
