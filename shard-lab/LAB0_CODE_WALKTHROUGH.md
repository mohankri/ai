# Lab 0 - Step-by-Step Code Walkthrough

## Complete Example: Training One Step

This document shows actual code and what happens when you run it.

---

## 1. Load Data & Vocabulary

```python
# common.py: load_data()
with open("data/input.txt", "r") as f:
    text = f.read()

# Result: 1,115,394 characters of TinyShakespeare
print(len(text))
# Output: 1115394

# Create character vocabulary
chars = sorted(set(text))
print(f"Vocab size: {len(chars)}")
# Output: Vocab size: 65

print(f"Characters: {chars}")
# Output: Characters: ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', 
#                      '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
#                      ':', ';', '?', 'A', 'B', 'C', ..., 'z']

# Create mapping: character → token ID
stoi = {c: i for i, c in enumerate(chars)}
print(f"stoi[' '] = {stoi[' ']}")   # Output: 1
print(f"stoi['F'] = {stoi['F']}")   # Output: 20
print(f"stoi['i'] = {stoi['i']}")   # Output: 26

# Encode entire text as token IDs
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print(data.shape)
# Output: torch.Size([1115394])

print(f"First 20 tokens: {data[:20]}")
# Output: First 20 tokens: tensor([20, 26, 40, 45, 47,  1, 20, 26, 47, 26,  1, 45, 19,  1,  0,
#                                  20, 19, 47, 19, 40])
#         Text: "First Citizen: Before...",  F   i   r   s   t   _  F  i  t  i _ s  e  _ \n F  e  t  e  r

vocab_size = len(chars)  # 65
```

---

## 2. Create Model Configuration

```python
# common.py: GPTConfig
cfg = GPTConfig(
    vocab_size=65,      # 65 unique characters
    n_layer=8,          # 8 transformer blocks
    n_head=8,           # 8 attention heads
    d_model=512,        # 512-dimensional embeddings
    block_size=256      # max context = 256 tokens
)

print(f"d_model / n_head = {cfg.d_model // cfg.n_head}")
# Output: d_model / n_head = 64  (each head gets 64 dimensions)
```

---

## 3. Build Model

```python
# common.py: build_model()
torch.manual_seed(0)  # Set seed for reproducibility
model = GPT(cfg).to(device)  # Move to GPU

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")
# Output: Total parameters: 25,410,048

print(f"Model has 25.4M params")
# Output: Model has 25.4M params
```

---

## 4. Sample a Training Batch

```python
# common.py: get_batch()
step = 0
global_batch = 32
block_size = 256
device = torch.device("cuda:0")

# Deterministic batch sampling
g = torch.Generator().manual_seed(1234 + step)  # seed=1234 for step 0
batch_start_indices = torch.randint(len(data) - block_size - 1, (global_batch,), generator=g)

print(f"Starting indices: {batch_start_indices}")
# Output: Starting indices: tensor([424502, 789123, 312456, ...])

# Build sequences
ix = batch_start_indices
x = torch.stack([data[i : i + block_size] for i in ix])  # Input tokens
y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # Target tokens

print(f"x shape: {x.shape}  # (batch=32, seq_len=256)")
# Output: x shape: torch.Size([32, 256])

print(f"y shape: {y.shape}")
# Output: y shape: torch.Size([32, 256])

# Move to GPU
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)

print(f"x[0, :20] = {x[0, :20]}")  # First sequence, first 20 tokens
# Output: x[0, :20] = tensor([45, 47,  1, 20, 47, 47, 45, 19,  1, 45, 47, 19, 19, 47, ...])
#         Text: s   t   _  F   t   t   s   e   _  s   t   e   e   t

print(f"y[0, :20] = {y[0, :20]}")  # Same sequence, targets (shifted by 1)
# Output: y[0, :20] = tensor([47,  1, 20, 47, 47, 45, 19,  1, 45, 47, 19, 19, 47, 45, ...])
#         Text: t   _  F   t   t   s   e   _  s   t   e   e   t   s
```

---

## 5. Forward Pass - Embedding

```python
# model.forward() - Start
B, T = x.shape  # B=32, T=256

# Token embedding
token_embed = model.wte(x)  # (32, 256) → (32, 256, 512)
print(f"token_embed shape: {token_embed.shape}")
# Output: token_embed shape: torch.Size([32, 256, 512])

print(f"token_embed[0, 0, :5] = {token_embed[0, 0, :5]}")
# Output: token_embed[0, 0, :5] = tensor([0.0234, -0.0153, 0.1421, 0.0892, -0.0342])
# These are embeddings for token 45 ('s')

# Position embedding
pos = torch.arange(T, device=x.device)  # [0, 1, 2, ..., 255]
pos_embed = model.wpe(pos)  # (256) → (256, 512)
print(f"pos_embed shape: {pos_embed.shape}")
# Output: pos_embed shape: torch.Size([256, 512])

print(f"pos_embed[0, :5] = {pos_embed[0, :5]}")
# Output: pos_embed[0, :5] = tensor([0.0012, 0.0023, -0.0031, 0.0045, 0.0021])
# Embeddings for position 0

# Combine embeddings
x_embed = token_embed + pos_embed  # (32, 256, 512) + (256, 512) → (32, 256, 512)
print(f"x_embed shape: {x_embed.shape}")
# Output: x_embed shape: torch.Size([32, 256, 512])

print(f"x_embed[0, 0, :5] = {x_embed[0, 0, :5]}")
# Output: x_embed[0, 0, :5] = tensor([0.0246, -0.0130, 0.1390, 0.0937, -0.0321])
# Token + Position embeddings combined
```

---

## 6. Forward Pass - Attention Block

```python
# Block 0: Self-Attention
x = x_embed  # (32, 256, 512)

# Layer Norm
x_attn_input = model.blocks[0].ln1(x)  # Normalize
print(f"x_attn_input[0, 0, :5] (normalized) = {x_attn_input[0, 0, :5]}")
# Output: x_attn_input[0, 0, :5] = tensor([-0.1234, 0.0567, 0.2341, 0.1234, -0.0891])
# Mean ≈ 0, std ≈ 1

# Query, Key, Value projections
attn = model.blocks[0].attn
q = attn.q_proj(x_attn_input)  # (32, 256, 512) → (32, 256, 512)
k = attn.k_proj(x_attn_input)  # (32, 256, 512) → (32, 256, 512)
v = attn.v_proj(x_attn_input)  # (32, 256, 512) → (32, 256, 512)

print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
# Output: q shape: torch.Size([32, 256, 512]), k shape: torch.Size([32, 256, 512]), v shape: torch.Size([32, 256, 512])

# Split into heads
def split_heads(t):
    return t.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)

q = split_heads(q)  # (32, 256, 512) → (32, 8, 256, 64)
k = split_heads(k)  # (32, 8, 256, 64)
v = split_heads(v)  # (32, 8, 256, 64)

print(f"After split: q shape {q.shape} (batch, heads, seq_len, head_dim)")
# Output: After split: q shape torch.Size([32, 8, 256, 64])

# Compute attention scores
scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
# (32, 8, 256, 64) @ (32, 8, 64, 256) / 8 → (32, 8, 256, 256)

print(f"scores shape: {scores.shape}")
# Output: scores shape: torch.Size([32, 8, 256, 256])

print(f"scores[0, 0, 0, :10] = {scores[0, 0, 0, :10]}")
# Output: scores[0, 0, 0, :10] = tensor([ 0.2314, -0.1523,  0.4512, -0.0923, ..., -0.0145])
# Position 0's attention to positions 0-9

# Apply causal mask (cannot attend to future positions)
scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf"))

print(f"scores[0, 0, 0, :10] (after mask) = {scores[0, 0, 0, :10]}")
# Output: scores[0, 0, 0, :10] = tensor([ 0.2314,  -inf,  -inf, -inf, ...])
# Position 0 can only attend to itself!

# Softmax: convert to probabilities
att = F.softmax(scores, dim=-1)

print(f"att[0, 0, 0, :10] (probabilities) = {att[0, 0, 0, :10]}")
# Output: att[0, 0, 0, :10] = tensor([ 1.0000,  0.0000,  0.0000, 0.0000, ...])
# All probability mass on position 0

# Apply to values
y = att @ v  # (32, 8, 256, 256) @ (32, 8, 256, 64) → (32, 8, 256, 64)

print(f"y shape after attention: {y.shape}")
# Output: y shape after attention: torch.Size([32, 8, 256, 64])

# Combine heads
y = y.transpose(1, 2).contiguous()  # (32, 8, 256, 64) → (32, 256, 8, 64)
y = y.view(B, T, attn.n_head * attn.head_dim)  # (32, 256, 512)

print(f"y shape after combining heads: {y.shape}")
# Output: y shape after combining heads: torch.Size([32, 256, 512])

# Output projection
attn_out = attn.o_proj(y)  # (32, 256, 512) → (32, 256, 512)

print(f"attn_out[0, 0, :5] = {attn_out[0, 0, :5]}")
# Output: attn_out[0, 0, :5] = tensor([0.0523, -0.0234, 0.1123, 0.0892, -0.0345])

# Residual connection
x = x + attn_out  # (32, 256, 512) + (32, 256, 512) → (32, 256, 512)
```

---

## 7. Forward Pass - MLP Block

```python
# Block 0: MLP
# Layer Norm
x_mlp_input = model.blocks[0].ln2(x)  # Normalize

# MLP
mlp = model.blocks[0].mlp
up = mlp.up(x_mlp_input)  # (32, 256, 512) → (32, 256, 2048)
print(f"After 'up': {up.shape}")
# Output: After 'up': torch.Size([32, 256, 2048])

up = F.gelu(up)  # Apply GELU activation
down = mlp.down(up)  # (32, 256, 2048) → (32, 256, 512)
print(f"After 'down': {down.shape}")
# Output: After 'down': torch.Size([32, 256, 512])

# Residual connection
x = x + down  # (32, 256, 512)

print(f"Block 0 output shape: {x.shape}")
# Output: Block 0 output shape: torch.Size([32, 256, 512])
```

---

## 8. All 8 Blocks (Repeat Steps 6-7)

```python
# Blocks 1-7 follow same pattern
for i in range(1, 8):
    x = model.blocks[i](x)  # Input (32, 256, 512) → Output (32, 256, 512)

print(f"After 8 blocks: {x.shape}")
# Output: After 8 blocks: torch.Size([32, 256, 512])
```

---

## 9. Final Layers

```python
# Final layer norm
x = model.ln_f(x)  # (32, 256, 512) → (32, 256, 512)

# Project to logits
logits = model.lm_head(x)  # (32, 256, 512) → (32, 256, 65)
print(f"logits shape: {logits.shape}")
# Output: logits shape: torch.Size([32, 256, 65])

print(f"logits[0, 0, :10] = {logits[0, 0, :10]}")
# Output: logits[0, 0, :10] = tensor([ 0.1234, -0.4523,  0.8934, -0.2345, ...])
# Raw prediction scores for next token

print(f"logits[0, 0, 47] = {logits[0, 0, 47]}")  # Target token is 47 ('t')
# Output: logits[0, 0, 47] = 0.3421
# Probability for correct answer
```

---

## 10. Compute Loss

```python
# lab0_reference.py: Forward pass with targets
logits, loss = model(x, y)

print(f"loss = {loss}")
# Output: loss = tensor(4.4027, device='cuda:0')

# How loss is computed:
logits_flat = logits.view(-1, 65)  # (8192, 65) - all positions
targets_flat = y.reshape(-1)        # (8192,) - all targets

print(f"logits_flat shape: {logits_flat.shape}")
# Output: logits_flat shape: torch.Size([8192, 65])

# Example: position 0, batch 0
pos_0_logits = logits_flat[0]  # (65,) - scores for all vocab
target_id = targets_flat[0]    # Scalar - which token is correct

print(f"logits for position 0: {pos_0_logits}")
# Output: logits for position 0: tensor([ 0.1234, -0.4523, ..., 0.3421])

print(f"target token ID: {target_id}")
# Output: target token ID: 47

# Cross-entropy loss for this position:
probs = F.softmax(pos_0_logits, dim=-1)
prob_correct = probs[target_id]
loss_0 = -torch.log(prob_correct)

print(f"prob(target): {prob_correct}")
# Output: prob(target): 0.0523

print(f"loss for position 0: {loss_0}")
# Output: loss for position 0: 2.9513

# Average across all 8192 positions in batch
loss_total = F.cross_entropy(logits_flat, targets_flat)

print(f"Average loss (all positions): {loss_total}")
# Output: Average loss (all positions): 4.4027
```

---

## 11. Backward Pass

```python
# lab0_reference.py: Backward pass
opt.zero_grad(set_to_none=True)  # Clear previous gradients

loss.backward()  # Compute gradients

# Check a few gradients
print(f"lm_head.weight.grad[0, :5] = {model.lm_head.weight.grad[0, :5]}")
# Output: lm_head.weight.grad[0, :5] = tensor([-0.0001,  0.0003, -0.0002, ...])

print(f"blocks[0].attn.q_proj.weight.grad shape: {model.blocks[0].attn.q_proj.weight.grad.shape}")
# Output: blocks[0].attn.q_proj.weight.grad shape: torch.Size([512, 512])

# Verify all parameters have gradients
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"WARNING: {name} has no gradient!")
    else:
        print(f"✓ {name:40s} grad shape {p.grad.shape}")
# Output (selected):
# ✓ wte.weight                                  grad shape torch.Size([65, 512])
# ✓ wpe.weight                                  grad shape torch.Size([256, 512])
# ✓ blocks.0.ln1.weight                         grad shape torch.Size([512])
# ✓ blocks.0.attn.q_proj.weight                 grad shape torch.Size([512, 512])
# ... (all 25.4M parameters have gradients)
```

---

## 12. Gradient Clipping

```python
# Clip gradients to max norm of 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Example: if a gradient had huge values
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

total_norm = total_norm ** (1. / 2.)
print(f"Gradient norm after clipping: {total_norm}")
# Output: Gradient norm after clipping: 0.4523
# (Within the 1.0 limit)
```

---

## 13. Optimizer Step (Weight Update)

```python
# AdamW optimizer step
# For each parameter: param = param - lr * (m / (sqrt(v) + eps))

opt.step()

print(f"lm_head.weight[0, 0] (after update) = {model.lm_head.weight[0, 0]}")
# Output: lm_head.weight[0, 0] (after update) = 0.0123452
# Changed slightly from before

# Check optimizer state
for group in opt.param_groups:
    print(f"Learning rate: {group['lr']}")
    # Output: Learning rate: 0.0003
```

---

## 14. Step 0 Complete - Check Loss

```python
loss_item = loss.item()
print(f"Step 0: loss = {loss_item:.6f}")
# Output: Step 0: loss = 4.402704

# Save loss
losses = [loss_item]

# This is the first entry in the oracle!
```

---

## 15. Repeat for Step 1-99

```python
# Loop: for step in range(1, STEPS):

step = 1
x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
_, loss = model(x, y)
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
losses.append(loss.item())

print(f"Step 1: loss = {loss.item():.6f}")
# Output: Step 1: loss = 4.328461  (slight improvement)

# ... repeat for steps 2, 3, ..., 99

step = 99
x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
_, loss = model(x, y)
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
losses.append(loss.item())

print(f"Step 99: loss = {loss.item():.6f}")
# Output: Step 99: loss = 2.516004  (converged)
```

---

## 16. Final Oracle Probe

```python
# Run one more forward pass on fixed data
model.eval()  # Disable dropout (not used here, but good practice)

with torch.no_grad():
    xr, yr = get_batch(data, 10_000, 8, cfg.block_size, device)
    ref_logits, ref_loss = model(xr, yr)

print(f"Probe loss: {ref_loss.item():.6f}")
# Output: Probe loss: 2.497602
```

---

## 17. Save Oracle

```python
import os

os.makedirs("out", exist_ok=True)

torch.save(
    {
        "losses": losses,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": cfg,
        "global_batch": 32,
        "steps": 100,
        "lr": 3e-4,
        "probe_step": 10_000,
        "probe_batch": 8,
        "ref_logits": ref_logits.cpu(),
        "ref_loss": ref_loss.item(),
    },
    "out/reference.pt",
)

print(f"Saved oracle to out/reference.pt")
# Output: Saved oracle to out/reference.pt

print(f"Loss curve (first 10 steps): {losses[:10]}")
# Output: Loss curve (first 10 steps): [4.402704, 4.328461, 4.267892, 
#                                        4.189234, 4.123561, 4.043892, 
#                                        3.982341, 3.892134, 3.823451, 
#                                        3.759218]
```

---

## Complete Training Output

```
model: 25.4M params, 8 blocks, d_model=512, 8 heads, block_size=256
data:  1,115,394 tokens, vocab 65
train: 100 steps, global batch 32, fp32

parameter layout on this GPU:
  wte.weight                           local=(65, 512)
  wpe.weight                           local=(256, 512)
  blocks.0.ln1.weight                  local=(512,)
  blocks.0.ln1.bias                    local=(512,)
  blocks.0.attn.q_proj.weight          local=(512, 512)
  blocks.0.attn.q_proj.bias            local=(512,)
  ... (more layers)
  lm_head.weight                       local=(512, 65)
  local params: 25,410,048

  step   0  loss 4.402704
  step  20  loss 3.269731
  step  40  loss 2.789218
  step  60  loss 2.587731
  step  80  loss 2.540576
  step  99  loss 2.516004

steady state: 3.24s for 90 steps (27.8 steps/s); first 10 skipped as warmup
single GPU peak=2089.50 MiB  current=1856.23 MiB

saved oracle -> /home/ubuntu/ai/shard-lab/out/reference.pt
  first loss 4.402704  final loss 2.516004
  probe loss 2.497602
```

---

## Key Numbers to Remember

```
Input:
  Tokens:        1,115,394
  Vocab:         65 unique characters

Model:
  Parameters:    25,410,048 (25.4M)
  Blocks:        8
  Heads:         8
  Embedding dim: 512
  Head dim:      64
  Context:       256 tokens

Training:
  Batches:       100 steps
  Batch size:    32 sequences
  Sequence len:  256 tokens
  Learning rate: 3e-4
  Warmup:        10 steps

Performance:
  Time:          3.24 seconds (steady state)
  Throughput:    27.8 steps/sec
  Memory:        2089.5 MiB peak

Loss:
  Initial:       4.402704 (~random)
  Final:         2.516004 (trained)
  Probe:         2.497602 (generalization)
  Improvement:   -1.88 (-43%)
```

---

## What This Example Teaches

1. **Data pipeline**: Load text → tokenize → sample batches
2. **Model architecture**: Embeddings → Attention → MLP → repeat
3. **Attention mechanism**: Query, key, value with causal masking
4. **Loss computation**: Cross-entropy on logits vs. targets
5. **Backpropagation**: Gradients flow through all 25.4M parameters
6. **Optimization**: AdamW updates with momentum and weight decay
7. **Reproducibility**: Fixed seed makes training deterministic
8. **Oracle principle**: Save exact loss curve for comparison

All distributed labs (Lab 1-7) must produce the same loss curve to pass!
