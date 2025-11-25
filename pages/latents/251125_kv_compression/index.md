## **KV Cache Compression**

Is it possible to compress the Key-Value cache without significant degradation in perplexity?

## **The Problem**

As context length grows ($L$), the KV cache memory requirement grows linearly ($O(L)$). For a 7B model with 8k context, this is manageable. But for 128k context, the KV cache becomes the bottleneck, not the model weights.

## **Hypothesis**

"Not all tokens are created equal."

Most attention heads focus on:

1. **Local context** (recent tokens)  
2. **Special tokens** (punctuation, separators)  
3. **"Heavy hitters"** (tokens with high attention scores globally)

If we can identify and keep only these heavy hitters, we might achieve 80% compression.

## **Experiment Sketch**

* **Method:** H2O (Heavy Hitter Oracle) approach.  
* **Metric:** Passkey retrieval accuracy.  
* **Goal:** Fit 128k context on a single A100 (40GB).

### **Draft Code Snippet**

```
def evict\_kv\_cache(cache, attention\_scores, budget):  
    \# Sort by accumulated attention scores  
    indices \= torch.topk(attention\_scores, k=budget, dim=-1).indices  
    return cache.gather(indices)
```
## **TODO**
- [ ]  Implement sliding window attention baseline.
- [x]  Visualize attention maps for long documents.