# Sample — read it to see the mechanism; it runs as-is with numpy installed (python3 causal_attention_demo.py)
import numpy as np

# Four token vectors standing in for the first pieces of the ticket: "Order", " ", "482", "13" (d_model = 4 for readability)
X = np.array([[1.0, 0.0, 0.5, 0.2],
              [0.3, 1.0, 0.1, 0.0],
              [0.2, 0.9, 0.0, 0.4],
              [0.0, 0.1, 1.0, 0.7]])
d_k = X.shape[1]
rng = np.random.default_rng(0)
# In a real model these projections are learned; random here so the shapes are honest (this is a single head)
W_q, W_k, W_v = (rng.normal(size=(d_k, d_k)) for _ in range(3))
Q, K, V = X @ W_q, X @ W_k, X @ W_v

scores = Q @ K.T / np.sqrt(d_k)            # every token scored against every token: n x n (the paper's eq. 1, before softmax)
causal = np.triu(np.ones_like(scores), k=1).astype(bool)
scores[causal] = -np.inf                    # decoder-only: a token may not see later tokens
weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)  # softmax per row; exp(-inf) is 0
out = weights @ V                           # each row: weighted mix of the values it is allowed to see

np.set_printoptions(precision=2, suppress=True)
print(weights)    # upper triangle prints as 0.00 — that is the mask; each row sums to 1
print(out.shape)  # (4, 4): one new representation per token
