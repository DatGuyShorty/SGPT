import torch
import torch.nn as nn # common practice naming for pytorch
import torch.nn.functional as F # common practice naming for pytorch
import numpy as np # not needed for now, will decide later
import time # can be used for timing
import mmap
import random
import pickle
device ="cuda" if torch.cuda.is_available() else "cpu" # check if GPU is available to use, if yes it sets cuda to device var, else it sets cpu. this var is used to identify which device we want code to execute on
print(device)
# very important hyperparameters for training
block_size = 256 #lenght of each sequence
batch_size = 32 # #num of blocks pararlell. batches can be used to acc. train process
max_iters = 5000
learning_rate = 0.0001
weight_decay =  0.00001
# eval_interval = 2500
eval_iters = 100
n_embd = 768 #384
n_layer = 12
n_head = 8
dropout = 0.2

with open("vocab.txt","r",encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
#print(chars)
vocab_size = len(chars)
#print("Vocabulary size:", vocab_size) # used for nn.embedding, can be used on per char level for learning the importance of a char as an embedding vector, embeding dimension is used for storing how much info is in every

string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

#print("Encoding 'hello':", encode("hello")) 
#print("Decoding [60, 57, 64, 64, 67]:", decode([60, 57, 64, 64, 67])) # it worked, so commenting out and continuing with course
data = torch.tensor(encode(text), dtype=torch.long)
#print(data[:100])

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we want
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): # post norm arch
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    
class Head(nn.Module):
    """ One head of the self-attention (MultiHeadAttention) """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores also called affinities
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple head of self attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F,) ---> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2.......])
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """Linear ->ReLU-> Linear
    Simple linear layer folllowed by non linearity
    """
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
             nn.Linear(n_embd, 4 * n_embd),
             nn.ReLU(),
             nn.Linear(4 * n_embd, n_embd),
             nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab size x vocab size for corret size of embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std= 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        # idx and targets are both (B, T) tensor of ints
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) #(B, T, vocab_size)
        logits = self.lm_head(x) #(B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # B - Batch, T - Time C - dimensions
            logits = logits.view(B*T, C) # .view used for unpacking and reshaping 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # ensure that the input sequence length doesn't exceed block_size
            idx_cond = index if index.size(1) <= block_size else index[:, -block_size:]
            # get the predictions
            logits, _ = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index
    
model = GPTLanguageModel(vocab_size)
#model = torch.compile(model)
#model.half() everything just outputs nan after applying
with open("model-o5.pkl", "rb") as f:
    model = pickle.load(f)
print("model loaded sucessfully")
m = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f"Completion:\n{generated_chars}")