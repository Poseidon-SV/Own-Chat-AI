import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

def main():
    batch_size = 12
    block_size = 22 
    max_iters = 5000
    eval_interval = 500
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 250
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    class Head(nn.Module):
        """ one head of self-attention """

        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B,T,C = x.shape
            k = self.key(x)   
            q = self.query(x) 
            wei = q @ k.transpose(-2,-1) * C**-0.5 
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x) 
            out = wei @ v
            return out

    class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedFoward(nn.Module):
        """ a simple linear layer followed by a non-linearity """

        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """ Transformer block: communication followed by computation """

        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class BigramLanguageModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd) 
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape

            tok_emb = self.token_embedding_table(idx) 
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
            x = tok_emb + pos_emb 
            x = self.blocks(x) 
            x = self.ln_f(x) 
            logits = self.lm_head(x)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1) 
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx


    def main():

        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                st.markdown(f"> _Step_ {iter}:  _train loss_ {losses['train']:.4f},  _val loss_ {losses['val']:.4f}")

            xb, yb = get_batch('train')

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def user_chat():        

        chat_sample = st.text_area("You can give your own chat, text must contain atleast 5,000 chats or atleast 50,000 words OR you can drag n drop chat.txt file")
        st.write("_You can export chat form your WhatsApp by clicking 3-dots by selecting your 'Chats' > 'Chat History' > 'Export'_. __Please remove all the dates and times as it will count in incorrect data.__")
        data = st.file_uploader("Upload a Chat sample", type=["txt"])
        chatting_length = st.number_input("AI genrated chat lenght", 0, 5000, 2000)

        ok = st.button("Genrate chats")

        if ok:
            if data:
                text = data.getvalue().decode('utf-8')
                if len(text.split('\n')) > 4500:       
                    text = chat_sample                    
                else:
                    st.write(':red[The given chat sample is less then 5,000 chats]')
                    return None, None

            elif len(chat_sample.split('\n')) > 4500:       
                    text = chat_sample                    
            else:
                st.write(':red[The given chat sample is less then 5,000 chats]')
                return None, None

            return text, chatting_length
        
        return None, None



    text, chat_lenght = user_chat()
    if text:
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] 
        decode = lambda l: ''.join([itos[i] for i in l])
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.95*len(data)) 
        train_data = data[:n]
        val_data = data[n:]
        
        model = BigramLanguageModel()
        m = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        st.markdown('#### Model training has been started, here you can see the stats ####')
        main()

        st.markdown(":green[AI has been trained Click on 'Genrate Chat' to se the AI genrated chat]")
        st.write("#### AI genrated chats")

        genrate = st.button('Genrate Chat')
        
        if genrate:
            st.write(f'''
            ```{decode(m.generate(context, max_new_tokens=chat_lenght)[0].tolist())}```
            ''')


        
