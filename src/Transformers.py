# For understanding about the code done here, one can check out the Jupyter Notebook attached in the same repository.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Embedding (nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
    def forward (self, token):
        embedding = self.embedding(token)
        return embedding
    
class PositionalEncoding(nn.Module):
    def __init__(self,max_seq_len,embedding_dim):
        super(PositionalEncoding,self).__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len,self.embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embedding_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embedding_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embedding_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, token):
        token = token * math.sqrt(self.embedding_dim)
        seq_len = token.size(1)
        token = token + self.pe[:, :seq_len]
        return token
    
class SelfAttention(nn.Module):
    def __init__(self, d_model, single_head_dim):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.single_head_dim = single_head_dim
        
        self.q_linear = nn.Linear(d_model, single_head_dim)
        self.k_linear = nn.Linear(d_model, single_head_dim)
        self.v_linear = nn.Linear(d_model, single_head_dim)

    def forward(self, inputs, mask=None):
        q = self.q_linear(inputs) 
        k = self.k_linear(inputs)
        v = self.v_linear(inputs)
        matmul = torch.matmul(q, k.transpose(-2, -1))
        s_scaled = matmul / math.sqrt(float(self.d_model))  
        if mask is not None:
            s_scaled = s_scaled.masked_fill(mask == 0, float("-1e20")) 
        w = F.softmax(s_scaled, dim=-1)  
        output = torch.matmul(w, v)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.single_head_dim = embedding_dim // n_heads
        self.attention_heads = nn.ModuleList([SelfAttention(self.embedding_dim,self.single_head_dim) for _ in range(n_heads)])
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input, mask=None):
        head_outputs = []
        for attention_head in self.attention_heads:
            head_output = attention_head(input, mask=mask)
            head_outputs.append(head_output)
        concatenated_heads = torch.cat(head_outputs, dim=-1)  
        output = self.out(concatenated_heads)
        return output

class EncoderUnit(nn.Module):
    def __init__(self, embedding_dim, expansion_factor, n_heads):
        super(EncoderUnit,self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(embedding_dim,n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, expansion_factor*embedding_dim), 
                                nn.ReLU(),
                                nn.Linear(expansion_factor*embedding_dim, embedding_dim)
                            )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,input,mask=None):
        attention_out = self.MultiHeadAttention(input)  
        attention_residual_out = attention_out + input
        norm1_out = self.dropout1(self.norm1(attention_residual_out))
        feed_fwd_out = self.ffn(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))
        return norm2_out

class Encoder(nn.Module):
    def __init__(self, num_units, embedding_dim, expansion_factor, n_heads):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderUnit(embedding_dim, expansion_factor, n_heads) for _ in range(num_units)
        ])
        self.norm = nn.LayerNorm(embedding_dim) 
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderUnit(nn.Module):
    def __init__(self, embedding_dim, expansion_factor, n_heads):
        super(DecoderUnit, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, n_heads)
        self.enc_dec_attention = MultiHeadAttention(embedding_dim, n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, expansion_factor * embedding_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embedding_dim, embedding_dim)
        )

    def forward(self, input, enc_output, tgt_mask, src_mask=None):
        self_attention_out = self.self_attention(input, mask=tgt_mask)
        self_attention_residual_out = self_attention_out + input
        norm1_out = self.dropout1(self.norm1(self_attention_residual_out))
        enc_dec_attention_out = self.enc_dec_attention(norm1_out, mask=src_mask)
        enc_dec_attention_residual_out = enc_dec_attention_out + norm1_out
        norm2_out = self.dropout2(self.norm2(enc_dec_attention_residual_out))
        ffn_out = self.ffn(norm2_out)
        ffn_residual_out = ffn_out + norm2_out
        norm3_out = self.dropout3(self.norm3(ffn_residual_out))
        return norm3_out

class Decoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, expansion_factor, n_heads):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderUnit(embedding_dim, expansion_factor, n_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input, enc_output, tgt_mask, src_mask=None):
        for layer in self.layers:
            output = layer(input, enc_output, src_mask, tgt_mask)
        return self.norm(output)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_layers, expansion_factor, n_heads):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, embedding_dim)
        self.encoder = Encoder(num_layers, embedding_dim, expansion_factor, n_heads)
        self.decoder = Decoder(num_layers, embedding_dim, expansion_factor, n_heads)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, tgt_mask , src_mask=None):
        src_emb = self.embedding(src)
        src_pos_enc = self.positional_encoding(src_emb)
        src_encoding = src_emb + src_pos_enc
        tgt_emb = self.embedding(tgt)
        tgt_pos_enc = self.positional_encoding(tgt_emb)
        tgt_encoding = tgt_emb + tgt_pos_enc
        enc_output = self.encoder(src_encoding, src_mask)
        dec_output = self.decoder(tgt_encoding, enc_output, src_mask, tgt_mask)
        output = self.linear(dec_output)
        output = self.softmax(output)
        return output