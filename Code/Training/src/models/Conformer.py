import torch
import torch.nn as nn
# import fairseq  # Commented out to avoid dependency issues
from torch.nn.modules.transformer import _get_clones
from torch import Tensor
import os
from transformers import Wav2Vec2Model, Wav2Vec2Config

# Local ConformerBlock implementation
class ConformerBlock(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=16):
        super(ConformerBlock, self).__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_expansion_factor, conv_kernel_size, padding=conv_kernel_size//2),
            nn.GELU(),
            nn.Conv1d(dim * conv_expansion_factor, dim, 1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Convolution (transpose for conv1d)
        conv_input = x.transpose(1, 2)  # (batch, dim, seq_len)
        conv_out = self.conv(conv_input)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, dim)
        
        x = self.norm3(x + conv_out)
        return x
class MyConformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MyConformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.encoder_blocks=_get_clones( ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads= heads, 
    ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
    n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    for layer in self.encoder_blocks:
            x = layer(x) #[bs,1+tiempo,emb_size]
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, embedding

class SSLModel(nn.Module): #W2V
    def __init__(self,device):
        super(SSLModel, self).__init__()
        self.device=device
        self.out_dim = 1024
        
        # Use transformers Wav2Vec2 instead of fairseq
        # Load a pre-trained Wav2Vec2 model (XLSR-53 is similar to XLSR-300M)
        try:
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        except:
            # Fallback to base model if XLSR-300M is not available
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.out_dim = 768  # Base model has 768 dimensions
        
        # Add classification head
        self.classifier = nn.Linear(self.out_dim, 2)
        
        # Move all components to device
        self.model.to(device)
        self.classifier.to(device)
        self.model.train()
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.classifier.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim] - transformers returns last_hidden_state
        with torch.no_grad():
            outputs = self.model(input_tmp)
            emb = outputs.last_hidden_state
        return emb
    
    def forward(self, x):
        # Extract features using the SSL model
        features = self.extract_feat(x.squeeze(-1))  # Remove last dimension if present
        
        # Global average pooling over time dimension
        pooled_features = features.mean(dim=1)  # [batch, out_dim]
        
        # Classification
        logits = self.classifier(pooled_features)  # [batch, 2]
        
        # Return logits and hidden output (features) as expected by trainer
        return logits, pooled_features

class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        
        # Use dynamic input size based on SSL model output
        self.LL = nn.Linear(self.ssl_model.out_dim, 144)
        print('W2V + Conformer')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer=MyConformer(emb_size=144, n_encoders=4,
        heads=4, kernel_size=31)
        
        # Move all components to device
        self.LL.to(device)
        self.first_bn.to(device)
        self.selu.to(device)
        self.conformer.to(device)
    def forward(self, x, Freq_aug=False):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        #out, _ =self.conformer(x,self.device)
        out, emb =self.conformer(x,self.device)
        return out,emb
        #return out

class Model2(nn.Module): #Variable len
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        self.ssl_model = SSLModel(self.device)
        # Use dynamic input size based on SSL model output
        self.LL = nn.Linear(self.ssl_model.out_dim, args.emb_size)
        print('W2V + Conformer: Variable Length')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer=MyConformer(emb_size=args.emb_size, n_encoders=args.num_encoders,
        heads=args.heads, kernel_size=args.kernel_size)
    def forward(self, x): # x is a list of np arrays
        nUtterances = len(x)
        output = torch.zeros(nUtterances, 2).to(self.device)
        for n, feat in enumerate(x):
            input_x = torch.from_numpy(feat[:, :]).float().to(self.device)
            x_ssl_feat = self.ssl_model.extract_feat(input_x.squeeze(-1))
            f=self.LL(x_ssl_feat) 
            f = f.unsqueeze(dim=1)
            f = self.first_bn(f)
            f = self.selu(f)
            f = f.squeeze(dim=1)
            out, _ =self.conformer(f,self.device)
            output[n, :] = out
        return output
