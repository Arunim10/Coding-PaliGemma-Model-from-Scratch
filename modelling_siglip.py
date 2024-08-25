from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    
    def __init__(
        self,
        hidden_size=768, # size of embedding vector
        intermediate_size=3072, # size of linear layer in FFN
        num_hidden_layers=12,
        num_attention_layers=12,
        num_attention_heads=12,
        num_channels=3, # RGB
        image_size=224, # PaliGemma comes in 3 sizes (224,448,896)
        patch_size=16, # image divided into patches of 16x16
        layer_norm_eps=1e-6, 
        attention_dropout=0.0,
        num_image_tokens: int = None, # how many output embeddings vit will output
        **kwargs
    ):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size,
            padding="valid" # This indicates no padding is added
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        
        # nn.Embedding Layer is basically used to convert token indices to continuous spaces. nn.Embedding(10,50) This means a weight matrix of shape (10,50) will be initialized
        # In NLP, we represent 10 as vocabulary size and 50 as the embedding dimension of each tokens.
        # For more info, rfer this article https://www.freedium.cfd/https://towardsdatascience.com/the-secret-to-improved-nlp-an-in-depth-look-at-the-nn-embedding-layer-in-pytorch-6e901e193e16
        self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
        
        # Buffer used to register a tensor as buffer and it wont be counted in parameters. 
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)), # tensor of shape (1,num_positions)
            persistent=False # means it wont be saved in model.state_dict. Its used when the tensor can be recomputed whil using again. 
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        
        _,_,height,width = pixel_values.shape # [Batch_Size,channels,height,width]
        
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to kernel size.
        # The output of the convolution will have shape [batch_size,embed_dim,num_patches_h,num_patches_w]
        # where num_patches_h = height//patch_size and num_patches_w = width//patch_Size
        patch_embeds = self.patch_embedding(pixel_values)
        
        # [batch_size, embed_dim,num_patches_h,num_patches_w] -> [batch_Size,embed_dim,num_patches]
        # where num_patches = num_patches_h * num_patches_w
        embeddings = patch_embeds.flatten(2)
        
        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim]
        embeddings = embeddings.transpose(1,2)
        
        # Add position Embeddings to each patch. Each positional encoding is a vector of size [embed_dim]
        # [batch_size, num_patches, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings
        
class SiglipAttention(nn.Module):
    """Multi Head Attention from "Attention is All you need" paper"""
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**(-0.5) # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        
        # query_states: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        
        # key_states: [batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)
        
        # value_states: [batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)
        
        # query_states: [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # key_states: [batch_size, num_heads, num_patches, head_dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # value_states: [batch_size, num_heads, num_patches, head_dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_wts: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is of size \
                {attn_weights.size()}"
            )
        
        # Apply the softmax row-wise. attn_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Multiply the attention weights by the value states. attn_output: [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f" `attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is \
                    {attn_output.size()}"
            )
            
        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()
        
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # [batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output) # combines outputs of all attention heads
        
        return attn_output, attn_weights
        
        
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forwad(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_Size]
        hidden_state = self.fc1(hidden_state)
        
        # hidden_states: [batch_size, num_patches, intermediate_size]
        hidden_state = nn.functional.gelu(hidden_state, approximate='tanh')
        
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_state = self.fc2(hidden_state)
        
        return hidden_state
        
class SiglipEncoderLayer(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
                
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states,_ = self.self_attn(hidden_states=hidden_states)
        
        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states
        
        # residual: [batch_size, num_patches, embed_dim] 
        residual = hidden_states
        
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        
        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states
        
        return hidden_states
        
class SiglipEncoder(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_attention_layers)]
        )
        
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        
        # inputs_embeds: [batch_size, num_patches, embed_dim]
        hidden_states = inputs_embeds
        
        for encoder_layer in self.layers:
            # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
            hidden_states = encoder_layer(hidden_states)
            
        return hidden_states
        
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_valus: [Batch_size, Channels, Height, Width] -> [Batch_size,Num_Patches,Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(inputs_embed=hidden_states)
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
    
class SiglipVisionModel(nn.Module):
    
    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
    

    