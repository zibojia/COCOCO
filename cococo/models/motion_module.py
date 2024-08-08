from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward

from einops import rearrange, repeat
import math


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels,
    motion_module_type: str, 
    motion_module_kwargs: dict
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs,)    
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        text_cross_attention_dim           = None,
        vision_cross_attention_dim         = None,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 64,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
        layer_id                           = 100,
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            text_cross_attention_dim=text_cross_attention_dim,
            vision_cross_attention_dim=vision_cross_attention_dim,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )
        
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, vision_encoder_hidden_states=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, vision_encoder_hidden_states, attention_mask)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        text_cross_attention_dim           = None,
        vision_cross_attention_dim         = None,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 64,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    text_cross_attention_dim=text_cross_attention_dim,
                    vision_cross_attention_dim=vision_cross_attention_dim,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)    
    
    def forward(self, hidden_states, encoder_hidden_states=None, vision_encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        shape = hidden_states.shape
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, \
                vision_encoder_hidden_states=vision_encoder_hidden_states, video_length=video_length, shape=shape)
        
        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        text_cross_attention_dim           = None,
        vision_cross_attention_dim         = None,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 64,
        layer_id = 100
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for block_name in attention_block_types:
            if 'Self' in block_name:
                attention_block = VersatileSelfAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            elif 'Cross' in block_name:
                attention_block = VersatileCrossAttention(
                    attention_mode=block_name.split("_")[0],
                    text_cross_attention_dim=text_cross_attention_dim if block_name.endswith("_Text_Cross") else None,
                    vision_cross_attention_dim=vision_cross_attention_dim if block_name.endswith("_Vision_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    )
            elif 'Light' in block_name:
                attention_block = LightAttention(
                    attention_mode=block_name.split("_")[0],
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    down_sampling='True' if '_down_sampling' in block_name else 'False',
                    down_resize='True' if '_down_resize' in block_name else 'False',
                    down_flash_attention='True' if '_down_flash_attention' in block_name else 'False',
                    down_resize_conv='True' if '_down_res_conv' in block_name else 'False',
                    down_resize_conv_use_conv='True' if layer_id<2 else 'False',
                    )
            else:
                attention_block = None
            attention_blocks.append(attention_block)
            norms.append(nn.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, vision_encoder_hidden_states=None, attention_mask=None, video_length=None, shape=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                vision_encoder_hidden_states=vision_encoder_hidden_states,
                video_length=video_length,
                shape=shape
            ) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 64
    ):
        #print('pe d_model', d_model, 'max_len', max_len, 4*d_model*max_len/(1024*1024), 'M')
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 64,            
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") else None

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)
            
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states

class VersatileSelfAttention(CrossAttention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 64,            
            *args, **kwargs
        ):

        self.attention_mode = 'Temporal'

        self.is_cross_attention = False

        super().__init__(*args, **kwargs)

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None

    def forward(self, hidden_states, encoder_hidden_states=None, vision_encoder_hidden_states=None, attention_mask=None, video_length=None, shape=None):
        batch_size, sequence_length, _ = hidden_states.shape

        #d = sequence_length
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)


        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        
        
        
        encoder_hidden_states = hidden_states

        query = self.to_q(encoder_hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=sequence_length)

        return hidden_states


class VersatileCrossAttention(CrossAttention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            text_cross_attention_dim           = None,
            vision_cross_attention_dim         = None, 
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 64,            
            *args, **kwargs
        ):

        self.attention_mode = attention_mode
        self.is_text_cross_attention = text_cross_attention_dim is not None
        self.is_vision_cross_attention = vision_cross_attention_dim is not None

        if self.is_vision_cross_attention:
            #print('is_vision_cross_attention')
            self.cross_attention_dim = vision_cross_attention_dim
            kwargs['cross_attention_dim'] = self.cross_attention_dim
        if self.is_text_cross_attention:
            #print('is_text_cross_attention')
            self.cross_attention_dim = text_cross_attention_dim
            kwargs['cross_attention_dim'] = self.cross_attention_dim

        self.is_cross_attention = self.is_text_cross_attention or self.is_vision_cross_attention

        super().__init__(*args, **kwargs)

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") else None

        if self.is_vision_cross_attention:
            self.pos_vision_encoder = PositionalEncoding(
                kwargs['cross_attention_dim'],
                dropout=0., 
                max_len=100
            )  
        else:
            self.pos_vision_encoder = None

    def forward(self, hidden_states, encoder_hidden_states=None, vision_encoder_hidden_states=None, attention_mask=None, video_length=None, shape=None):
        batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        if self.pos_vision_encoder is not None and self.is_vision_cross_attention:
            vision_encoder_hidden_states = self.pos_vision_encoder(vision_encoder_hidden_states)

        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=sequence_length) if self.is_text_cross_attention else None
        vision_encoder_hidden_states = repeat(vision_encoder_hidden_states, "b n c -> (b d) n c", d=sequence_length) if self.is_vision_cross_attention else None

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.is_text_cross_attention:
            #print('is_text_cross_attention')
            encoder_hidden_states = encoder_hidden_states
        elif self.is_vision_cross_attention:
            #print('is_vision_cross_attention')
            encoder_hidden_states = vision_encoder_hidden_states
        else:
            #print('None')
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=sequence_length)

        return hidden_states

class LightAttention(CrossAttention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 64,
            down_sampling='False',
            down_resize='False',
            down_resize_conv='False',
            down_flash_attention='False',
            down_resize_conv_use_conv='False',
            *args, **kwargs
        ):

        print(args, kwargs)

        self.attention_mode = 'Temporal'

        self.is_cross_attention = True

        super().__init__(*args, **kwargs)

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=10000
        )

        kwargs['down_sampling'] = down_sampling
        kwargs['down_resize'] = down_resize
        kwargs['down_flash_attention'] = down_flash_attention
        kwargs['down_resize_conv'] = down_resize_conv
        kwargs['down_resize_conv_use_conv'] = down_resize_conv_use_conv
        self.kwargs = kwargs

    def forward(self, hidden_states, encoder_hidden_states=None, vision_encoder_hidden_states=None, attention_mask=None, video_length=None, shape=None):
        batch_size, sequence_length, c = hidden_states.shape
        batch_size, c, video_length, w, h = shape
        #print(shape)
        encoder_hidden_states = rearrange(hidden_states, "(b f) d c -> b c f d",f=video_length)
        encoder_hidden_states = rearrange(encoder_hidden_states, "b c f (w h) -> b c f w h", h=h, w=w)

        if encoder_hidden_states.shape[-2] > 12 and self.kwargs['down_resize'] == 'True':
             w_,h_ = 12, int(12/w*h)
             encoder_hidden_states = rearrange(encoder_hidden_states, "b c f w h -> (b f) c w h")
             encoder_hidden_states = torch.nn.functional.interpolate(encoder_hidden_states, (w_,h_)).to(encoder_hidden_states.device)
             encoder_hidden_states = rearrange(encoder_hidden_states, "(b f) c w h -> b c f w h", f=video_length)
        else:
            w_ = w
            h_ = h
        encoder_hidden_states = rearrange(encoder_hidden_states, "b c f w h -> b (f w h) c")
        encoder_hidden_states = self.pos_encoder(encoder_hidden_states)

        if self.group_norm is not None:
            encoder_hidden_states = self.group_norm(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(encoder_hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, w_*h_, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "b (f d) c -> b c f d", f=video_length, d=w_*h_)
        hidden_states = rearrange(hidden_states, "b c f (w h) -> (b f) c w h", w=w_, h=h_)
        if w > 12 and self.kwargs['down_resize'] == 'True':
            hidden_states = torch.nn.functional.interpolate(hidden_states, (w, h))

        hidden_states = rearrange(hidden_states, "(b f) c w h -> b c f w h", f=video_length)
        hidden_states = rearrange(hidden_states, "b c f w h -> (b f) (w h) c")

        return hidden_states
