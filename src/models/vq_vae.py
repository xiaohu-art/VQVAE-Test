import sys
import torch

from einops import rearrange
from torch import nn

from torch.nn import functional as F

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.modules.encoder import Encoder
from src.modules.decoder import Decoder
from src.models.model_utils import get_model_params
from src.local_vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


class VQVAE(nn.Module):
  def __init__(self, args):
    super().__init__()
    channels, resolution, z_channels, embed_dim, n_embed = get_model_params(args.dataset, args.f)
    self.args = args
    self.num_codes = n_embed
    self.cosine = (args.codebook == 'cosine')
    decay = 0.8  # Default value.

    self.encoder = Encoder(ch=8, in_channels=channels, z_channels=z_channels, double_z=False)
    self.decoder = Decoder(ch=8, out_channels=channels, z_channels=z_channels, double_z=False)

    # Note: ema_update=True and learnable_codebook=False, so will use ema updates to learn codebook vectors.
    self.vq = VectorQuantize(dim=embed_dim, 
                             codebook_size=n_embed,
                             commitment_weight=args.commit_weight, 
                             decay=decay,
                             accept_image_fmap=True, 
                             use_cosine_sim=(args.codebook == 'cosine'),
                             threshold_ema_dead_code=0)

    # Set up projections into and out of codebook.
    if args.codebook == 'cosine':
      self.pre_quant_proj = nn.Sequential(nn.Linear(z_channels, embed_dim),
                                          nn.LayerNorm(embed_dim)) if embed_dim != z_channels else nn.Identity()
    else:
      self.pre_quant_proj = nn.Sequential(
        nn.Linear(z_channels, embed_dim)) if embed_dim != z_channels else nn.Identity()
    self.post_quant_proj = nn.Linear(embed_dim, z_channels) if embed_dim != z_channels else nn.Identity()

  def get_codes(self, x):
    # Encode.
    x = self.encoder(x)
    x = rearrange(x, 'b c h w -> b h w c')
    x = self.pre_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')

    # VQ lookup.
    quantized, indices, _ = self.vq(x)
    return indices

  def decode(self, indices):
    q = self.vq.get_codes_from_indices(indices)
    if self.cosine:
      q = q / torch.norm(q, dim=1, keepdim=True)

    # Decode.
    x = self.post_quant_proj(q)
    x = rearrange(x, 'b (h w) c -> b c h w', h=16)
    x = self.decoder(x)
    return x

  def encode_forward(self, x):
    # Encode.
    x = self.encoder(x)
    x = rearrange(x, 'b c h w -> b h w c')
    x = self.pre_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')

    # VQ lookup.
    quantized, indices, _ = self.vq(x)
    return quantized

  def decoder_forward(self, q):
    if self.cosine:
      q = q / torch.norm(q, dim=1, keepdim=True)

    # Decode.
    x = rearrange(q, 'b c h w -> b h w c')
    x = self.post_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')
    x = self.decoder(x)
    return x

  @staticmethod
  def get_very_efficient_rotation(e_hat, q_hat, e):
    r = ((e_hat + q_hat) / torch.norm(e_hat + q_hat, dim=1, keepdim=True)).detach()
    e = e - 2 * torch.bmm(torch.bmm(e, r.unsqueeze(-1)), r.unsqueeze(1)) + 2 * torch.bmm(
      torch.bmm(e, e_hat.unsqueeze(-1).detach()), q_hat.unsqueeze(1).detach())
    return e

  def forward(self, x, rot=False):
    init_x = x
    # Encode.
    x = self.encoder(x)
    x = rearrange(x, 'b c h w -> b h w c')
    x = self.pre_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')

    # ViT-VQGAN codebook: "We also apply l2 normalization on the encoded latent variables ze(x)
    # and codebook latent variables e."
    if self.cosine:
      x = x / torch.norm(x, dim=1, keepdim=True)

    e = x
    # VQ lookup.
    quantized, indices, commit_loss = self.vq(x)
    q = quantized

    # If using the rotation trick.
    if rot:
      b, c, h, w = x.shape
      x = rearrange(x, 'b c h w -> (b h w) c')
      quantized = rearrange(quantized, 'b c h w -> (b h w) c')
      pre_norm_q = self.get_very_efficient_rotation(x / (torch.norm(x, dim=1, keepdim=True) + 1e-6),
                                                    quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                                                    x.unsqueeze(1)).squeeze()
      quantized = pre_norm_q * (
              torch.norm(quantized, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-6)).detach()
      quantized = rearrange(quantized, '(b h w) c -> b c h w', b=b, h=h, w=w)

    if self.cosine:
      quantized = quantized / torch.norm(quantized, dim=1, keepdim=True)

    # Use codebook ema: no embed loss.
    # emb_loss = F.mse_loss(quantized, x.detach())

    # Decode.
    x = rearrange(quantized, 'b c h w -> b h w c')
    x = self.post_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')
    x = self.decoder(x)
    rec = x
    rec_loss = F.mse_loss(init_x, x)

    return rec_loss, commit_loss
