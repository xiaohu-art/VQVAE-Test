from functools import partial
from collections import namedtuple

import torch
from torch.nn import Module
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from einops import rearrange, repeat, reduce, pack, unpack

from typing import Callable


def exists(val):
  return val is not None


def default(val, d):
  return val if exists(val) else d


def noop(*args, **kwargs):
  pass


def identity(t):
  return t


def l2norm(t):
  return F.normalize(t, p=2, dim=-1)


def Sequential(*modules):
  modules = [*filter(exists, modules)]
  if len(modules) == 0:
    return None
  elif len(modules) == 1:
    return modules[0]

  return nn.Sequential(*modules)


def cdist(x, y):
  x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
  y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
  xy = einsum('b i d, b j d -> b i j', x, y) * -2
  return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min=0).sqrt()


def log(t, eps=1e-20):
  return torch.log(t.clamp(min=eps))


def entropy(prob, eps=1e-5):
  return (-prob * log(prob, eps=eps)).sum(dim=-1)


def ema_inplace(old, new, decay):
  is_mps = str(old.device).startswith('mps:')

  if not is_mps:
    old.lerp_(new, 1 - decay)
  else:
    old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
  return pack([t], pattern)


def unpack_one(t, ps, pattern):
  return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
  t = torch.empty(shape)
  nn.init.kaiming_uniform_(t)
  return t


def gumbel_noise(t):
  noise = torch.zeros_like(t).uniform_(0, 1)
  return -log(-log(noise))


def gumbel_sample(
        logits,
        temperature=1.,
        stochastic=False,
        straight_through=False,
        reinmax=False,
        dim=-1,
        training=True
):
  dtype, size = logits.dtype, logits.shape[dim]

  if training and stochastic and temperature > 0:
    sampling_logits = (logits / temperature) + gumbel_noise(logits)
  else:
    sampling_logits = logits

  ind = sampling_logits.argmax(dim=dim)
  one_hot = F.one_hot(ind, size).type(dtype)

  assert not (
            reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'

  if not straight_through or temperature <= 0. or not training:
    return ind, one_hot

  # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
  # algorithm 2

  if reinmax:
    π0 = logits.softmax(dim=dim)
    π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
    π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
    π2 = 2 * π1 - 0.5 * π0
    one_hot = π2 - π2.detach() + one_hot
  else:
    π1 = (logits / temperature).softmax(dim=dim)
    one_hot = one_hot + π1 - π1.detach()

  return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
  denom = x.sum(dim=dim, keepdim=True)
  return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
  num_samples, device = samples.shape[0], samples.device
  if num_samples >= num:
    indices = torch.randperm(num_samples, device=device)[:num]
  else:
    indices = torch.randint(0, num_samples, (num,), device=device)

  return samples[indices]


def batched_sample_vectors(samples, num):
  return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
  return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
  device = probs.device
  probs = probs.cpu()

  total_count = probs.new_full((), total_count)
  remainder = probs.new_ones(())
  sample = torch.empty_like(probs, dtype=torch.long)

  for i, p in enumerate(probs):
    s = torch.binomial(total_count, p / remainder)
    sample[i] = s
    total_count -= s
    remainder -= p

  assert total_count == 0, f'invalid total count {total_count}'

  return sample.to(device)


def all_gather_sizes(x, dim):
  size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
  all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
  distributed.all_gather(all_sizes, size)
  return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
  rank = distributed.get_rank()
  all_x = []

  for i, size in enumerate(sizes):
    t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
    distributed.broadcast(t, src=i, async_op=True)
    all_x.append(t)

  distributed.barrier()
  return all_x


def batched_bincount(x, *, minlength):
  batch, dtype, device = x.shape[0], x.dtype, x.device
  target = torch.zeros(batch, minlength, dtype=dtype, device=device)
  values = torch.ones_like(x)
  target.scatter_add_(-1, x, values)
  return target


def kmeans(
        samples,
        num_clusters,
        num_iters=10,
        use_cosine_sim=False,
        sample_fn=batched_sample_vectors
):
  num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

  means = sample_fn(samples, num_clusters)

  for _ in range(num_iters):
    if use_cosine_sim:
      dists = samples @ rearrange(means, 'h n d -> h d n')
    else:
      dists = -cdist(samples, means)

    buckets = torch.argmax(dists, dim=-1)
    bins = batched_bincount(buckets, minlength=num_clusters)

    zero_mask = bins == 0
    bins_min_clamped = bins.masked_fill(zero_mask, 1)

    new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

    new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
    new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')

    if use_cosine_sim:
      new_means = l2norm(new_means)

    means = torch.where(
      rearrange(zero_mask, '... -> ... 1'),
      means,
      new_means
    )

  return means, bins


def batched_embedding(indices, embeds):
  batch, dim = indices.shape[1], embeds.shape[-1]
  indices = repeat(indices, 'h b n -> h b n d', d=dim)
  embeds = repeat(embeds, 'h c d -> h b c d', b=batch)
  return embeds.gather(2, indices)


# regularization losses

def orthogonal_loss_fn(t):
  # eq (2) from https://arxiv.org/abs/2112.00384
  h, n = t.shape[:2]
  normed_codes = l2norm(t)
  cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
  return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)


# distance types

class EuclideanCodebook(Module):
  def __init__(
          self,
          dim,
          codebook_size,
          num_codebooks=1,
          kmeans_init=False,
          kmeans_iters=10,
          sync_kmeans=True,
          decay=0.8,
          eps=1e-5,
          threshold_ema_dead_code=2,
          reset_cluster_size=None,
          learnable_codebook=False,
          gumbel_sample=gumbel_sample,
          sample_codebook_temp=1.,
          ema_update=True,
          affine_param=False,
          sync_affine_param=False,
          affine_param_batch_decay=0.99,
          affine_param_codebook_decay=0.9
  ):
    super().__init__()
    self.transform_input = identity

    self.decay = decay
    self.ema_update = ema_update

    init_fn = uniform_init if not kmeans_init else torch.zeros
    embed = init_fn(num_codebooks, codebook_size, dim)

    self.codebook_size = codebook_size
    self.num_codebooks = num_codebooks

    self.kmeans_iters = kmeans_iters
    self.eps = eps
    self.threshold_ema_dead_code = threshold_ema_dead_code
    self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

    assert callable(gumbel_sample)
    self.gumbel_sample = gumbel_sample
    self.sample_codebook_temp = sample_codebook_temp

    self.register_buffer('initted', torch.Tensor([not kmeans_init]))
    self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
    self.register_buffer('embed_avg', embed.clone())

    self.learnable_codebook = learnable_codebook
    if learnable_codebook:
      self.embed = nn.Parameter(embed)
    else:
      self.register_buffer('embed', embed)

    # affine related params

    self.affine_param = affine_param
    self.sync_affine_param = sync_affine_param

    if not affine_param:
      return

    self.affine_param_batch_decay = affine_param_batch_decay
    self.affine_param_codebook_decay = affine_param_codebook_decay

    self.register_buffer('batch_mean', None)
    self.register_buffer('batch_variance', None)

    self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
    self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
    self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
    self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

  @torch.jit.ignore
  def init_embed_(self, data):
    if self.initted:
      return

    embed, cluster_size = kmeans(
      data,
      self.codebook_size,
      self.kmeans_iters,
      sample_fn=batched_sample_vectors
    )

    embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

    self.embed.data.copy_(embed)
    self.embed_avg.data.copy_(embed_sum)
    self.cluster_size.data.copy_(cluster_size)
    self.initted.data.copy_(torch.Tensor([True]))

  @torch.jit.ignore
  def update_with_decay(self, buffer_name, new_value, decay):
    old_value = getattr(self, buffer_name)

    needs_init = getattr(self, buffer_name + "_needs_init", False)

    if needs_init:
      self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

    if not exists(old_value) or needs_init:
      self.register_buffer(buffer_name, new_value.detach())

      return

    value = old_value * decay + new_value.detach() * (1 - decay)
    self.register_buffer(buffer_name, value)

  @torch.jit.ignore
  def update_affine(self, data, embed):
    assert self.affine_param

    var_fn = partial(torch.var, unbiased=False)

    # calculate codebook mean and variance

    embed = rearrange(embed, 'h ... d -> h (...) d')

    if self.training:
      self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
      self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn),
                             self.affine_param_codebook_decay)

    # prepare batch data, which depends on whether it has masking

    data = rearrange(data, 'h ... d -> h (...) d')

    # calculate batch mean and variance

    if not self.sync_affine_param:
      self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
      self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
      return

    num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

    # number of vectors, for denominator

    num_vectors = torch.tensor([num_vectors], device=device, dtype=dtype)

    # calculate distributed mean

    batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
    batch_mean = batch_sum / num_vectors

    self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)

    # calculate distributed variance

    variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
    batch_variance = variance_numer / num_vectors

    self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

  def replace(self, batch_samples, batch_mask):
    for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
      sampled = batched_sample_vectors(rearrange(samples, '... -> 1 ...'), mask.sum().item())
      sampled = rearrange(sampled, '1 ... -> ...')

      self.embed.data[ind][mask] = sampled
      self.cluster_size.data[ind][mask] = self.reset_cluster_size
      self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

  def expire_codes_(self, batch_samples):
    if self.threshold_ema_dead_code == 0:
      return

    expired_codes = self.cluster_size < self.threshold_ema_dead_code

    if not torch.any(expired_codes):
      return

    batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
    self.replace(batch_samples, batch_mask=expired_codes)

  @autocast(enabled=False)
  def forward(
          self,
          x
  ):
    needs_codebook_dim = x.ndim < 4
    sample_codebook_temp = self.sample_codebook_temp

    x = x.float()

    if needs_codebook_dim:
      x = rearrange(x, '... -> 1 ...')

    dtype = x.dtype
    flatten, ps = pack_one(x, 'h * d')

    self.init_embed_(flatten)

    if self.affine_param:
      self.update_affine(flatten, self.embed)

    embed = self.embed if self.learnable_codebook else self.embed.detach()

    if self.affine_param:
      codebook_std = self.codebook_variance.clamp(min=1e-5).sqrt()
      batch_std = self.batch_variance.clamp(min=1e-5).sqrt()
      embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

    dist = -cdist(flatten, embed)

    embed_ind, embed_onehot = self.gumbel_sample(dist, dim=-1, temperature=sample_codebook_temp, training=self.training)

    embed_ind = unpack_one(embed_ind, ps, 'h *')

    if self.training:
      unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
      quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
    else:
      quantize = batched_embedding(embed_ind, embed)

    if self.training and self.ema_update:

      if self.affine_param:
        flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

      cluster_size = embed_onehot.sum(dim=1)

      ema_inplace(self.cluster_size.data, cluster_size, self.decay)

      embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
      embed_sum = embed_sum.contiguous()

      ema_inplace(self.embed_avg.data, embed_sum, self.decay)

      cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim=-1,
                                                                                                                keepdim=True)

      embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
      self.embed.data.copy_(embed_normalized)
      self.expire_codes_(x)

    if needs_codebook_dim:
      quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

    dist = unpack_one(dist, ps, 'h * d')

    return quantize, embed_ind, dist


class CosineSimCodebook(Module):
  def __init__(
          self,
          dim,
          codebook_size,
          num_codebooks=1,
          kmeans_init=False,
          kmeans_iters=10,
          sync_kmeans=True,
          decay=0.8,
          eps=1e-5,
          threshold_ema_dead_code=2,
          reset_cluster_size=None,
          learnable_codebook=False,
          gumbel_sample=gumbel_sample,
          sample_codebook_temp=1.,
          ema_update=True,
  ):
    super().__init__()
    self.transform_input = l2norm

    self.ema_update = ema_update
    self.decay = decay

    if not kmeans_init:
      embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
    else:
      embed = torch.zeros(num_codebooks, codebook_size, dim)

    self.codebook_size = codebook_size
    self.num_codebooks = num_codebooks

    self.kmeans_iters = kmeans_iters
    self.eps = eps
    self.threshold_ema_dead_code = threshold_ema_dead_code
    self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

    assert callable(gumbel_sample)
    self.gumbel_sample = gumbel_sample
    self.sample_codebook_temp = sample_codebook_temp

    self.register_buffer('initted', torch.Tensor([not kmeans_init]))
    self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
    self.register_buffer('embed_avg', embed.clone())

    self.learnable_codebook = learnable_codebook
    if learnable_codebook:
      self.embed = nn.Parameter(embed)
    else:
      self.register_buffer('embed', embed)

  @torch.jit.ignore
  def init_embed_(self, data):
    if self.initted:
      return

    embed, cluster_size = kmeans(
      data,
      self.codebook_size,
      self.kmeans_iters,
      use_cosine_sim=True,
      sample_fn=batched_sample_vectors
    )

    embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

    self.embed.data.copy_(embed)
    self.embed_avg.data.copy_(embed_sum)
    self.cluster_size.data.copy_(cluster_size)
    self.initted.data.copy_(torch.Tensor([True]))

  def replace(self, batch_samples, batch_mask):
    batch_samples = l2norm(batch_samples)

    for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
      sampled = batched_sample_vectors(rearrange(samples, '... -> 1 ...'), mask.sum().item())
      sampled = rearrange(sampled, '1 ... -> ...')

      self.embed.data[ind][mask] = sampled
      self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
      self.cluster_size.data[ind][mask] = self.reset_cluster_size

  def expire_codes_(self, batch_samples):
    if self.threshold_ema_dead_code == 0:
      return

    expired_codes = self.cluster_size < self.threshold_ema_dead_code

    if not torch.any(expired_codes):
      return

    batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
    self.replace(batch_samples, batch_mask=expired_codes)

  @autocast(enabled=False)
  def forward(
          self,
          x
  ):
    needs_codebook_dim = x.ndim < 4
    sample_codebook_temp = self.sample_codebook_temp

    x = x.float()

    if needs_codebook_dim:
      x = rearrange(x, '... -> 1 ...')

    dtype = x.dtype

    flatten, ps = pack_one(x, 'h * d')

    self.init_embed_(flatten)

    embed = self.embed if self.learnable_codebook else self.embed.detach()

    dist = einsum('h n d, h c d -> h n c', flatten, embed)

    embed_ind, embed_onehot = self.gumbel_sample(dist, dim=-1, temperature=sample_codebook_temp, training=self.training)
    embed_ind = unpack_one(embed_ind, ps, 'h *')

    if self.training:
      unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
      quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
    else:
      quantize = batched_embedding(embed_ind, embed)

    if self.training and self.ema_update:

      bins = embed_onehot.sum(dim=1)

      ema_inplace(self.cluster_size.data, bins, self.decay)

      embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
      embed_sum = embed_sum.contiguous()

      ema_inplace(self.embed_avg.data, embed_sum, self.decay)

      cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim=-1,
                                                                                                                keepdim=True)

      embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
      embed_normalized = l2norm(embed_normalized)

      self.embed.data.copy_(l2norm(embed_normalized))
      self.expire_codes_(x)

    if needs_codebook_dim:
      quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

    dist = unpack_one(dist, ps, 'h * d')
    return quantize, embed_ind, dist


# main class
class VectorQuantize(Module):
  def __init__(
          self,
          dim,
          codebook_size,
          decay=0.8,
          eps=1e-5,
          kmeans_init=False,
          kmeans_iters=10,
          sync_kmeans=True,
          use_cosine_sim=False,
          threshold_ema_dead_code=0,
          channel_last=True,
          accept_image_fmap=False,
          commitment_weight=1.,
          commitment_use_cross_entropy_loss=False,
          stochastic_sample_codes=False,
          sample_codebook_temp=1.,
          straight_through=False,
          reinmax=False,  # using reinmax for improved straight-through, assuming straight through helps at all
          sync_affine_param=False,
          ema_update=True,
          learnable_codebook=False,
          # Optimizer used to update the codebook embedding if using learnable_codebook
          affine_param=False,
          affine_param_batch_decay=0.99,
          affine_param_codebook_decay=0.9,
          sync_update_v=0.,
          # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
  ):
    super().__init__()
    self.dim = dim
    self.heads = 1

    self.eps = eps

    self.commitment_weight = commitment_weight
    self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss  # whether to use cross entropy loss to codebook as commitment loss

    self.learnable_codebook = learnable_codebook

    # assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

    assert 0 <= sync_update_v <= 1.
    assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'

    self.sync_update_v = sync_update_v

    codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

    gumbel_sample_fn = partial(
      gumbel_sample,
      stochastic=stochastic_sample_codes,
      reinmax=reinmax,
      straight_through=straight_through
    )

    codebook_kwargs = dict(
      dim=dim,
      num_codebooks=1,
      codebook_size=codebook_size,
      kmeans_init=kmeans_init,
      kmeans_iters=kmeans_iters,
      sync_kmeans=sync_kmeans,
      decay=decay,
      eps=eps,
      threshold_ema_dead_code=threshold_ema_dead_code,
      learnable_codebook=learnable_codebook,
      sample_codebook_temp=sample_codebook_temp,
      gumbel_sample=gumbel_sample_fn,
      ema_update=ema_update
    )

    if affine_param:
      assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
      codebook_kwargs = dict(
        **codebook_kwargs,
        affine_param=True,
        sync_affine_param=sync_affine_param,
        affine_param_batch_decay=affine_param_batch_decay,
        affine_param_codebook_decay=affine_param_codebook_decay,
      )

    self._codebook = codebook_class(**codebook_kwargs)

    self.codebook_size = codebook_size

    self.accept_image_fmap = accept_image_fmap
    self.channel_last = channel_last

    self.register_buffer('zero', torch.tensor(0.), persistent=False)

  @property
  def codebook(self):
    codebook = self._codebook.embed
    return rearrange(codebook, '1 ... -> ...')

  @codebook.setter
  def codebook(self, codes):
    codes = rearrange(codes, '... -> 1 ...')
    self._codebook.embed.copy_(codes)

  def get_codes_from_indices(self, indices):
    codebook = self.codebook

    codes = codebook[indices]

    if not self.channel_last:
      codes = rearrange(codes, 'b ... d -> b d ...')

    return codes

  def forward(
          self, x
  ):
    shape, device = x.shape, x.device

    need_transpose = not self.channel_last and not self.accept_image_fmap

    # rearrange inputs

    if self.accept_image_fmap:
      height, width = x.shape[-2:]
      x = rearrange(x, 'b c h w -> b (h w) c')

    if need_transpose:
      x = rearrange(x, 'b d n -> b n d')

    # l2norm for cosine sim, otherwise identity

    x = self._codebook.transform_input(x)

    # quantize

    quantize, embed_ind, distances = self._codebook(x)

    # losses for loss breakdown

    commit_loss = self.zero

    if self.training:
      # determine code to use for commitment loss
      maybe_detach = torch.detach if not self.learnable_codebook else identity

      commit_quantize = maybe_detach(quantize)

      # straight through

      quantize = x + (quantize - x).detach()

      if self.sync_update_v > 0.:
        # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
        quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

    # function for calculating cross entropy loss to distance matrix
    # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

    def calculate_ce_loss(codes):
      dist_einops_eq = '1 b n l -> b l n'

      ce_loss = F.cross_entropy(
        rearrange(distances, dist_einops_eq, b=shape[0]),
        codes,
        ignore_index=-1
      )

      return ce_loss

    # transform embedding indices

    if self.accept_image_fmap:
      embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)

    # aggregate loss

    loss = torch.tensor([0.], device=device, requires_grad=self.training)

    if self.training:

      # commitment loss
      if self.commitment_use_cross_entropy_loss:
        commit_loss = calculate_ce_loss(embed_ind)
      else:
        commit_loss = F.mse_loss(commit_quantize, x)

      loss = loss + commit_loss * self.commitment_weight

    # rearrange quantized embeddings

    if need_transpose:
      quantize = rearrange(quantize, 'b n d -> b d n')

    if self.accept_image_fmap:
      quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)

    return quantize, embed_ind, loss
