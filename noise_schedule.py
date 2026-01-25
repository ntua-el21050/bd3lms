import abc
import torch
import torch.nn as nn
import numpy as np
import math

def get_noise(config, noise_type=None):
  if noise_type is None:
    noise_type = config.noise.type

  if noise_type == 'loglinear':
    return LogLinearNoise()
  elif noise_type == 'square':
    return ExpNoise(2)
  elif noise_type == 'square_root':
    return ExpNoise(0.5)
  elif noise_type == 'log':
    return LogarithmicNoise()
  elif noise_type == 'cosine':
    return CosineNoise()
  elif noise_type == 'gaussian':
    mu = float(getattr(config.noise, 'mu', 0.5))
    sigma = float(getattr(config.noise, 'sigma', 0.1))
    p_min = float(getattr(config.noise, 'p_min', 1e-5))
    u_eps = float(getattr(config.noise, 'u_eps', 1e-6))
    return GaussianNoise(mu=mu, sigma=sigma, p_min=p_min, u_eps=u_eps)
  else:
    raise ValueError(f'{noise_type} is not a valid noise')

class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """

  def __init__(self):
    super().__init__()
    # Used by Diffusion._sigma_from_p to clamp sigma.
    # Keep these as Python floats so they behave as scalar constants
    # regardless of device.
    self.sigma_min = 0.0
    self.sigma_max = float('inf')
  
  def forward(self, t):
    return self.compute_loss_scaling_and_move_chance(t)

  def _w(self, a_t, da_t, w_type="simple", k = 0):
    l = torch.log(a_t/(1-a_t))
    match w_type:
      case "edm":
        mu = 2.4
        sigma = 2.4
        return (1 / (torch.sqrt(torch.tensor(2 * torch.pi * sigma**2)))) * \
           torch.exp(-((l - mu)**2) / (2 * sigma**2)) * \
           (torch.exp(-l)+0.5**2) / (0.5**2)
      case "iddpm":
        return 1/(torch.cosh(l/2))
      case "sigmoid":
        return 1/(1 + torch.exp(-(-l+k)))
      case "fm":
        return torch.exp(-l/2)
      case "simple":
        return -(1-a_t)/da_t
      case _:
        raise ValueError(f"Unknown w_type: {w_type}")
  
class CosineNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def compute_loss_scaling_and_move_chance(self, t):
    cos = - (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = - (1 - self.eps) * torch.sin(t * torch.pi / 2)
    move_chance = cos + 1
    loss_scaling = sin / (move_chance + self.eps) * torch.pi / 2
    return loss_scaling, move_chance

class ExpNoise(Noise):
  def __init__(self, exp=2, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.exp = exp
  
  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.pow(t, self.exp)
    move_chance = torch.clamp(move_chance, min=self.eps)
    loss_scaling = - (self.exp * torch.pow(t, self.exp-1)) / move_chance
    return loss_scaling, move_chance

class LogarithmicNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
    loss_scaling = - 1 / (move_chance * torch.log(torch.tensor(2.0)) * (1 + t))
    return loss_scaling, move_chance

class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = float(self.total_noise(torch.tensor(1.0)).item())
    self.sigma_min = float((self.eps + self.total_noise(torch.tensor(0.0))).item())

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def compute_loss_scaling_and_move_chance(self, t):
    loss_scaling = - 1 / t
    return loss_scaling, t


class GaussianNoise(Noise):
  """Truncated-Gaussian schedule over move_chance p.

  We interpret input t as u ~ Uniform(0, 1) and map it to
  p in (0, 1) via a truncated Normal(mu, sigma) on [0, 1].

  This ensures p is a valid probability for masking, and provides an
  analytic loss scaling:

    loss_scaling(t) = a'(t) / (1 - a(t)) = -p'(t) / p(t)
  where a(t) = 1 - p(t).
  """

  def __init__(self, mu=0.5, sigma=0.1, p_min=1e-5, u_eps=1e-6):
    super().__init__()
    if sigma <= 0:
      raise ValueError('GaussianNoise requires sigma > 0')
    if not (0.0 < p_min < 0.5):
      raise ValueError('GaussianNoise requires 0 < p_min < 0.5')
    if not (0.0 < u_eps < 0.5):
      raise ValueError('GaussianNoise requires 0 < u_eps < 0.5')
    self.mu = float(mu)
    self.sigma = float(sigma)
    self.p_min = float(p_min)
    self.u_eps = float(u_eps)
    # For sigma = -log(1 - p), the worst case is p -> 1.
    # Clamp via p_min so sigma_max is finite.
    self.sigma_min = float(-math.log(1.0 - self.p_min))
    self.sigma_max = float(-math.log(self.p_min))

  def _phi(self, z):
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

  def _Phi(self, z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

  def _Phi_inv(self, u):
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

  def _p_and_dpdt(self, t):
    # t is u ~ Uniform(0, 1)
    u = t.clamp(self.u_eps, 1.0 - self.u_eps)
    mu = u.new_tensor(self.mu)
    sigma = u.new_tensor(self.sigma)

    alpha = (0.0 - mu) / sigma
    beta = (1.0 - mu) / sigma
    Phi_alpha = self._Phi(alpha)
    Phi_beta = self._Phi(beta)
    Z = (Phi_beta - Phi_alpha).clamp_min(1e-12)

    u_trunc = (Phi_alpha + u * Z).clamp(self.u_eps, 1.0 - self.u_eps)
    z = self._Phi_inv(u_trunc)
    p = (mu + sigma * z).clamp(self.p_min, 1.0 - self.p_min)

    pdf = self._phi(z).clamp_min(1e-12)
    dp_du = sigma * Z / pdf
    return p, dp_du

  def total_noise(self, t):
    p, _ = self._p_and_dpdt(t)
    return -torch.log1p(-p)

  def rate_noise(self, t):
    p, dp_du = self._p_and_dpdt(t)
    return dp_du / (1.0 - p)

  def compute_loss_scaling_and_move_chance(self, t):
    p, dp_du = self._p_and_dpdt(t)
    loss_scaling = -dp_du / p
    return loss_scaling, p