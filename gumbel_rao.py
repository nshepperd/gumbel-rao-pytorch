"""Implementation of the straight-through gumbel-rao estimator.

Paper: "Rao-Blackwellizing the Straight-Through Gumbel-Softmax
Gradient Estimator" <https://arxiv.org/abs/2010.04838>.

Note: the implementation here differs from the paper in that we DO NOT
propagate gradients through the conditional G_k | D
reparameterization. The paper states that:

    Note that the total derivative d(softmax_τ(θ + Gk))/dθ is taken
    through both θ and Gk. For the case K = 1, our estimator reduces to
    the standard ST-GS estimator.

I believe that this is a mistake - the expectation of this estimator
is only equal to that of ST-GS if the derivative is *not* taken
through G_k, as the ST-GS estimator ∂f(D)/dD d(softmax_τ(θ + G))/dθ
does not.

With the derivative ignored through G_k, the value of this estimator
with k=1 is numerically equal to that of ST-GS, and as k->∞ the
estimator for any given outcome D converges to the expectation of
ST-GS over G conditional on D.

"""

import torch

@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
    + Q) is given by D (one hot vector)."""
    # iid. exponential
    E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)
    # partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted logits
    adjusted = (D * (-torch.log(E) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted - logits

def exact_conditional_gumbel(logits, D, k=1):
    """Same as conditional_gumbel but uses rejection sampling."""
    # Rejection sampling.
    idx = D.argmax(dim=-1)
    gumbels = []
    while len(gumbels) < k:
        gumbel = torch.rand_like(logits).log().neg().log().neg()
        if logits.add(gumbel).argmax() == idx:
            gumbels.append(gumbel)
    return torch.stack(gumbels)



def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()

def gumbel_rao(logits, k, temp=1.0):
    """
    Returns the argmax(input, dim=-1) as a one-hot vector, with
    gumbel-rao gradient.

    k: integer number of samples to use in the rao-blackwellization.
    1 sample reduces to straight-through gumbel-softmax
    """
    num_classes = logits.shape[-1]
    I = torch.distributions.categorical.Categorical(logits=logits).sample()
    D = torch.nn.functional.one_hot(I, num_classes).float()
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
    return replace_gradient(D, surrogate)

# >>> exact_conditional_gumbel(torch.tensor([[1.0,2.0, 3.0]]), torch.tensor([[0.0, 1.0, 0.0]]), k=10000).std(dim=0)
# tensor([[0.9952, 1.2695, 0.8132]])
# >>> conditional_gumbel(torch.tensor([[1.0,2.0, 3.0]]), torch.tensor([[0.0, 1.0, 0.0]]), k=10000).std(dim=0)
# tensor([[0.9905, 1.2951, 0.8148]])