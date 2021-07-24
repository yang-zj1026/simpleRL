# Vanilla Policy Gradient (VPG)

## Quick Facts
+ VPG is an on-policy algorithm.
+ VPG can be used in environments with either discrete or continuous action spaces.

## Key Equations
Let $\pi_{\theta}$ denote a policy with parameters $\theta$, 
and $J(\pi_{\theta})$ denote the expected finite-horizon undiscounted return of the policy. 
The gradient of $J(\pi_{\theta})$ is
$$
\nabla_{\theta} J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta}}{E}\left[{
    \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
    } \right ],
$$
where $\tau$ is a trajectory and $A^{\pi_{\theta}}$ is the advantage function for the current policy.

The policy gradient algorithm works by updating policy parameters via **stochastic gradient ascent** on policy performance:
$$
\theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})
$$

Policy gradient implementations typically compute advantage function estimates based on the infinite-horizon discounted return.

##Usage

+ Train: ``

