
# De-noising Diffusion Probability Model

Reference: [Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, 6840-6851.](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)

## Methodology

### Inputs

training data: set of $x_0 \in \mathbb{R}^{m\times n}$

variance schedule: $\beta_t$, $t = 1, \ldots, T$

model: $\epsilon_\theta$ - a neural network that has input and out dimensions equal to that of $x_0$

### Definitions

$$\alpha_t = 1 - \beta_t$$

$$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

$$L(\theta) = E\left(\Vert \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon) \Vert^2\right)$$

### Training Algorithm

Repeat until convergence:

Sample $x_0$ from dataset

Sample $t \sim \mathcal{U}\[1,\ldots,T\]$

Sample $\epsilon \sim \mathcal{N}(0,I)$

$$\theta \longleftarrow \theta - \eta \nabla_\theta L(\theta)$$

### Sampling Algorithm

Sample $x_T \sim \mathcal{N}(0,I)$

Repeat until $t=1$:

If $t > 1$, sample $z \sim \mathcal{N}(0,I)$. Otherwise, $z=0$.
 
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t) \right) + \sqrt{\beta_t} z$$

Return $x_0$
