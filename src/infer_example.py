import torch
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC, EmpiricalMarginal
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def scale(guess: float):
    weight = pyro.sample('weight', dist.Normal(guess, 1.))
    measurement = pyro.sample('measurement', dist.Normal(weight, .75))
    return measurement

def scale_infer(measurement: float, guess_prior: float):

    conditioned_scale = pyro.condition(
        scale, 
        {'measurement': torch.tensor(measurement)}
    )

    guess_prior = torch.tensor(guess_prior)
    nuts_kernel = NUTS(conditioned_scale, adapt_step_size = True)
    posterior = MCMC(nuts_kernel, num_samples = 1000, warmup_steps = 300)
    posterior.run(guess_prior)
    marginal = posterior.get_samples()['weight']

    plt.figure(figsize = (14, 7))
    plt.hist(marginal)
    plt.title('P(weight | measurement = 14)')
    plt.xlabel('Weight')
    plt.ylabel('#')
    plt.show()

if __name__ == '__main__':
    scale_infer(measurement = 14., guess_prior = 13.)
