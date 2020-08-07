import torch
import pyro
import pyro.distributions as dist

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def intervention():

    prob_A = torch.tensor([
        .50, # P(A = 'on')
        .50 # P(A = 'off')
    ])

    prob_B = torch.tensor([
        [
            .90, # P(B = 'on' | A = 'on')
            .10  # P(B = 'off' | A = 'on')
        ],
        [
            .20, # P(B = 'on' | A = 'off')
            .80  # P(B = 'off' | A = 'off')
        ]
    ])

    prob_C = torch.tensor([
        [
            [
                .60, # P(C = 'on' | A = 'on', B = 'on')
                .40  # P(C = 'off' | A = 'on', B = 'on')
            ],
            [
                .01, # P(C = 'on' | A = 'on', B = 'off')
                .99  # P(C = 'off' | A = 'on', B = 'off')
            ]
        ],
        [
            [
                .90, # P(C = 'on' | A = 'off', B = 'on')
                .10  # P(C = 'off' | A = 'off', B = 'on')
            ],
            [
                .10, # P(C = 'on' | A = 'off', B = 'off')
                .90  # P(C = 'off' | A = 'off', B = 'off')
            ]
        ]
    ])

    A = pyro.sample('A', dist.Categorical(probs = prob_A))
    B = pyro.sample('S', dist.Categorical(probs = prob_B[A]))
    C = pyro.sample('E', dist.Categorical(probs = prob_C[A][B]))

    return C

if __name__ == '__main__':

    intervened = pyro.do(
        intervention,
        {'B': torch.tensor(0)}
    )
    conditioned = pyro.condition(
        intervened, 
        {'C': torch.tensor(0)}
    )
    posterior = pyro.infer.Importance(conditioned, num_samples = 10000).run()
    marginal = pyro.infer.EmpiricalMarginal(posterior, 'A')
    samples = [marginal() for _ in range(10000)]

    ons = [sample for sample in samples if sample == torch.tensor(0)]
    print(len(ons) / len(samples))

    # Plot the distribution of P(A | O = 'self', R = 'big')
    plt.figure(figsize = (14, 7))
    plt.hist(samples, bins = 'auto')
    plt.xticks([0, 1], ['on', 'off'])
    plt.title('P(A | B = "on", C = "on")')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
