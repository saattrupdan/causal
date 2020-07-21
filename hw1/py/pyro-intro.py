import torch
import pyro
import pyro.distributions as dist

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def survey():

    prob_A = torch.tensor([
        .36, # P(A = 'adult')
        .16, # P(A = 'old')
        .48  # P(A = 'young')
    ])

    prob_S = torch.tensor([
        .55, # P(S = 'F')
        .45  # P(A = 'M')
    ])

    prob_E = torch.tensor([
        [
            [
                .64, # P(E = 'high' | A = 'adult', S = 'F')
                .36  # P(E = 'uni' | A = 'adult', S = 'F')
            ],
            [
                .72, # P(E = 'high' | A = 'adult', S = 'M')
                .28  # P(E = 'uni' | A = 'adult', S = 'M')
            ]
        ],
        [
            [
                .84, # P(E = 'high' | A = 'old', S = 'F')
                .16  # P(E = 'uni' | A = 'old', S = 'F')
            ],
            [
                .89, # P(E = 'high' | A = 'old', S = 'M')
                .11  # P(E = 'uni' | A = 'old', S = 'M')
            ]
        ],
        [
            [
                .16, # P(E = 'high' | A = 'young', S = 'F')
                .84  # P(E = 'uni' | A = 'young', S = 'F')
            ],
            [
                .81, # P(E = 'high' | A = 'young', S = 'M')
                .19  # P(E = 'uni' | A = 'young', S = 'M')
            ]
        ]
    ])

    prob_O = torch.tensor([
        [
            .98, # P(O = 'emp' | E = 'high')
            .02  # P(O = 'self' | E = 'high')
        ],
        [
            .97, # P(O = 'emp' | E = 'uni')
            .03  # P(O = 'self' | E = 'uni')
        ]
    ])

    prob_R = torch.tensor([
        [
            .72, # P(R = 'big' | E = 'high')
            .28  # P(R = 'small' | E = 'high')
        ],
        [
            .94, # P(R = 'big' | E = 'uni')
            .06  # P(R = 'small' | E = 'uni')
        ]
    ])

    prob_T = torch.tensor([
        [
            [
                .71, # P(T = 'car'   | R = 'big', O = 'emp')
                .14, # P(T = 'other' | R = 'big', O = 'emp')
                .15  # P(T = 'train' | R = 'big', O = 'emp')
            ],
            [
                .69, # P(T = 'car'   | R = 'big', O = 'self')
                .16, # P(T = 'other' | R = 'big', O = 'self')
                .16  # P(T = 'train' | R = 'big', O = 'self')
            ]
        ],
        [
            [
                .55, # P(T = 'car'   | R = 'small', O = 'emp')
                .08, # P(T = 'other' | R = 'small', O = 'emp')
                .38  # P(T = 'train' | R = 'small', O = 'emp')
            ],
            [
                .73, # P(T = 'car'   | R = 'small', O = 'self')
                .25, # P(T = 'other' | R = 'small', O = 'self')
                .02  # P(T = 'train' | R = 'small', O = 'self')
            ]
        ]
    ])

    A = pyro.sample('A', dist.Categorical(probs = prob_A))
    S = pyro.sample('S', dist.Categorical(probs = prob_S))
    E = pyro.sample('E', dist.Categorical(probs = prob_E[A][S]))
    O = pyro.sample('O', dist.Categorical(probs = prob_O[E]))
    R = pyro.sample('R', dist.Categorical(probs = prob_R[E]))
    T = pyro.sample('T', dist.Categorical(probs = prob_T[R][O]))

    return T

if __name__ == '__main__':
    conditioned = pyro.condition(
        survey, 
        {'O': torch.tensor(1), 'R': torch.tensor(0)}
    )
    posterior = pyro.infer.Importance(conditioned, num_samples = 1000).run()
    marginal = pyro.infer.EmpiricalMarginal(posterior, 'A')
    samples = [marginal() for _ in range(1000)]

    # Plot the distribution of P(A | O = 'self', R = 'big')
    plt.figure(figsize = (14, 7))
    plt.hist(samples, bins = 'auto')
    plt.xticks([0, 1, 2], ['adult', 'old', 'young'])
    plt.title('P(A | O = "self", R = "big")')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
