# Causal Inference Course: Homework 2
*Dan Saattrup Nielsen, August 2020*

This homework concerns building generative models, both Bayesian networks using `bnlearn` in `R`, and probabilistic programming using `pyro` in `Python`.

Our data is the survey data, containing the following categorical variables:

  - **Age (A)**: The age of the individual, which is *young* (**young**) if they'reless than 30 years old, *adult* (**adult**) if they're between 30 and 60 years old, and *old* (**old**) otherwise
  - **Sex (S)**: The biological sex of the individual, which here is assumed to be binary: *male* (**M**) or *female* (**F**)
  - **Education (E)**: The highest level of education completed by the individual, which can be *high school* (**high**) or *university degree* (**uni**)
  - **Occupation (O)**: Whether the individual is an *employee* (**emp**) or is *self employed* (**self**)
  - **Residence (R)**: The size of the city the individual lives in, which can be either *small* (**small**) or *big* (**big**)
  - **Travel (T)**: The means of transport favoured by the individual, recorded as *car* (**car**), *train* (**train**) or *other* (**other**)

Here travel is the target of the survey. We're using the following DAG as our model of the generative process of the data:

![](img/dag.png)


## Question 1: 
