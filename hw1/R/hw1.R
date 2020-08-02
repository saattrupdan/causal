library(bnlearn)
library(Rgraphviz)
library(dplyr)

# 1.3
dag <- bnlearn::model2network('[A][S][E|A:S][O|E][R|E][T|O:R]')

# 1.4
class(dag)

# 1.5
graphviz.plot(dag)

# 2.1
bnlearn::nodes(dag)
bnlearn::arcs(dag)

# 2.2
bnlearn::parents(dag, 'A')
bnlearn::children(dag, 'A')

bnlearn::parents(dag, 'E')
bnlearn::children(dag, 'E')

bnlearn::parents(dag, 'O')
bnlearn::children(dag, 'O')
bnlearn::parents(dag, 'R')
bnlearn::children(dag, 'R')

bnlearn::parents(dag, 'S')
bnlearn::children(dag, 'S')

bnlearn::parents(dag, 'T')
bnlearn::children(dag, 'T')

# 2.3
bnlearn::mb(dag, 'A')
bnlearn::mb(dag, 'E')
bnlearn::mb(dag, 'T')

# 3.1
df <- read.table('../data/survey2.txt', header = TRUE)
dag.fitted <- bnlearn::bn.fit(dag, df, method = 'bayes')

# 3.2
dag.fitted.newprior <- bnlearn::bn.fit(dag, df, method = 'bayes', iss = 10)

dag.fitted['A']
dag.fitted.newprior['A']
dplyr::count(df, A)

dag.fitted['O']
dag.fitted.newprior['O']
dplyr::count(df, O)

# 4.1
dag2 <- bnlearn::drop.arc(dag, 'E', 'O')
graphviz.plot(dag2)

# 4.2
dag2.fitted <- bnlearn::bn.fit(dag2, df, method = 'bayes')

dag.fitted['S']
dag2.fitted['S']

# 5.1
cpdag <- bnlearn::cpdag(dag)
cpdag

# 5.2
dag3 <- bnlearn::set.arc(dag, 'O', 'R')
graphviz.plot(dag3)

cpdag3 <- bnlearn::cpdag(dag3)
cpdag3
graphviz.plot(cpdag3)

# 5.3
dag4 <- bnlearn::set.arc(dag, 'R', 'O')
graphviz.plot(dag4)

cpdag4 <- bnlearn::cpdag(dag4)
graphviz.plot(cpdag4)

# 5.4
bnlearn::score(dag3, df)
bnlearn::score(dag4, df)


dag.fitted
