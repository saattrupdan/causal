library(bnlearn)
library(dplyr)
library(Rgraphviz)

# Define the DAG
net <- bnlearn::model2network('[A][S][E|A:S][O|E][R|E][T|O:R]')
graphviz.plot(net)

# Load the data
df <- read.table('../data/survey.txt', header = TRUE)

# Create set `arg_sets` of all node combinations and potential blocking node sets
vars <- nodes(net)
pairs <- combn(x = vars, 2, list)
arg_sets <- list()
for(pair in pairs){
  others <- setdiff(vars, pair)
  conditioning_sets <- unlist(lapply(0:4, function(.x) combn(others, .x, list)), recursive = F)
  for(set in conditioning_sets){
    args <- list(x = pair[1], y = pair[2], z = set)
    arg_sets <- c(arg_sets, list(args))
  }
}

# Define convenient d-separation function
d_sep <- bnlearn:::dseparation

# 1.1
dseps <- list()
for(args in arg_sets){
  if (d_sep(bn = net, x = args$x, y = args$y, z = args$z) == TRUE){
    dseps <- c(dseps, list(args))
  }
}
length(dseps)

# 1.3
dseps.nonredundant <- list()
for(args in dseps){
  nonredundant <- TRUE
  for(z in args$z){
    z.removed <- args$z[args$z != z]
    if(d_sep(bn = net, x = args$x, y = args$y, z = z.removed) == TRUE){
      nonredundant <- FALSE
      break
    }
  }
  if(nonredundant == TRUE){
    dseps.nonredundant <- c(dseps.nonredundant, list(args))
  }
}
dseps.nonredundant

# 1.5
dseps.ci <- list()
for(args in dseps){
  indep <- bnlearn::ci.test(x = args$x, y = args$y, z = args$z, data = df)
  if(indep$p.value > 0.05){
    dseps.ci <- c(dseps.ci, list(args))
  }
}
dseps.proportion <- length(dseps.ci) / length(dseps)
dseps.proportion

# 1.6
dseps.nonredundant.ci <- list()
for(args in dseps.nonredundant){
  indep <- ci.test(x = args$x, y = args$y, z = args$z, data = df)
  if(indep$p.value > 0.05){
    dseps.nonredundant.ci <- c(dseps.nonredundant.ci, list(args))
  }
}
dseps.nonredundant.proportion <- length(dseps.nonredundant.ci) / length(dseps.nonredundant)
dseps.nonredundant.proportion

# 2.1
cis <- list()
for(args in dseps){
  indep <- ci.test(x = args$x, y = args$y, z = args$z, data = df)
  if(indep$p.value > 0.05){
    cis <- c(cis, list(args))
  }
}
length(cis)

# 2.2
cis.dsep <- list()
for(args in cis){
  if(d_sep(bn = net, x = args$x, y = args$y, z = args$z) == TRUE){
    cis.dsep <- c(cis.dsep, list(args))
  }
}
cis.dsep.proportion <- length(cis.dsep) / length(cis)
cis.dsep.proportion

# 2.3
cis.dsep.nonredundant_proportion <- length(intersect(cis.dsep, dseps.nonredundant)) / length(cis.dsep)
cis.dsep.nonredundant_proportion

# 3.0
net <- model2network('[A][B|A][C|B:A]')
alias <- c('off', 'on')
cptA <- matrix(c(0.5, 0.5), ncol=2)
dimnames(cptA) <- list(NULL, alias)
cptB <- matrix(c(.8, .2, .1, .9), ncol=2)
dimnames(cptB) <- list(B = alias, A = alias)
cptC <- matrix(c(.9, .1, .99, .01, .1, .9, .4, .6))
dim(cptC) <- c(2, 2, 2)
dimnames(cptC) <- list(C = alias, A = alias, B = alias)
data <- list(A = cptA, B = cptB, C = cptC)
model <- custom.fit(net, data)
graphviz.plot(model)

# 3.2
rbns <- bnlearn::rbn(model, n = 1000) %>%
        dplyr::filter(B == 'on', C == 'on')
nrow(rbns %>% dplyr::filter(A == 'on')) / nrow(rbns)

# 3.3
net.mutilated <- bnlearn::mutilated(net, list(B='on'))
graphviz.plot(net.mutilated)

# 3.5
