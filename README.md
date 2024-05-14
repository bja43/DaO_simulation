# DaO_simulation
The DAG adaptation of the Onion method


## Example using DaO in R

```r
g <- er_dag(10, ad=4)
g <- sf_rewire_out(g)
g <- randomize_graph(g)

model <- corr(g)

X <- simulate(model$B, model$O, 100)
X <- standardize(X)
```
