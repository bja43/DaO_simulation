# DaO_simulation
The DAG adaptation of the Onion method

## Example using DaO in Python

```python
p = 10    # number of variables
ad = 4    # average degree
n = 100   # number of samples

g = er_dag(p, ad=ad)
g = sf_out(g)
g = randomize_graph(g)

R, B, O = corr(g)
X = simulate(B, O, n)
X = standardize(X)

df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(p)])
```

## Example using DaO in R

```r
p <- 10    # number of varibales
ad <- 4    # average degree
n <- 100   # number of samples

g <- er_dag(p, ad=ad)
g <- sf_out(g)
g <- randomize_graph(g)

model <- corr(g)

X <- simulate(model$B, model$O, n)
X <- standardize(X)
```
