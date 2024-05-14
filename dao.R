library(Matrix)


er_dag <- function(p, d=0.5, ad=NULL) {
  # Randomly generates an Erdos-Renyi (lower triangular)
  # direct acyclic graph given an ordering.
  # p = |variables|
  # d = |edges| / |possible edges|   (ignored if ad is not NULL)
  # ad = average degree
  
  # npe = |possible edges|
  npe <- as.integer(p * (p - 1) / 2)
  
  # ne = |edges|
  if (!is.null(ad)) {
    d <- ad / (p - 1)
  }
  ne <- as.integer(d * npe)
  
  # generate edges
  e <- c(rep(0, npe - ne), rep(1, ne))
  e <- sample(e)
  
  # generate graph
  g <- matrix(0, nrow=p, ncol=p)
  g[lower.tri(g, diag=FALSE)] <- e
  
  return(g)
}


sf_out <- function(g) {
  # Rewire within rows so row sum (in degree) doesnt change.
  # g = directed acyclic graph

  # p = |variables|
  p <- ncol(g)
  
  for (i in 2 : p) {
    J <- NULL
    for (j in 1 : (i - 1)) {
      J <- c(J, rep(j, 1 + sum(g[1 : (i - 1), j]))) # CHECK THIS
    }
    J <- sample(J)
    
    in_deg <- sum(g[i, ])
    g[i, ] <- rep(0, p)
    
    for (j in J) {
      if (in_deg == 0) break
      if (g[i, j] == 0) {
        in_deg <- in_deg - 1
        g[i, j] <- 1
      }
    }
  }
  
  return(g)
}


sf_in <- function(g) {
  # Rewire within cols so col sum (out degree) doesnt change.
  # g = directed acyclic graph

  # p = |variables|
  p <- ncol(g)
  
  for (i in (p - 1) : 2) {
    J <- NULL
    for (j in 1 : (i - 1)) {
      J <- c(J, 1 + rep(j, sum(g[j, i : p])))
    }
    J <- sample(J)
    
    J <- c(seq_len(p - i) + i, rep(seq_len(p - i), times = rowSums(g[(i + 1):p, i])))

    out_deg <- sum(g[, i])
    g[, i] <- rep(0, p)
    
    for (j in J) {
      if (out_deg == 0) break
      if (g[j, i] == 0) {
        out_deg <- out_deg - 1
        g[j, i] <- 1
      }
    }
  }
  
  return(g)
}


num_source <- function(g) {
  # Helper function: counts the number of source variables.
  # g = directed acyclic graph
  
  # p = |variables|
  p <- ncol(g)
  
  m <- 0
  for(i in 1 : p) {
    if(sum(g[i,]) == 0) {
      m <- m + 1
    }
  }
  
  return(m)
}


sofic_order <- function(g) {
  # Helper function: returns a source first consistent order.
  # g = directed acyclic graph
  
  # convert g to booleans
  g <- g != 0
  
  # p = |variables|
  p <- ncol(g)
  
  ord <- integer(p)
  idx <- 1
  
  for(i in 1 : p) {
    if(sum(g[i,]) == 0) {
      ord[idx] <- i
      idx <- idx + 1
    }
  }
  
  while(idx <= p) {
    for(i in 1 : p) {
      if (i %in% ord) {
        next
      }
      if(sum(g[i,]) == sum(g[i, ord[1 : idx - 1]])) {
        ord[idx] <- i
        idx <- idx + 1
        break
      }
      if (i == p) {
        stop("cycle detected")
      }
    }
  }
  
  return(ord)
}


invert_order <- function(ord) {
  # Helper function: inverts the order.
  # ord = order
  
  # p = |variables|
  p <- length(ord)
  
  inv_ord <- integer(p)
  for (i in 1 : p) {
    inv_ord[ord[i]] <- i
  }
  
  return(inv_ord)
}


mpii <- function(g, i) {
  # Helper function: samples a multivariate Pearson type II.
  # g = (lower triangular) directed acyclic graph
  # i = index
  
  # p = |variables|
  p <- ncol(g)
  
  # d = |output|
  d <- i - 1
  
  # k = |parents|
  k <- sum(g[i,])
  
  # initialize w
  w <- double(d)
  
  # update w
  if (k > 0) {
    q <- rbeta(1, k / 2, (p - i + 1) / 2)
    y <- rnorm(k)
    u <- y / norm(y, type="2")
    w[1:k] <- sqrt(q) * u
  }
  
  return(w)
}


pmat <- function(g, i) {
  # Helper function: returns a permutation matrix.
  # g = (lower triangular) directed acyclic graph
  # i = index

  # d = |output|
  d <- i - 1

  P <- NULL
  for (j in 1 : d) {
    c <- rep(0, d)
    c[j] <- 1
    if (g[i, j]) {
      P <- c(c, P)
    } else {
      P <- c(P, c)
    }
  }
  P <- matrix(P, nrow=d, ncol=d)
  
  return(P)
}


corr <- function(g) {
  # Randomly generates a correlation matrix where f(R) ~ 1
  # given a direct acyclic graph.
  # g = (lower triangular) directed acyclic graph
  
  # reorder g
  ord <- sofic_order(g)
  g <- g[ord, ord]
  
  # p = |variables|; m = |source variables|
  p <- ncol(g)
  m <- num_source(g)
  
  # initialize correlation / coefficient / error matrices
  R <- diag(p)
  B <- matrix(0, nrow=p, ncol=p)
  O <- rep(1, p)

  for (i in m : (p - 1)) {
    P <- pmat(g, i + 1)
    L <- t(chol(t(P) %*% R[1 : i, 1 : i] %*% P))
    w <- mpii(g, i + 1)
    
    r <- P %*% L %*% w
    b <- P %*% t(solve(L)) %*% w
    o <- 1 - sum(w * w)
    
    R[1 : i, i + 1] <- r
    R[i + 1, 1 : i] <- r
    B[i + 1, 1 : i] <- b
    O[i] <- o
  }
  
  # reorder R, B, and O
  ord <- invert_order(ord)
  R <- R[ord, ord]
  B <- B[ord, ord]
  O <- O[ord]

  return(list(R=R, B=B, O=O))
}


cov <- function(g, lb_b=0, ub_b=1, lb_o=1, ub_o=2) {
  # Randomly generates a covariance matrix given a directed acyclic graph.
  # g = directed acyclic graph
  # lb_b = lower bound for beta
  # ub_b = upper bound for beta
  # lb_o = lower bound for omega
  # ub_o = upper bound for omega
  
  # reorder g
  ord <- sofic_order(g)
  g <- g[ord, ord]
  
  # p = |variables|
  p <- ncol(g)
  
  # e = |edges|
  e <- sum(g)
  
  # generate edge weights
  B <- matrix(0, nrow=p, ncol=p)
  B[g == 1] <- sample(c(1, -1), e, TRUE) * runif(e, lb_b, ub_b)
  
  # generate variance terms
  O <- runif(p, lb_o, ub_o)
  
  # calculate covariance
  IB <- solve(diag(p) - B)
  S <- IB %*% diag(O) %*% t(IB)
  
  # reorder S, B, and O
  ord <- invert_order(ord)
  S <- S[ord, ord]
  B <- B[ord, ord]
  O <- O[ord]
  
  return(list(S=S, B=B, O=O))
}


simulate <- function(B, O, n, err=NULL) {
  # Randomly simulates data.
  # B = (lower triangular) beta matrix
  # O = (diagonal entries) error matrix
  # n = sample size
  # err = additive error distribution
  
  # p = |variables|
  p <- ncol(B)
  
  # reorder B and O
  ord <- sofic_order(B)
  B <- B[ord, ord]
  O <- O[ord]
  
  # set default additive error as normal
  if (is.null(err)) {
    err <- function(n, var) rnorm(n, 0, sqrt(var))
  }
  
  # simulate data
  X <- matrix(0, n, p)
  for (i in 1 : p) {
    
    # linear effect
    for (j in which(B[i,] != 0)) {
      X[, i] <- X[, i] + B[i, j] * X[, j]
    }
    
    # additive error
    X[, i] <- X[, i] + err(n, O[i])
  }
  
  # reorder X
  ord <- invert_order(ord)
  X <- X[, ord]
  
  return(X)
}


standardize <- function(X) {
  # Standardize data.
  # X = data
  
  # p = |variables|
  p <- ncol(X)
  
  for (i in 1 : p) {
    X[, i] <- (X[, i] - mean(X[, i])) / sd(X[, i])
  }
  
  return(X)
}


randomize_graph <- function(g) {
  # Randomly reorders the variables of the graph.
  # g = directed acyclic graph
  
  # p = |variables|
  p <- ncol(g)
  
  # randomly reorder g
  ord <- sample(1 : p)
  g <- g[ord, ord]
  
  return(g)
}


cov_to_corr <- function(S) {
  # Rescales covariance to correlation.
  # S = covariance matrix
  
  D <- diag(sqrt(diag(S)))
  ID = solve(D)
  
  return(ID %*% S %*% ID)
}


cov_to_dag <- function(g, S) {
  # Converts covariance to directed acyclic graph parameters.
  # g = directed acyclic graph
  # S = covariacne matrix
  
  # p = |variables|
  p <- ncol(S)
  
  B <- matrix(0, nrow=p, ncol=p)
  O <- diag(S)
  
  for (i in 1 : p) {
    pa <- which(g[i, ] == 1)
    if (length(pa)) {
      yX <- as.matrix(S[i, pa])
      IXX <- solve(S[pa, pa])
      B[i, pa] <- t(yX) %*% IXX
      O[i] <- O[i] - t(yX) %*% IXX %*% yX
    }
  }
  
  return(list(B=B, O=O))
}
