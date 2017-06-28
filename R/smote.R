#' SMOTE (Synthetic Minority Over-sampling TEchnique)
#'
#' Function that implements original SMOTE (Synthetic Minority Over-sampling TEchnique)
#'
#' Via introducing efficient knn algorithms, it can be speed up.
#'
#' @param X The dataset, a numeric matrix, each row is a sample.
#' @param Y A vector, the label of \code{X}. 0 represents minority while 1 represents majority.
#' @param N A the percentage of over-sampling to carry out.
#' @param k The number of nearest neighours to use for the generation.
#' @param algorithm Nearest neighbor searching algorithm. For details, see \code{?get.knn}
#' @param seed The random seed.
#' @param ... some parameters can be setted for \emph{rp_forest} algorithms, see \code{?randomProjectionTreeSearch}
#' @return The generated set exculding \code{X}
#' @examples
#'
#' @export
#' @references
#' Chawla, Nitesh V., et al. \dQuote{SMOTE: synthetic minority over-sampling technique.} arXiv preprint arXiv:1106.1813 (2011).
#'
#' Tang, Jian, et al. \dQuote{Visualizing Large-scale and High-dimensional Data.} \emph{International Conference on World Wide Web}, 2016:287-297.
#'
Smote <- function(X, Y, N=200, k=5, algorithm=c("kd_tree", "cover_tree", "CR", "brute", "rp_forest"), seed=1212,

                  n_trees = 20,
                  tree_threshold = max(10, min(nrow(data), ncol(data))),
                  max_iter = 2,
                  distance_method = "Euclidean",
                  threads = NULL,
                  verbose = getOption("verbose", TRUE)

                  ){
  set.seed(seed)
  if(N <= 100)
    stop("N must greater than 100")
  data   <- X[which(Y==0),]
  rm(X)
  oldN   <- dim(data)[1]
  feaN   <- dim(data)[2]
  # for each old example generate newN examples
  newN   <- as.integer((N-100)/100)
  baseID <- sample(oldN)
  # compensate decimal part
  addN   <- as.integer(oldN * ((N-100)/100 - newN))
  newData<- matrix(nrow = newN*oldN+addN, ncol = feaN)
  cNewID <- 0

  if(algorithm=="rp_forest") {
    knnObj <- get.knn(data, k, algorithm,

                      n_trees = n_trees,
                      tree_threshold = tree_threshold,
                      max_iter = max_iter,
                      distance_method = distance_method,
                      threads = threads,
                      verbose = verbose)
  }  else  {
    knnObj <- get.knn(data, k, algorithm)
  }

  for(i in 1:oldN) {
    cID = baseID[i]
    # calculate KNN for data[i,]
    kNNs <- knnObj$nn.index[cID,]

    repN = newN
    if(i <= addN) repN = repN+1
    for(j in 1:repN){
      # select randomly one of the k NNs
      nnID <- sample(kNNs,1)

      cNewID <- cNewID+1

      # the attribute values of the generated case
      difs <- data[nnID,]-data[cID,]
      newData[cNewID,] <- data[cID,]+runif(feaN)*difs
    }
  }

  newData
}


