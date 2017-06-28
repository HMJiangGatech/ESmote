################################################################################
# Search k nearest neighbors                                                   #
# File:   KNN.R                                                                #
# Author: Haoming Jiang                                                        #                                                         #
# Modified Date:   March 22, 2017                                              #          #
#                                                                              #
################################################################################

#' Search Nearest Neighbors
#'
#' Fast k-nearest neighbor searching algorithms including a kd-tree, cover-tree and the algorithm implemented in class package.
#'
#' @param data an input data matrix.
#' @param query a query data matrix.
#' @param algorithm nearest neighbor searching algorithm.
#' @param k the maximum number of nearest neighbors to search. The default value is set to 5.
#'
#' @details It is based on the package of 'FNN'. \emph{kd tree} is only suitable for low dimensional data (below 20D).
#'
#' The \emph{cover tree} is O(n) space data structure which allows us to answer queries in the same O(log(n)) time as \emph{kd tree} given a fixed intrinsic dimensionality. Templated code from \url{http://hunch.net/~jl/projects/cover_tree/cover_tree.html} is used.
#'
#' The \emph{kd tree} algorithm is implemented in the Approximate Near Neighbor (ANN) C++ library (see  \url{http://www.cs.umd.edu/~mount/ANN/}). The exact nearest neighbors are searched in this package.
#'
#' The \emph{CR} algorithm is the \emph{VR} using distance \emph{1-x'y} assuming \code{x} and \code{y} are unit vectors.
#'
#' The \emph{brute} algorithm searches linearly. It is a naive method.
#'
#' The \emph{rp_forest} algorithm searches approximate k-Nearest Neighbors using random projection trees search. (see \dQuote{Visualizing Large-scale and High-dimensional Data.}
#'
#' @return a list contains:
#' @param nn.index an n x k matrix for the nearest neighbor indice.
#' @param nn.dist (not for \emph{rp_forest}) an n x k matrix for the nearest neighbor Euclidean distances.
#' @param ... some parameters can be setted for \emph{rp_forest} algorithms, see \code{?randomProjectionTreeSearch}
#'
#' @author Haoming Jiang. To report any bugs or suggestions please email: \email{jinghm@mail.ustc.edu.cn}
#'
#' @references
#' Bentley J.L. (1975), \dQuote{Multidimensional binary search trees used for associative search,} \emph{Communication ACM}, \bold{18}, 309-517.
#'
#' Arya S. and Mount D.M. (1993), \dQuote{Approximate nearest neighbor searching,} \emph{Proc. 4th Ann. ACM-SIAM Symposium on Discrete Algorithms (SODA'93)}, 271-280.
#'
#' Arya S., Mount D.M., Netanyahu N.S., Silverman R. and Wu A.Y. (1998), \dQuote{An optimal algorithm for approximate nearest neighbor searching,} \emph{Journal of the ACM}, \bold{45}, 891-923.
#'
#' Beygelzimer A., Kakade S. and Langford J. (2006), \dQuote{Cover trees for nearest neighbor,} \emph{ACM Proc. 23rd international conference on Machine learning}, \bold{148}, 97-104.
#'
#' @export
#' @examples
#' data<- query<- cbind(1:10, 1:10)
#'
#' get.knn(data, k=5)
#' get.knnx(data, query, k=5)
#' get.knnx(data, query, k=5, algo="kd_tree")
#'
#' @useDynLib esmote get_KNN_cover
#' @useDynLib esmote get_KNN_kd
#' @useDynLib esmote get_KNN_CR
#' @useDynLib esmote get_KNN_brute
#' @useDynLib esmote get_KNNX_cover
#' @useDynLib esmote get_KNNX_kd
#' @useDynLib esmote get_KNNX_CR
#' @useDynLib esmote get_KNNX_brute
get.knn<- function (data, k = 5, algorithm=c("kd_tree", "cover_tree", "CR", "brute", "rp_forest"),

                    n_trees = 50,
                    tree_threshold = max(10, min(nrow(data), ncol(data))),
                    max_iter = 1,
                    distance_method = "Euclidean",
                    threads = NULL,
                    verbose = getOption("verbose", TRUE)

                    )
{
  algorithm<- match.arg(algorithm);

  #check data
  if(!is.matrix(data)) data<- as.matrix(data);
  if(!is.numeric(data)) stop("Data non-numeric")
  if(any(is.na(data))) stop("Data include NAs")
  if(storage.mode(data)=="integer") storage.mode(data)<- "double";


  n <- nrow(data);
  d <- ncol(data);

  if(k>=n) warning("k should be less than sample size!");

  if(algorithm == "rp_forest") {
    nn.index=1+t(largeVis::randomProjectionTreeSearch(x = t(data),
                                          n_trees = n_trees,
                                          tree_threshold = tree_threshold,
                                          K = k,
                                          max_iter = max_iter,
                                          distance_method = distance_method,
                                          threads = threads,
                                          verbose = verbose));

    return(list(nn.index=nn.index));
  }


  Cname<- switch(algorithm,
              cover_tree = "get_KNN_cover",
              kd_tree= "get_KNN_kd",
              CR = "get_KNN_CR",
              brute = "get_KNN_brute"
              );
  knnres<- .C(Cname, t(data), as.integer(k), as.integer(d), as.integer(n), nn.index = integer(n*k), nn.dist = double(n*k), DUP=FALSE);

  nn.index<-  matrix(knnres$nn.index, byrow=T, nrow=n, ncol=k);
  nn.dist<- matrix(knnres$nn.dist, byrow=T, nrow=n, ncol=k);

  if(k>=n) {
      nn.index[, n:k]<- NA;
      nn.dist[, n:k]<- NA;
  }

  return(list(nn.index=nn.index, nn.dist=nn.dist));
}


get.knnx<- function (data, query, k = 5, algorithm=c("kd_tree", "cover_tree", "CR", "brute"))
{
  #k neearest neighbor Euclidean distances
  algorithm<- match.arg(algorithm);

  #check data
  if(!is.matrix(data)) data<- as.matrix(data);
  if(!is.numeric(data)) stop("Data non-numeric")
  if(any(is.na(data))) stop("Data include NAs")
  if(storage.mode(data)=="integer") storage.mode(data)<- "double";


  #check query
  if(!is.matrix(query)) query<- as.matrix(query);
  if(!is.numeric(query)) stop("Data non-numeric")
  if(any(is.na(query))) stop("Data include NAs")
  if(storage.mode(query)=="integer") storage.mode(query)<- "double";

  n <- nrow(data); m<- nrow(query);
  d <- ncol(data); p<- ncol(query);

  if(d!=p) stop("Number of columns must be same!.");
  if(k>n) warning("k should be less than sample size!");

  Cname<- switch(algorithm,
                cover_tree = "get_KNNX_cover",
                kd_tree= "get_KNNX_kd",
                CR = "get_KNNX_CR",
                brute = "get_KNNX_brute"
  );
  knnres<- .C(Cname, t(data), t(query), as.integer(k), d, n, m, nn.index = integer(m*k), nn.dist = double(m*k), DUP=FALSE);

  nn.index<- matrix(knnres$nn.index, byrow=T, nrow=m, ncol=k);
  nn.dist<-  matrix(knnres$nn.dist,  byrow=T, nrow=m, ncol=k);
#2012_10_15
#  if(k>=n) {
#    nn.index[, n:k]<- NA;
#    nn.dist[, n:k]<- NA;
#  }
   if (k > n) {
     nn.index[, (n+1):k] <- NA
     nn.dist[, (n+1):k] <- NA
  }
  return(list(nn.index=nn.index, nn.dist=nn.dist));
}
knn.index<- function (data, k = 10, algorithm=c("kd_tree", "cover_tree", "CR", "brute"))
{
  get.knn(data, k, algorithm )$nn.index;
}
knn.dist<- function (data, k = 10, algorithm=c("kd_tree", "cover_tree", "CR", "brute"))
{
  get.knn(data, k, algorithm )$nn.dist;
}
knnx.dist<- function (data, query, k = 10, algorithm=c("kd_tree", "cover_tree", "CR", "brute"))
{
  get.knnx(data, query, k, algorithm )$nn.dist
}
knnx.index<- function (data, query, k = 10, algorithm=c("kd_tree", "cover_tree", "CR", "brute"))
{
  get.knnx(data, query, k, algorithm )$nn.index;
}
