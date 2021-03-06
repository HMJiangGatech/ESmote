% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/smote.R
\name{Smote}
\alias{Smote}
\title{SMOTE (Synthetic Minority Over-sampling TEchnique)}
\usage{
Smote(X, Y, N = 200, k = 5, algorithm = c("kd_tree", "cover_tree", "CR",
  "brute", "rp_forest"), seed = 1212, n_trees = 50,
  tree_threshold = max(10, min(nrow(data), ncol(data))), max_iter = 1,
  distance_method = "Euclidean", threads = NULL,
  verbose = getOption("verbose", TRUE))
}
\arguments{
\item{X}{The dataset, a numeric matrix, each row is a sample.}

\item{Y}{A vector, the label of \code{X}. 0 represents minority while 1 represents majority.}

\item{N}{A the percentage of over-sampling to carry out.}

\item{k}{The number of nearest neighours to use for the generation.}

\item{algorithm}{Nearest neighbor searching algorithm. For details, see \code{?get.knn}}

\item{seed}{The random seed.}

\item{...}{some parameters can be setted for \emph{rp_forest} algorithms, see \code{?randomProjectionTreeSearch}}
}
\value{
The generated set exculding \code{X}
}
\description{
Function that implements original SMOTE (Synthetic Minority Over-sampling TEchnique)
}
\details{
Via introducing efficient knn algorithms, it can be speed up.
}
\examples{

}
\references{
Chawla, Nitesh V., et al. \dQuote{SMOTE: synthetic minority over-sampling technique.} arXiv preprint arXiv:1106.1813 (2011).

Tang, Jian, et al. \dQuote{Visualizing Large-scale and High-dimensional Data.} \emph{International Conference on World Wide Web}, 2016:287-297.
}
