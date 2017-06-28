
#' @useDynLib esmote
#' @importFrom Rcpp sourceCpp

.onAttach <- function(libname, pkgname)
{
  packageStartupMessage("This package came from my final year project. It may be vulnerable. Please use it carefully.")
}

.onUnload <- function (libpath) {
  library.dynam.unload("esmote", libpath)
}
