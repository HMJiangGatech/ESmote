
#'Visualize the digits. It turns a vector with 784 dimensions into a 28*28 pickture. It needs 'softmaxreg' package to be installed.
#'
#'@param x A vector with 784 dimensions.
#'@export
visulizeDigits <- function(x)
{
  if (!requireNamespace("softmaxreg", quietly = TRUE)) {
    stop("\'softmaxreg\' needed for this function to work. Please install it.",
         call. = FALSE)
  }
  softmaxreg::show_digit(x);
}
