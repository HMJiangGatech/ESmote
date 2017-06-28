#' Autoencoder
#' Implementation of autoencoder via \code{mxnet} library. It can be used for denoising and dimension reduction.
#'
#'
#'
autoencode <- function(X.train,X.test=NULL,N.hidden=c(100),actType=c("sigmoid","tanh","relu"),
                       lambda,beta,rho,epsilon,optim.method=c("BFGS","L-BFGS-B","CG"),
                       rel.tol=sqrt(.Machine$double.eps),max.iterations=2000,rescale.flag=c(F,T),rescaling.offset=0.001){

  if (is.matrix(X.train)) training.matrix <- X.train else stop("X.train must be a matrix!")
  Ntrain <- nrow(training.matrix)
  Nfeatures <- ncol(training.matrix)

  actType <- match.arg(actType)
  optimizer <- match.arg(optim.method)

  #   Setup the autoencoder's architecture:
  NNLayer_module <- Rcpp::Module( "NNLayer_module", inline::getDynLib("esmote"))
  NNLayer <- NNLayer_module$NNLayer

  nl = length(N.hidden)
  for (l in 1:nl){
    if (l==1 || l==nl) sl[[l]] <- N.input else sl[[l]] <- N.hidden[l-1]
  }
  W <- list()  #create an empty list of weight matrices
  b <- list()  #create an empty list of biases




  }






# autoEncoder <- function(data, nn = c(100), internal_act = 'sigmoid')
# {
#     nfeatures <- nrow(data);
#
#     inLayer <- mxnet::mx.symbol.Variable("data");
#
#     if(length(nn)==1) {
#       middleLayer <- mxnet::mx.symbol.FullyConnected(inLayer, num_hidden = nn[length(nn)]);
#       middleLayer <- mxnet::mx.symbol.Activation(middleLayer, act.type = internal_act);
#       outLayer <-  mxnet::mx.symbol.FullyConnected(middleLayer, num_hidden = nfeatures);
#       outLayer <-  mxnet::mx.symbol.Activation(outLayer, act.type = internal_act);
#     }  else  {
#       newHLayer <- mxnet::mx.symbol.FullyConnected(inLayer, num_hidden = nn[1]);
#       newHLayer <- mxnet::mx.symbol.Activation(newHLayer, act.type = internal_act);
#       for(i in nn[c(-1,-length(nn))]) {
#         newHLayer <- mxnet::mx.symbol.FullyConnected(newHLayer, num_hidden = i);
#         newHLayer <- mxnet::mx.symbol.Activation(newHLayer, act.type = internal_act);
#       }
#
#       middleLayer <- mxnet::mx.symbol.FullyConnected(newHLayer, num_hidden = nn[length(nn)]);
#       middleLayer <- mxnet::mx.symbol.Activation(middleLayer, act.type = internal_act);
#
#       newHLayer <- mxnet::mx.symbol.FullyConnected(middleLayer, num_hidden = nn[length(nn)-1]);
#       newHLayer <- mxnet::mx.symbol.Activation(newHLayer, act.type = internal_act);
#       for(i in nn[length(nn):1][c(-1,-2)]){
#         newHLayer <- mxnet::mx.symbol.FullyConnected(newHLayer, num_hidden = i);
#         newHLayer <- mxnet::mx.symbol.Activation(newHLayer, act.type = internal_act);
#       }
#
#       outLayer <- mxnet::mx.symbol.FullyConnected(newHLayer, num_hidden = nfeatures);
#       outLayer <- mxnet::mx.symbol.Activation(outLayer, act.type = internal_act);
#     }
#
#     inLayercp <- mxnet::mx.symbol.Variable("label");
#     lro <- mxnet::mx.symbol.MakeLoss(mx.symbol.sum(mxnet::mx.symbol.square(outLayer - inLayer)));
#
#     mx.set.seed(0)
#     model <- mxnet::mx.model.FeedForward.create(lro, X=as.matrix(tdata[,-785]), y=as.matrix(rep(0,60000),ncol=1),
#                                          ctx=mxnet::mx.cpu(),     num.round=50, array.batch.size=20,
#                                          learning.rate=2e-6, momentum=0.9,  eval.metric=mxnet::mx.metric.rmse, array.layout="rowmajor")
# }
