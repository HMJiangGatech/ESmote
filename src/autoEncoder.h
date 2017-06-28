#ifndef esmote_autoencoder
#define esmote_autoencoder
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <math.h>
#include "utilsC.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::mat sigmoidAct(const arma::mat& X)
{
  arma::mat Y;
  Y = 1.0/(1 + arma::exp(-X));
  return Y;
}

//' It's for backpropagation, the input X is actFun(X)
// [[Rcpp::export]]
arma::mat sigmoidGrad(const arma::mat& X)
{
  arma::mat Y;
  Y = X - arma::square(X);
  return Y;
}

// [[Rcpp::export]]
arma::mat reluAct(const arma::mat& X)
{
  arma::mat Y(arma::size(X));
  Y = arma::clamp( X, 0, arma::datum::inf);
  return Y;
}

//' It's for backpropagation, the input X is actFun(X)
// [[Rcpp::export]]
arma::mat reluGrad(const arma::mat& X)
{
  arma::mat Y(arma::size(X));
  Y = arma::clamp( X, 0, arma::datum::inf);
  Y = sign(Y);
  return Y;
}

// [[Rcpp::export]]
arma::mat tanhAct(const arma::mat& X)
{
  arma::mat Y;
  Y = arma::tanh(X);
  return X;
}

//' It's for backpropagation, the input x is actFun(x)
// [[Rcpp::export]]
arma::mat tanhGrad(const arma::mat& X)
{
  arma::mat Y;
  Y = 1-arma::square(X);
  return Y;
}

// U_n = Y_(n-1)*W_n+1*b_n  sigma(U_n) = Y_n
class NNLayer
{
public:
  arma::mat* weight;
  arma::rowvec* bias;
  NNLayer* preLayer;
  NNLayer* nextLayer;
  int LayerType; // Decoder Layer: 0 Encoder Layer: 1
  String actName; // must be one of "sigmoid", "relu", "tanh"
  arma::mat (*actFun)(const arma::mat& x);
  arma::mat (*actGrad)(const arma::mat& x);

  // used for training
  arma::mat fwOutput;  //the Y_n in the formular
  arma::mat *fwInput;  //the Y_(n-1) in the formular
  arma::mat *weightVel;    // used for Momentum Methods
  arma::rowvec *biasVel;   // used for Momentum Methods


  // constructer
  NNLayer(int iNode, int oNode, double epsilon, String actName); // epsilon is the range of initial parameters
  NNLayer(int iNode, int oNode, String actName); // automatically assign the initial weight
  void initialParams(int iNode, int oNode, String actName);

  // modify methods
  void setNextL(NNLayer& nextL) {this->nextLayer = &nextL;}
  void setPreL (NNLayer& preL)  {this->preLayer  = &preL; }

  // methods
  List getweight();
  arma::mat feedforward(arma::mat &data, bool isTraining); // compute feedforward output up to this layer.

private:
};

class NNet // autoencoder network
{
public:
  // data
  arma::mat   data;
  arma::ivec  labels;       // 0 fr minority 1 for majority
  arma::ivec  trainlabels;  // label info for training
  arma::vec   priorProb;    // label info for training
  double      datamin;
  double      datamax;

  // auxilary parameters
  int         nInput;
  int         nRecord;
  int         nLayers;
  String      actName;
  arma::vec   nHidden;
  double      lambda;     // used for L2-norm regularizer term (train mode 1)
  double      alpha;      // used for Probability Loss term (train mode 2)

  //network
  NNLayer*    firstLayer;
  NNLayer*    lastLayer;
  NNLayer*    midLayer;

  // constructer
  NNet(arma::mat data, arma::vec nHidden, double epsilon, String actName);
  NNet(arma::mat data, arma::vec nHidden, String actName); // automatically assign the initial weight
  void initialParams(arma::mat &data, arma::vec nHidden, String actName);
  void setLabels(arma::ivec labels);

  // methods
  List getweight();
  void recoverWei(List wei);
  arma::mat feedforward(arma::mat &data, bool isTraining);

  // backprop methods
  void backprop2(NNLayer* cLayer, arma::mat &G1, arma::mat &G2, double learnRate);  // for train mode 2
  void backprop(NNLayer* cLayer, arma::mat &G, double learnRate, int trainMode);
  void prebackprop(NNLayer* cLayer, arma::mat &G, double learnRate, NNLayer* trainLayer, int trainMode);
  void backpropM(NNLayer* cLayer, arma::mat &G, double learnRate, double momentum, int trainMode); // momentum method
  void backpropLauncher(arma::mat &data, double learnRate, int trainMode, double momentum, bool ispretrain, NNLayer* trainLayer); // trainMode: 0 only L2 loss

  // training methods: mode (0: L2Loss, 1: L2Loss+L2Reg, 2: L2Loss+L2Reg+ProbLoss )
  //                   lambda is for L2Reg; alpha is for ProbLoss
  void pretrain (double learnRate, int iterations, int trainMode, double lambda, double alpha);
  void bgdtrain (double learnRate, int iterations, int trainMode, double lambda, double alpha);
  void sgdtrain (double learnRate, int iterations, int batchSize, int trainMode, double lambda, double alpha);
  void sgdMtrain(double learnRate, int iterations, int batchSize, double momentum, int trainMode, double lambda, double alpha);

  double squareLoss(arma::mat &data);
  double squareReg ();
  arma::mat midFeature(arma::mat &data){ return this->midLayer ->feedforward(data, false); };
  arma::mat netOutput (arma::mat &data);

private:

};

RCPP_MODULE(NNLayer_module)
{
  class_<NNLayer>("NNLayer")
  .constructor<int,int,double,String>()
  .constructor<int,int,String>()
  .method("getweight", &NNLayer::getweight)
  ;
  class_<NNet>("NNet")
    .constructor<arma::mat,arma::vec,double,String>()
    .constructor<arma::mat,arma::vec,String>()
    .method("getweight",  &NNet::getweight)
    .method("recoverWei", &NNet::recoverWei)
    .method("setLabels",  &NNet::setLabels)
    .method("pretrain",   &NNet::pretrain)
    .method("bgdtrain",   &NNet::bgdtrain)
    .method("sgdtrain",   &NNet::sgdtrain)
    .method("sgdMtrain",  &NNet::sgdMtrain)
    .method("midFeature", &NNet::midFeature)
    .method("netOutput",  &NNet::netOutput)
  ;
}

//NNLayer_module <- Rcpp::Module( "NNLayer_module", inline::getDynLib("esmote"))
//NNLayer <- NNLayer_module$NNLayer
//l1 <- new(NNLayer,3,5,1,"tanh")

#endif
