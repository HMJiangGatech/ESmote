#include "autoEncoder.h"

// NNLayer

NNLayer::NNLayer(int iNode, int oNode, double epsilon, String actName)
{
  this->initialParams(iNode, oNode, actName);

  this->weight->randu();
  *this->weight-=0.5;
  *this->weight*=2*epsilon;
  this->bias->randu()*epsilon;
  *this->bias-=0.5;
  *this->bias*=2*epsilon;

  if(actName == "sigmoid")  {
    actFun = sigmoidAct;
    actGrad = sigmoidGrad;
  } else if(actName == "tanh"){
    actFun = tanhAct;
    actGrad = tanhGrad;
  } else if(actName == "relu"){
    actFun = reluAct;
    actGrad = reluGrad;
  }

}

NNLayer::NNLayer(int iNode, int oNode, String actName)
{
  this->initialParams(iNode, oNode, actName);
  double epsilon;

  if(actName == "sigmoid")  {
    actFun = sigmoidAct;
    actGrad = sigmoidGrad;
    epsilon = sqrt(6.0 / (iNode + oNode));
    this->weight->randu();
    *this->weight-=0.5;
    *this->weight*=2*epsilon;
    this->bias->zeros();
  }
  else if(actName == "tanh"){
    actFun = tanhAct;
    actGrad = tanhGrad;
    epsilon = 4.0*sqrt(6.0 / (iNode + oNode));
    this->weight->randu();
    *this->weight-=0.5;
    *this->weight*=2*epsilon;
    this->bias->zeros();
  }
  else if(actName == "relu"){
    actFun = reluAct;
    actGrad = reluGrad;
    epsilon = sqrt(2.0 / iNode);
    weight->randn()*epsilon;
    this->bias->zeros();
  }

}

void NNLayer::initialParams(int iNode, int oNode, String actName)
{
  if(actName != "sigmoid" && actName != "tanh" && actName != "relu")
    stop("activation have to be sigmoid, tanh or relu");
  this->actName = actName;
  this->weight = new arma::mat(iNode,oNode);
  this->bias = new arma::rowvec(oNode);
  this->nextLayer = NULL;
  this->preLayer = NULL;
  this->weightVel = new arma::mat(iNode,oNode);
  weightVel->zeros();
  this->biasVel = new arma::rowvec(oNode);
  biasVel->zeros();
}


List NNLayer::getweight()
{
  return List::create(Named("WeiMat") = *(this->weight),
                      Named("BiasVec") = *(this->bias));
}

arma::mat NNLayer::feedforward(arma::mat &data, bool isTraining = false)
{
  arma::mat output;
  if(this->preLayer == NULL)
  {
    output = data * (*(this->weight));
    //Rcout<<endl;
  }
  else
    output = this->preLayer->feedforward(data, isTraining) * (*(this->weight));
  output.each_row() += *this->bias;
  output = this->actFun(output);

  if(isTraining)
  {
    this->fwOutput = output;
    if(this->preLayer == NULL)
      this->fwInput = &data;
    else
      this->fwInput = &(this->preLayer->fwOutput);
  }

  //Rcout<<".";
  return output;
}

// NNet

NNet::NNet(arma::mat data, arma::vec nHidden, double epsilon, String actName)
{
  this->initialParams(data, nHidden, actName);

  int nIn, nOut;
  NNLayer *newLayer;
  nIn = this->nInput;
  nOut= nHidden[0];
  newLayer = new NNLayer(nIn, nOut, epsilon, actName);
  this->firstLayer = newLayer;
  newLayer->LayerType = 1;
  for(int i = 1; i<nHidden.n_elem; i++)
  {
    nIn = nOut;
    nOut = nHidden[i];
    newLayer->nextLayer = new NNLayer(nIn, nOut, epsilon, actName);
    newLayer->nextLayer->preLayer = newLayer;
    newLayer = newLayer->nextLayer;
    newLayer->LayerType = 1;
  }
  this->midLayer = newLayer;
  for(int i = nHidden.n_elem-2; i>=0; i--)
  {
    nIn = nOut;
    nOut = nHidden[i];
    newLayer->nextLayer = new NNLayer(nIn, nOut, epsilon, actName);
    newLayer->nextLayer->preLayer = newLayer;
    newLayer = newLayer->nextLayer;
    newLayer->LayerType = 0;
  }
  newLayer->nextLayer = new NNLayer(nOut, this->nInput, epsilon, actName);
  newLayer->nextLayer->preLayer = newLayer;
  newLayer = newLayer->nextLayer;
  this->lastLayer = newLayer;
  newLayer->LayerType = 0;
}


NNet::NNet(arma::mat data, arma::vec nHidden, String actName)
{
  this->initialParams(data, nHidden, actName);

  int nIn, nOut;
  NNLayer *newLayer;
  nIn = this->nInput;
  nOut= nHidden[0];
  newLayer = new NNLayer(nIn, nOut, actName);
  this->firstLayer = newLayer;
  newLayer->LayerType = 1;
  for(int i = 1; i<nHidden.n_elem; i++)
  {
    nIn = nOut;
    nOut = nHidden[i];
    newLayer->nextLayer = new NNLayer(nIn, nOut, actName);
    newLayer->nextLayer->preLayer = newLayer;
    newLayer = newLayer->nextLayer;
    newLayer->LayerType = 1;
  }
  this->midLayer = newLayer;
  for(int i = nHidden.n_elem-2; i>=0; i--)
  {
    nIn = nOut;
    nOut = nHidden[i];
    newLayer->nextLayer = new NNLayer(nIn, nOut, actName);
    newLayer->nextLayer->preLayer = newLayer;
    newLayer = newLayer->nextLayer;
    newLayer->LayerType = 0;
  }
  newLayer->nextLayer = new NNLayer(nOut, this->nInput, actName);
  newLayer->nextLayer->preLayer = newLayer;
  newLayer = newLayer->nextLayer;
  this->lastLayer = newLayer;
  newLayer->LayerType = 0;
}

void NNet::initialParams(arma::mat &data, arma::vec nHidden, String actName)
{

  this->nInput  = data.n_cols;
  this->nRecord = data.n_rows;
  this->data    = data;
  this->nLayers = 2*nHidden.n_elem-1;
  this->nHidden = nHidden;

  this->datamin = this->data.min();
  this->datamax = this->data.max();

  if(actName == "sigmoid")
  {
    this->data -= datamin;
    this->data /= (datamax-datamin);
  }
  else if(actName == "tanh")
  {
    this->data -= datamin;
    this->data /= (datamax-datamin);
    this->data -= 0.5;
    this->data *= 2;
  }

}

void NNet::setLabels(arma::ivec labels)
{
  this->labels = labels;
  arma::vec priorProb(labels.max()-labels.min()+1);
  for(int i = labels.min(); i <= labels.max(); i++)
  {
    priorProb[i-labels.min()] = arma::sum(labels == i)*(1.0/labels.n_elem);
    Rcout<<priorProb[i-labels.min()]<<endl;
  }
  this->labels = this->labels - this->labels.min();
  this->priorProb = priorProb;
}

List NNet::getweight()
{
  List allWei;
  NNLayer *cL;
  cL = this->firstLayer;
  while(cL != NULL)
  {
    allWei.push_back(cL->getweight());
    cL = cL->nextLayer;
  }
  return allWei;
}

void NNet::recoverWei(List wei)
{
  NNLayer *cL;
  List cWei;
  cL = this->firstLayer;
  int i = 0;
  SEXP newWeiSEXP;
  SEXP newBiasSEXP;
  while(cL != NULL)
  {
    cWei = wei.at(i);
    i++;
    newWeiSEXP  = cWei[cWei.findName("WeiMat")];
    newBiasSEXP = cWei[cWei.findName("BiasVec")];
    //Rcpp::NumericMatrix newWei(newWeiSEXP);
    //Rcpp::NumericVector newBias(newBiasSEXP);
    //arma::mat newWeiArma;
    //arma::rowvec newBiasArma;
    //newWeiArma = as<arma::mat>(newWeiSEXP);
    //newBiasArma = as<arma::rowvec>(newBiasSEXP);
    *(cL->weight) = as<arma::mat>(newWeiSEXP);
    *(cL->bias)   = as<arma::rowvec>(newBiasSEXP);
    cL = cL->nextLayer;
  }
}

arma::mat NNet::feedforward(arma::mat &data, bool isTraining)
{
  return lastLayer->feedforward(data, isTraining);
}

double NNet::squareLoss(arma::mat &data)
{
  return 1.0*arma::sum(arma::sum(arma::square(this->feedforward(data, false) - data), 1))/data.n_rows;
}

double NNet::squareReg()
{
  double reg_wei = 0;

  NNLayer *cL;
  cL = this->firstLayer;
  while(cL != NULL)
  {
    reg_wei += arma::sum(arma::sum(arma::square(*cL->weight)));
    cL = cL->nextLayer;
  }
  return reg_wei;
}

void NNet::prebackprop(NNLayer* cLayer, arma::mat &G, double learnRate, NNLayer* trainLayer, int trainMode) // trainMode: 0 only L2 loss
{
  //Rcout<<".";
  if(cLayer == trainLayer)
  {
    if(trainMode == 0)
      *(cLayer->weight) -= learnRate*(cLayer->fwInput->t() * G);
    else if(trainMode == 1 || trainMode == 2)
      *(cLayer->weight) -= learnRate*((cLayer->fwInput->t() * G) + (this->lambda) * *(cLayer->weight));
    *(cLayer->bias) -= learnRate*arma::sum(G,0);
    return;
  }
  if(cLayer->preLayer != NULL)
  {
    if( trainMode == 2 && cLayer->preLayer == this->midLayer)
    {
      G = (G * cLayer->weight->t());
      arma::mat Gp(size(G));
      Gp.zeros();
      for(int i = 0; i < Gp.n_rows; i++)
        for(int j = 0; j < Gp.n_rows; j++)
        if(i != j)
        {
          arma::rowvec deltay;
          deltay = cLayer->fwInput->row(i) - cLayer->fwInput->row(j);
          double sqnorm_deltay;
          sqnorm_deltay = arma::norm(deltay, 2);
          sqnorm_deltay = sqnorm_deltay*sqnorm_deltay;
          if(this->trainlabels[i] != this->trainlabels[j])
            Gp.row(i) -= (deltay)*(2.0/sqnorm_deltay/(1+sqnorm_deltay));
          else
            Gp.row(i) += (deltay)*(2.0/(1+sqnorm_deltay));
        }
      Gp = Gp*(1.0*this->alpha/Gp.n_rows/Gp.n_rows);
      G = G + Gp;
      G = G % cLayer->preLayer->actGrad(*(cLayer->fwInput));
    }
    else
      G = (G * cLayer->weight->t()) % cLayer->preLayer->actGrad(*(cLayer->fwInput));
    backprop(cLayer->preLayer, G, learnRate, trainMode);
  }
  return;
}

void NNet::backprop(NNLayer* cLayer, arma::mat &G, double learnRate, int trainMode) // trainMode: 0 only L2 loss
{
  //Rcout<<".";
  if(trainMode == 0)
    *(cLayer->weight) -= learnRate*(cLayer->fwInput->t() * G);
  else if(trainMode == 1 || trainMode == 2)
    *(cLayer->weight) -= learnRate*((cLayer->fwInput->t() * G) + (this->lambda) * *(cLayer->weight));
  *(cLayer->bias) -= learnRate*arma::sum(G,0);
  if(cLayer->preLayer != NULL)
  {
    if( trainMode == 2 && cLayer->preLayer == this->midLayer)
    {
      G = (G * cLayer->weight->t());
      arma::mat Gp(size(G));
      Gp.zeros();
      for(int i = 0; i < Gp.n_rows; i++)
        for(int j = 0; j < Gp.n_rows; j++)
          if(i != j)
          {
            arma::rowvec deltay;
            deltay = cLayer->fwInput->row(i) - cLayer->fwInput->row(j);
            double sqnorm_deltay;
            sqnorm_deltay = arma::norm(deltay, 2);
            sqnorm_deltay = sqnorm_deltay*sqnorm_deltay;
            if(this->trainlabels[i] != this->trainlabels[j])
              Gp.row(i) -= (deltay)*(2.0/sqnorm_deltay/(1+sqnorm_deltay));
            else
              Gp.row(i) += (deltay)*(2.0/(1+sqnorm_deltay));
          }
          Gp = Gp*(1.0*this->alpha/Gp.n_rows/Gp.n_rows);
          G = G + Gp;
          G = G % cLayer->preLayer->actGrad(*(cLayer->fwInput));
    }
    else
      G = (G * cLayer->weight->t()) % cLayer->preLayer->actGrad(*(cLayer->fwInput));
    backprop(cLayer->preLayer, G, learnRate, trainMode);
  }
  return;
}

void NNet::backpropM(NNLayer* cLayer, arma::mat &G, double learnRate, double momentum, int trainMode) // trainMode: 0 only L2 loss
{
  //Rcout<<".";
  *(cLayer->weightVel) *= momentum;
  *(cLayer->biasVel)   *= momentum;
  if(trainMode == 0)
    *(cLayer->weightVel) -= learnRate*(cLayer->fwInput->t() * G);
  else if(trainMode == 1 || trainMode == 1)
    *(cLayer->weightVel) -= learnRate*((cLayer->fwInput->t() * G) + (this->lambda) * *(cLayer->weight));
  *(cLayer->biasVel) -= learnRate*arma::sum(G,0);

  *(cLayer->weight) += *(cLayer->weightVel);
  *(cLayer->bias)   += *(cLayer->biasVel);

  if(cLayer->preLayer != NULL)
  {
    if( trainMode == 2 && cLayer->preLayer == this->midLayer)
    {
      G = (G * cLayer->weight->t());
      arma::mat Gp(size(G));
      Gp.zeros();
      for(int i = 0; i < Gp.n_rows; i++)
        for(int j = 0; j < Gp.n_rows; j++)
          if(i != j)
          {
            arma::rowvec deltay;
            deltay = cLayer->fwInput->row(i) - cLayer->fwInput->row(j);
            double sqnorm_deltay;
            sqnorm_deltay = arma::norm(deltay, 2);
            sqnorm_deltay = sqnorm_deltay*sqnorm_deltay;
            if(this->trainlabels[i] != this->trainlabels[j])
              Gp.row(i) -= (deltay)*(2.0/sqnorm_deltay/(1+sqnorm_deltay));
            else
              Gp.row(i) += (deltay)*(2.0/(1+sqnorm_deltay));
          }
          Gp = Gp*(1.0*this->alpha/Gp.n_rows/Gp.n_rows);
          G = G + Gp;
          G = G % cLayer->preLayer->actGrad(*(cLayer->fwInput));
    }
    else
      G = (G * cLayer->weight->t()) % cLayer->preLayer->actGrad(*(cLayer->fwInput));
    backprop(cLayer->preLayer, G, learnRate, trainMode);
  }
  return;
}

void NNet::backpropLauncher(arma::mat &data, double learnRate, int trainMode = 0, double momentum = 0, bool ispretrain = false, NNLayer* trainLayer = NULL)
{
  this->feedforward(data, true);

  arma::mat G;
  G = (this->lastLayer->fwOutput-data)%this->lastLayer->actGrad(this->lastLayer->fwOutput);
  G /= (2.0*data.n_rows);
  if(ispretrain)
  {
    this->prebackprop(this->lastLayer, G, learnRate, trainLayer, trainMode);
    return;
  }

  if(momentum == 0)
    this->backprop(this->lastLayer, G, learnRate, trainMode);
  else
    this->backpropM(this->lastLayer, G, learnRate, momentum, trainMode);

}

void NNet::pretrain (double learnRate, int iterations, int trainMode, double lambda = 0, double alpha = 0)
{
  this->lambda = lambda;
  this->alpha  = alpha;
  this->trainlabels = this->labels;
  double loss = arma::datum::inf;
  double learnRateStart, learnRateEnd;
  learnRateStart = learnRate;
  learnRateEnd = learnRateStart*0.01;

  NNLayer* trainLayer;

  trainLayer = this->firstLayer;

  while(trainLayer!=NULL)
  {
    for(int i=0;i<iterations;i++)
    {
      Rcout<<"Pretrain Iteration: "<<i+1;
      learnRate = (1.0-1.0*i/iterations)*learnRateStart + 1.0*i/iterations*learnRateEnd;
      this->backpropLauncher((this->data), learnRate, trainMode, 0, true, trainLayer);
      Rcpp::checkUserInterrupt();
      Rcout<<" Loss: ";
      loss = this->squareLoss((this->data));
//      if(trainMode == 1)        loss += this->lambda * 0.5 * this->squareReg();
      Rcout<<loss<<endl;
    }
    trainLayer = trainLayer->nextLayer;
  }

}

void NNet::bgdtrain(double learnRate, int iterations, int trainMode, double lambda = 0, double alpha = 0)
{
  this->lambda = lambda;
  this->alpha  = alpha;
  this->trainlabels = this->labels;
  double loss = arma::datum::inf;
  double learnRateStart, learnRateEnd;
  learnRateStart = learnRate;
  learnRateEnd = learnRateStart*0.01;

  for(int i=0;i<iterations;i++)
  {
    Rcout<<"Iteration: "<<i+1;
    learnRate = (1.0-1.0*i/iterations)*learnRateStart + 1.0*i/iterations*learnRateEnd;
    this->backpropLauncher((this->data), learnRate, trainMode);
    Rcout<<" Loss: ";
    loss = this->squareLoss((this->data));
//    if(trainMode == 1)      loss += this->lambda * 0.5 * this->squareReg();
    Rcout<<loss<<endl;
    Rcpp::checkUserInterrupt();
  }
}

void NNet::sgdtrain(double learnRate, int iterations, int batchSize, int trainMode, double lambda = 0, double alpha = 0)
{
  this->lambda = lambda;
  this->alpha  = alpha;
  double loss = arma::datum::inf;
  double learnRateStart, learnRateEnd;
  learnRateStart = learnRate;
  learnRateEnd = learnRateStart*0.01;

  int fullSize, idStart, idEnd;
  fullSize = this->data.n_rows;
  arma::uvec shuffledID(fullSize);
  arma::uvec subID;
  for(int i=0;i<fullSize;i++) shuffledID[i]=i;
  arma::mat trainBatch;
  int alliter, citer=0;
  alliter = iterations*ceil(1.0*fullSize/batchSize);
  for(int i=0;i<iterations;i++)
  {
    shuffledID = arma::shuffle(shuffledID);
    idStart = 0;
    idEnd = batchSize-1;
    Rcout<<"Iteration: "<<i+1;
    while(idStart < fullSize)
    {
      idEnd = (fullSize-1) < idEnd ? (fullSize-1) : idEnd;
      subID = shuffledID.subvec( idStart, idEnd );
      trainBatch = this->data.rows(subID);
      this->trainlabels = this->labels(subID);
      //citer++;
      //learnRate = (1.0-1.0*citer/alliter)*learnRateStart + 1.0*citer/alliter*learnRateEnd;
      learnRate = (1.0-1.0*i/iterations)*learnRateStart + 1.0*i/iterations*learnRateEnd;
      this->backpropLauncher(trainBatch, learnRate, trainMode);
      idEnd += batchSize;
      idStart += batchSize;
    }
    Rcout<<" Loss: ";
    loss = this->squareLoss((this->data));
//    if(trainMode == 1)      loss += this->lambda * 0.5 * this->squareReg();
    Rcout<<loss<<endl;
    Rcpp::checkUserInterrupt();
  }
}

void NNet::sgdMtrain(double learnRate, int iterations, int batchSize, double momentum, int trainMode, double lambda = 0, double alpha = 0)
{
  this->lambda = lambda;
  this->alpha  = alpha;
  double loss = arma::datum::inf;

  int fullSize, idStart, idEnd;
  fullSize = this->data.n_rows;
  arma::uvec shuffledID(fullSize);
  arma::uvec subID;
  for(int i=0;i<fullSize;i++) shuffledID[i]=i;
  arma::mat trainBatch;

  for(int i=0;i<iterations;i++)
  {
    shuffledID = arma::shuffle(shuffledID);
    idStart = 0;
    idEnd = batchSize-1;
    Rcout<<"Iteration: "<<i+1;
    while(idStart < fullSize)
    {
      idEnd = (fullSize-1) < idEnd ? (fullSize-1) : idEnd;
      subID = shuffledID.subvec( idStart, idEnd );
      this->trainlabels = this->labels(subID);
      trainBatch = this->data.rows(subID);
      this->backpropLauncher(trainBatch, learnRate, momentum, trainMode);
      idEnd += batchSize;
      idStart += batchSize;
    }
    Rcout<<" Loss: ";
    loss = this->squareLoss((this->data));
//    if(trainMode == 1)      loss += this->lambda * 0.5 * this->squareReg();
    Rcout<<loss<<endl;
    Rcpp::checkUserInterrupt();
  }
}

arma::mat NNet::netOutput(arma::mat &data)
{
  arma::mat lastOut;
  lastOut = this->lastLayer->feedforward(data, false);

  if(actName == "sigmoid" || actName == "relu")
  {
    this->data *= (datamax-datamin);
    this->data += datamin;
  }
  else if(actName == "tanh")
  {
    this->data /= 2;
    this->data += 0.5;
    this->data *= (datamax-datamin);
    this->data += datamin;
  }
  return lastOut;
}







