#### 计算程序的运行时间！！！
newlabel = digitsTrainLabel;
newlabel[newlabel>0] = 1;
newID = sample(60000);
timestart<-Sys.time();

newdata<-esmote::Smote(digitsTrain[newID,],newlabel[newID], algorithm="rp_forest");
# newdata <- DMwR::SMOTE(label ~.,tdata, perc.over = 200,perc.under=100);
# newdata <- smotefamily::SMOTE(digitsTrain,newlabel);
# newdata <- ubSMOTE(X=digitsTrain,Y=as.factor(1-newlabel),perc.under = 100);

timeend<-Sys.time()
runningtime<-timeend-timestart
print(runningtime)


#### 预测模型
require(mxnet)
newlabelTrain = digitsTrainLabel;
newlabelTrain[newlabelTrain>0] = 1;
newlabelTest = digitsTestLabel;
newlabelTest[newlabelTest>0] = 1;

modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  model = mx.mlp(digitsTrain, newlabelTrain, hidden_node = 10, out_node = 2, out_activation = "softmax",
                 learning.rate = 0.1, num.round=40,  eval.metric=mx.metric.rmse)
  preds = predict(model, digitsTest)
  pred.label = max.col(t(preds))-1
  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!newlabelTest))
  predMat[1,2] = sum((!pred.label)&(newlabelTest))
  predMat[2,1] = sum((pred.label)&(!newlabelTest))
  predMat[2,2] = sum(pred.label&newlabelTest)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
}
modelAccuracy = modelAccuracy/10
modelRecall = modelRecall/10
####


#### 预测模型with Smote
require(mxnet)


modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  newlabel = digitsTrainLabel;
  newlabel[newlabel>0] = 1;

  timestart<-Sys.time();
  newdata<-Smote(digitsTrain,newlabel, N=900, k=10, algorithm="rp_forest");
  timeend<-Sys.time()
  runningtime<-timeend-timestart
  print(runningtime)

  newlabelTrain = digitsTrainLabel;
  newlabelTrain[newlabelTrain>0] = 1;
  newlabelTrain = c(newlabelTrain,rep(0,nrow(newdata)));
  newdigitsTrain = rbind(digitsTrain,newdata);
  newID = sample(nrow(newdigitsTrain));
  newdigitsTrain = newdigitsTrain[newID,];
  newlabelTrain = newlabelTrain[newID];
  newlabelTest = digitsTestLabel;
  newlabelTest[newlabelTest>0] = 1;

  model = mx.mlp(newdigitsTrain, newlabelTrain, hidden_node = 10, out_node = 2, out_activation = "softmax",
                 learning.rate = 0.1, num.round=40,  eval.metric=mx.metric.rmse)
  preds = predict(model, digitsTest)
  pred.label = max.col(t(preds))-1
  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!newlabelTest))
  predMat[1,2] = sum((!pred.label)&(newlabelTest))
  predMat[2,1] = sum((pred.label)&(!newlabelTest))
  predMat[2,2] = sum(pred.label&newlabelTest)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
  cat(modelAccuracy)
  cat(modelRecall)
}
modelAccuracy = modelAccuracy/10
modelRecall = modelRecall/10
####

###NB

library(e1071)
modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  newlabel = digitsTrainLabel;
  newlabel[newlabel>0] = 1;

  # timestart<-Sys.time();
  # newdata<-Smote(digitsTrain,newlabel, N=900, k=10, algorithm="rp_forest");
  # timeend<-Sys.time()
  # runningtime<-timeend-timestart
  # print(runningtime)
  #
  # newlabelTrain = digitsTrainLabel;
  # newlabelTrain[newlabelTrain>0] = 1;
  # newlabelTrain = c(newlabelTrain,rep(0,nrow(newdata)));
  # newdigitsTrain = rbind(digitsTrain,newdata);
  # newID = sample(nrow(newdigitsTrain));
  # newdigitsTrain = newdigitsTrain[newID,];
  # newlabelTrain = newlabelTrain[newID];
  # newlabelTest = digitsTestLabel;
  # newlabelTest[newlabelTest>0] = 1;
  #
  # model <- naiveBayes(newdigitsTrain,newlabelTrain)

  newlabelTest = digitsTestLabel;
  newlabelTest[newlabelTest>0] = 1;

  model <- naiveBayes(digitsTrain,newlabel)

  preds = predict(model, digitsTest, type = "raw")
  pred.label = max.col(preds)-1
  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!newlabelTest))
  predMat[1,2] = sum((!pred.label)&(newlabelTest))
  predMat[2,1] = sum((pred.label)&(!newlabelTest))
  predMat[2,2] = sum(pred.label&newlabelTest)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
  cat(modelAccuracy)
  cat(modelRecall)
}
modelAccuracy = modelAccuracy/10
modelRecall = modelRecall/10




#### test autoencoder


NNLayer_module <- Rcpp::Module( "NNLayer_module", inline::getDynLib("esmote"))
NNet<-NNLayer_module$NNet
net1 <- new(NNet,digitsTrain[sample(60000),],c(100),"sigmoid")
net1$pretrain(0.25,10,0);
#net1$bgdtrain(0.25,100,0);
#
net1$sgdtrain(0.25,100,100,0);
#net1$sgdMtrain(0.25,100,100,0.9,0);
visulizeDigits(net1$netOutput(t(as.matrix(digitsTrain[1200,]))))
visulizeDigits((t(as.matrix(digitsTrain[1200,]))))


#vis the mid feature mode 0
colors = rainbow(length(unique(digitsTestLabel)))
names(colors) = unique(digitsTestLabel)

library(Rtsne)
rtsne_out <- Rtsne(midFeatures)
#save(rtsne_out, file="C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\midFeatures_mode0_tsne.RData")
#jpeg("C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\midFeatures_mode0_tsne.jpg", width=1200, height=900)
plot(rtsne_out$Y, t='n')
text(rtsne_out$Y, labels=digitsTrainLabel, col=colors[digitsTrainLabel+1])
dev.off()

#vis the mid feature mode 0 with smote
newlabel = digitsTrainLabel;
newlabel[newlabel>0] = 1;
newdata<-Smote(midFeatures,newlabel, N=900, k=10, algorithm="rp_forest");
newlabelTrain = digitsTrainLabel;
newlabelTrain[newlabelTrain>0] = 1;
newlabelTrain = c(newlabelTrain,rep(0,nrow(newdata)));
newdigitsTrain = rbind(midFeatures,newdata);
newID = sample(nrow(newdigitsTrain));
newdigitsTrain = newdigitsTrain[newID,];
newlabelTrain = newlabelTrain[newID];

rtsne_out <- Rtsne(newdigitsTrain)
jpeg("C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\midFeatures_mode0_smote_tsne.jpg", width=1200, height=900)
plot(rtsne_out$Y, t='n')
text(rtsne_out$Y, labels=newlabelTrain, col=colors[newlabelTrain+1])
dev.off()



### the prove of the usefulness of mode3
data1=matrix(runif(200*800, -1, +1), ncol = 200)*4;
data2=matrix(runif(200*200, -1, +1), ncol = 200)*4+(64/200)^0.5+0.1;  # discrimination
data2=matrix(runif(200*200, -1, +1), ncol = 200)*4+(64/200)^0.5+1.4;  # imbalanced problem
data3 = rbind(data1, data2)

rtsne_out <- Rtsne(data3)
plot(rtsne_out$Y, col=c(rep(1,800),rep(2,200)))

NNLayer_module <- Rcpp::Module( "NNLayer_module", inline::getDynLib("esmote"))
NNet<-NNLayer_module$NNet
newID = sample(1000);
newlabels = c(rep(1,800),rep(0,200));
newlabels = newlabels[newID]
net1 <- new(NNet,data3[newID,],c(2),"sigmoid")
net1$setLabels(newlabels)
net1$pretrain(0.25,30,2,0.1,0.08);
#net1$bgdtrain(5,10000,2,0.1,0.00001);
#
net1$sgdtrain(0.25,100,100,2,0.1,0.08)


midF <- net1$midFeature(data3)
plot(midF, col=c(rep(1,800),rep(2,200)))




#### test autoencoder


NNLayer_module <- Rcpp::Module( "NNLayer_module", inline::getDynLib("esmote"))
NNet<-NNLayer_module$NNet
newID = sample(60000);
net1 <- new(NNet,digitsTrain[newID,],c(100),"sigmoid")
net1$setLabels(digitsTrainLabel[newID])
net1$pretrain(0.25,15,0,0,0.8);
#net1$sgdtrain(0.25,10,100,0,0,0.8)
net1$sgdtrain(0.25,300,1000,2,0,8)

visulizeDigits(net1$netOutput(t(as.matrix(digitsTrain[300,]))))
visulizeDigits((t(as.matrix(digitsTrain[300,]))))

colors = rainbow(length(unique(digitsTestLabel)))
names(colors) = unique(digitsTestLabel)

midFeatures = net1$midFeature(digitsTrain)
midFeaturesTest = net1$midFeature(digitsTest)

library(Rtsne)
rtsne_out <- Rtsne(midFeatures);
save(rtsne_out, file="C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\midFeaturesTrain_mode3_tsne.RData")
jpeg("C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\midFeaturesTrain_mode3_tsne.jpg", width=1200, height=900)
plot(rtsne_out$Y, t='n')
text(rtsne_out$Y, labels=digitsTrainLabel, col=colors[digitsTrainLabel+1])
dev.off()

##
modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  newlabel = digitsTrainLabel;
  newlabel[newlabel>0] = 1;

  timestart<-Sys.time();
  newdata<-Smote(midFeatures,newlabel, N=900, k=10, algorithm="brute");
  timeend<-Sys.time()
  runningtime<-timeend-timestart
  print(runningtime)

  newlabelTrain = digitsTrainLabel;
  newlabelTrain[newlabelTrain>0] = 1;
  newlabelTrain = c(newlabelTrain,rep(0,nrow(newdata)));
  newdigitsTrain = rbind(midFeatures,newdata);
  newID = sample(nrow(newdigitsTrain));
  newdigitsTrain = newdigitsTrain[newID,];
  newlabelTrain = newlabelTrain[newID];
  newlabelTest = digitsTestLabel;
  newlabelTest[newlabelTest>0] = 1;

  model = mx.mlp(newdigitsTrain, newlabelTrain, hidden_node = 10, out_node = 2, out_activation = "softmax",
                 learning.rate = 0.1, num.round=40,  eval.metric=mx.metric.rmse)
  preds = predict(model, midFeaturesTest)
  pred.label = max.col(t(preds))-1
  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!newlabelTest))
  predMat[1,2] = sum((!pred.label)&(newlabelTest))
  predMat[2,1] = sum((pred.label)&(!newlabelTest))
  predMat[2,2] = sum(pred.label&newlabelTest)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
  cat(modelAccuracy)
  cat(modelRecall)
}
modelAccuracy = modelAccuracy/10
modelRecall = modelRecall/10

library(e1071)
modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  newlabel = digitsTrainLabel;
  newlabel[newlabel>0] = 1;

  timestart<-Sys.time();
  newdata<-Smote(midFeatures,newlabel, N=900, k=10, algorithm="brute");
  timeend<-Sys.time()
  runningtime<-timeend-timestart
  print(runningtime)

  newlabelTrain = digitsTrainLabel;
  newlabelTrain[newlabelTrain>0] = 1;
  newlabelTrain = c(newlabelTrain,rep(0,nrow(newdata)));
  newdigitsTrain = rbind(midFeatures,newdata);
  newID = sample(nrow(newdigitsTrain));
  newdigitsTrain = newdigitsTrain[newID,];
  newlabelTrain = newlabelTrain[newID];
  newlabelTest = digitsTestLabel;
  newlabelTest[newlabelTest>0] = 1;

  model <- naiveBayes(newdigitsTrain,newlabelTrain)

  preds = predict(model, midFeaturesTest, type = "raw")
  pred.label = max.col(preds)-1
  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!newlabelTest))
  predMat[1,2] = sum((!pred.label)&(newlabelTest))
  predMat[2,1] = sum((pred.label)&(!newlabelTest))
  predMat[2,2] = sum(pred.label&newlabelTest)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
  cat(modelAccuracy)
  cat(modelRecall)
}
modelAccuracy = modelAccuracy/10
modelRecall = modelRecall/10






#### tweet data
require(mxnet)


modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  testID = sample(14485,1000);

  trainTweetData = tweetData[-testID,];
  trainTweetLabel = tweetLabel[-testID];
  trainTweetLabel[trainTweetLabel==3] = 0;
  trainTweetLabel[trainTweetLabel>0] = 1;

  testTweetData = tweetData[testID,];
  testTweetLabel = tweetLabel[testID];
  testTweetLabel[testTweetLabel==3] = 0;
  testTweetLabel[testTweetLabel>0] = 1;

  newdata<-Smote(trainTweetData,trainTweetLabel, N=500, k=10, algorithm="rp_forest");
  trainTweetData = rbind(trainTweetData,newdata);
  trainTweetLabel = c(trainTweetLabel, rep(0, nrow(newdata)));
  newID = sample(nrow(trainTweetData));
  trainTweetData = trainTweetData[newID,]
  trainTweetLabel = trainTweetLabel[newID]

  model = mx.mlp(trainTweetData, trainTweetLabel, hidden_node = 10, out_node = 2, out_activation = "softmax",
                 learning.rate = 0.1, num.round=40,  eval.metric=mx.metric.rmse)
  preds = predict(model, testTweetData)
  pred.label = max.col(t(preds))-1
  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!testTweetLabel))
  predMat[1,2] = sum((!pred.label)&(testTweetLabel))
  predMat[2,1] = sum((pred.label)&(!testTweetLabel))
  predMat[2,2] = sum(pred.label&testTweetLabel)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
  cat(modelAccuracy,modelRecall)
}

##NB tweet

modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  testID = sample(14485,1000);

  trainTweetData = tweetData[-testID,];
  trainTweetLabel = tweetLabel[-testID];
  trainTweetLabel[trainTweetLabel==3] = 0;
  trainTweetLabel[trainTweetLabel>0] = 1;

  testTweetData = tweetData[testID,];
  testTweetLabel = tweetLabel[testID];
  testTweetLabel[testTweetLabel==3] = 0;
  testTweetLabel[testTweetLabel>0] = 1;

  #newdata<-Smote(trainTweetData,trainTweetLabel, N=500, k=10, algorithm="brute");
  #trainTweetData = rbind(trainTweetData,newdata);
  #trainTweetLabel = c(trainTweetLabel, rep(0, nrow(newdata)));
  #newID = sample(nrow(trainTweetData));
  #trainTweetData = trainTweetData[newID,]
  #trainTweetLabel = trainTweetLabel[newID]

  model = naiveBayes(trainTweetData,trainTweetLabel)
  preds = predict(model, testTweetData, type = "raw")
  pred.label = max.col(preds)-1
  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!testTweetLabel))
  predMat[1,2] = sum((!pred.label)&(testTweetLabel))
  predMat[2,1] = sum((pred.label)&(!testTweetLabel))
  predMat[2,2] = sum(pred.label&testTweetLabel)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
  cat(modelAccuracy,modelRecall)
}

#### test autoencoder airline

library(Rtsne)
newtweetLabel = tweetLabel;
newtweetLabel[newtweetLabel==3] = 0;
newtweetLabel[newtweetLabel>0] = 1;

isd = duplicated(tweetData);

rtsne_out <- Rtsne(tweetData[!isd,]);
#save(rtsne_out, file="C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\midFeaturesTrain_mode3_tsne.RData")
jpeg("C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\tweet_tsne.jpg", width=1200, height=900)
plot(rtsne_out$Y, t='n')
text(rtsne_out$Y, labels=newtweetLabel[!isd], col=colors[newtweetLabel[!isd]*2+1])
dev.off()


testID = sample(14485,1000);

trainTweetData = tweetData[-testID,];
trainTweetLabel = tweetLabel[-testID];
trainTweetLabel[trainTweetLabel==3] = 0;
trainTweetLabel[trainTweetLabel>0] = 1;

testTweetData = tweetData[testID,];
testTweetLabel = tweetLabel[testID];
testTweetLabel[testTweetLabel==3] = 0;
testTweetLabel[testTweetLabel>0] = 1;

NNLayer_module <- Rcpp::Module( "NNLayer_module", inline::getDynLib("esmote"))
NNet<-NNLayer_module$NNet
newID = sample(14485-1000);
net1 <- new(NNet,trainTweetData[newID,],c(200,100),"sigmoid")
net1$setLabels(trainTweetLabel[newID])
net1$pretrain(0.25,15,0,0,0.8);
NNwei = net1$getweight();

#save(NNwei, file="C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\tweetPreTrainNNWei.RData")

## mode0
#net1$sgdtrain(0.25,10,100,0,0,0.8)
net2 <- new(NNet,trainTweetData[newID,],c(200,100),"sigmoid")
net2$setLabels(trainTweetLabel[newID])
net2$recoverWei(NNwei);
net2$sgdtrain(0.25,300,1000,0,0,0)
#NNwei = net2$getweight();
#save(NNwei, file="C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\tweetmode0NNWei.RData")


colors = rainbow(length(unique(digitsTestLabel)))
names(colors) = unique(digitsTestLabel)

midFeatures = net2$midFeature(trainTweetData)
midFeaturesTest = net2$midFeature(testTweetData)

isd = duplicated(midFeatures);
library(Rtsne)
rtsne_out <- Rtsne(midFeatures[!isd,]);
#save(rtsne_out, file="C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\midFeaturesTrain_mode3_tsne.RData")
jpeg("C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\tweet_mode0_midF_tsne.jpg", width=1200, height=900)
plot(rtsne_out$Y, t='n')
text(rtsne_out$Y, labels=trainTweetLabel[!isd], col=colors[trainTweetLabel[!isd]*2+1])
dev.off()


## mode3
isd = duplicated(trainTweetData);
ndtrainTweetData = trainTweetData[!isd,]
ndtrainTweetLabel = trainTweetLabel[!isd]

newID = sample(nrow(ndtrainTweetData));

net2 <- new(NNet,ndtrainTweetData[newID,],c(200,100),"sigmoid")
net2$setLabels(ndtrainTweetLabel[newID])
net2$recoverWei(NNwei);
net2$sgdtrain(0.25,300,1000,2,0,80)
NNwei3 = net2$getweight();
save(NNwei3, file="C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\tweetmode3NNWei.RData")

midFeatures = net2$midFeature(ndtrainTweetData)
midFeaturesTest = net2$midFeature(testTweetData)

rtsne_out <- Rtsne(midFeatures);
jpeg("C:\\Users\\surface\\Desktop\\HighD-Imbalance\\Work Log\\tweet_mode3_midF_tsne.jpg", width=1200, height=900)
plot(rtsne_out$Y, t='n')
text(rtsne_out$Y, labels=ndtrainTweetLabel, col=colors[ndtrainTweetLabel*2+1])
dev.off()



modelAccuracy = 0
modelRecall = 0
for(i in 1:10)
{
  # midFeatures;
  # ndtrainTweetLabel;
  #
  # midFeaturesTest;
  # testTweetLabel;

  newdata<-Smote(midFeatures,ndtrainTweetLabel, N=500, k=10, algorithm="rp_forest");
  alltrainTweetData = rbind(midFeatures,newdata);
  alltrainTweetLabel = c(ndtrainTweetLabel, rep(0, nrow(newdata)));
  newID = sample(nrow(alltrainTweetData));
  alltrainTweetData = alltrainTweetData[newID,]
  alltrainTweetLabel = alltrainTweetLabel[newID]

  #model = naiveBayes(alltrainTweetData,alltrainTweetLabel)
  #preds = predict(model, midFeaturesTest, type = "raw")
  #pred.label = max.col(preds)-1


  model = mx.mlp(alltrainTweetData, alltrainTweetLabel, hidden_node = 10, out_node = 2, out_activation = "softmax",
                 learning.rate = 0.1, num.round=40,  eval.metric=mx.metric.rmse)
  preds = predict(model, midFeaturesTest, type = "raw")
  pred.label = max.col(t(preds))-1

  predMat = matrix(1:4,ncol=2)
  predMat[1,1] = sum((!pred.label)&(!testTweetLabel))
  predMat[1,2] = sum((!pred.label)&(testTweetLabel))
  predMat[2,1] = sum((pred.label)&(!testTweetLabel))
  predMat[2,2] = sum(pred.label&testTweetLabel)
  modelAccuracy = (predMat[1,1]+predMat[2,2])/sum(predMat) + modelAccuracy
  modelRecall = (predMat[1,1])/sum(predMat[,1]) + modelRecall
  cat(modelAccuracy,modelRecall)
}
