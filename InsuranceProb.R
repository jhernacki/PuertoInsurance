require(leaps)
require(car)
require(MASS)
require(randomForest)
require(doSNOW)
library(caret) # for model-building
library(DMwR) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculations
require(mboost)
require(MLmetrics)
require(nnet)
require(ggplot2)
require(mxnet)
ctrl <- NULL
indexes <- createDataPartition(trainer$target,
                               times = 1,
                               p = 0.75,
                               list = FALSE)
trainer<- train[-indexes,]
tester <- train[indexes,]
str(trainer)
save(trainer, file = 'trainer.rda')
save(tester, file = 'tester.rda')

setwd('/home/jthernacki/Documents/Puerto_kaggle_insurance/')
load('trainer.rda')
load('tester.rda')

#just glm and lm to try to check for multicollinearity
fit <- lm(ps_car_12~ . -id -target2,data = trainer)
alias(fit)
vif(fit)

plot(fit)
fit <- glm(target~ . -id,data = pp, family = binomial)
summary(fit)
fit <- glm(target~ ps_ind_14,data = pp, family = binomial)
glm.probs <- predict(fit, type = "response")
glm.pred <- rep("negative", dim(pp)[1])
glm.pred[glm.probs > 0.5] = "possitive"
table(pp$target)
names(head(glm.probs))

table(glm.probs, pp$target)

#setting up for cross validation
ctrl <- NULL

target <- ifelse(wholeReduced$target2 == 'Not', 0, 1)
target <- as.numeric(target)
length(target)
folds <- 10
cvIndex <- createFolds(target, folds, returnTrain = T)
#For stratified Cross Validation
tc <- trainControl(index = cvIndex,
                   method = 'cv', 
                   number = folds,
                   summaryFunction = twoClassSummary,
                   classProbs = T)
#Regular Cross Validation
ctrl <- trainControl(method = "cv",
                     number = 10,#10
                     summaryFunction = twoClassSummary,
                     #returnResamp = "all",
                     classProbs = T)
ctrl$sampling = 'up'

#instantiating weights
y2 <- ifelse(trainer$target2 == 'Not', 0, 1)
model_weights <- ifelse(var_reduced_test$target2 == 'Not', 
                        (1/table(var_reduced_test$target2)[2])*.4, 
                        (1/table(var_reduced_test$target2)[1])*.6)
sum(model_weights)
#Basic logistic regression
ctrl$sampling = 'smote'
ctrl$sampling = NULL
set.seed(1103)
glmfit <- train(target2 ~.,
                   data = wholeReduced,
                   method = "glm", 
                   family = binomial,
                   preProcess=c('scale', 'center'),
                   weights = model_weights,
                   metric = "ROC",
                   trControl = tc)

glmfit
preds <- predict(glmfit, var_reduced_train, type = 'prob') 
target <- ifelse(var_reduced_train$target2 == 'Default', 1, 0)
tester$target <- as.numeric(tester$target)
NormalizedGini(preds[[1]], target)
summary(glmfit)
#origC     weight(.6)    down       up      smote 
#0.2440    0.2457       0.2382    0.2450    0.2205

#boosted logistic
install.packages('mboost')
require(mboost)
glmfit <- train(target2 ~.,
                data = var_reduced_train,
                method = "glmboost",
                preProcess=c('scale', 'center'),
                weights = model_weights,
                metric = "ROC",
                trControl = ctrl)
plot(glmfit)
preds <- predict(glmfit, var_reduced_test,type = 'prob') 
target <- ifelse(tester$target2 == 'Default', 1, 0)
target <- as.numeric(target)
NormalizedGini(preds[[1]], target)

#weighted  Up        down        smote
#0.2419
#reduced Vars
#0.2419    0.242078  0.2367365   0.2288453

#Picking the best weight to use
target <- ifelse(tester$target2 == 'Not', 0, 1)
target <- as.numeric(target)
cv <- rep(0, 10)
j=1
for(i in seq(.3, .7, .1)){
  model_weights <- ifelse(trainer$target2 == 'Default', 
                          (1/table(trainer$target)[1])*i, 
                          (1/table(trainer$target)[2])*(1-i))
  set.seed(1103)
  glmfit <- train(target2 ~.,
                  data = trainer,
                  method = "glmboost",
                  preProcess=c('scale', 'center'),
                  weights = model_weights,
                  metric = "ROC",
                  trControl = ctrl)
  pred <- predict(glmfit, tester, type = 'prob') 
  cv[j] <- NormalizedGini(pred[[1]], target)
  j=j+1
}
cv
#Weights .4 and .6
model_weights <- ifelse(trainer$target2 == 'Default', 
                        (1/table(trainer$target)[1])*.4, 
                        (1/table(trainer$target)[2])*(.6))
sum(model_weights)
glmfit <- train(target2 ~.,
                data = trainer,
                method = "glmboost",
                preProcess=c('scale', 'center'),
                weights = model_weights,
               # tunelength = 5,
                metric = "ROC",
                trControl = ctrl)
pred <- predict(glmfit, tester, type = 'prob') 
NormalizedGini(pred[[1]], target)

#gradient boosting for optimizing arbitrary loss functions where component-wise linear models are base-learners
#move slightly opposite of gradient every time (vector of direction with fastest increase)
set.seed(1103)
cl <- makeCluster(8, type = 'SOCK')
registerDoSNOW(cl)
model_weights <- ifelse(trainer$target2 == 'Default', 
                        (1/table(trainer$target2)[1])*.4, 
                        (1/table(trainer$target2)[2])*(.6))
sum(model_weights)
glmfit <- train(target2 ~.,
                data = wholeReduced,
                method = "glmboost",
                family = Binomial(),
                tuneGrid = expand.grid(mstop = (c(150, 300)), prune = 'no') ,
                preProcess=c('scale', 'center'),
                weights = model_weights,
                metric = "ROC",
                trControl = tc)

glmfit
#.242458
#mstop = 300, prune = No, .2449
#
gamfit$results
plot(gamfit)
?mboost::mstop
preds <- predict(gamfit, var_reduced_test, type = 'prob') 
target <- ifelse(tester$target2 == 'Default', 1, 0)
target <- as.numeric(target)
NormalizedGini(preds[[1]], target)

#cv the weights, then nu, and mstop
boostedglm <- glmboost(target2~., data = trainer, weights = model_weights,family = Binomial(), 
         control = boost_control(mstop = 500, nu = .001, risk = 'inbag', center = T))
save(boostedglm, file = 'boostedglm.rda')
preds <- predict(boostedglm, var_reduced_test, type = 'response') 
NormalizedGini(preds[,1], target)
#.1885

#LogitBoost represents an application of established logistic regression 
#techniques to the AdaBoost method. Rather than minimizing error with respect 
#to y, weak learners are chosen to minimize the (weighted least-squares) error
set.seed(1103)
plot(boostedLogit)
boostedLogit <- train(target2 ~.,
                data = var_reduced_train,
                method = "LogitBoost",
                family = Binomial(),
                #tuneGrid = expand.grid(mstop = c(500, 700, 900), prune = 'no') ,
                preProcess=c('scale', 'center'),
                weights = model_weights,
                metric = "ROC",
                trControl = ctrl)

preds <- predict(boostedLogit, tester, type = 'prob') 
NormalizedGini(preds[[1]], target)


model_weights <- ifelse(var_reduced_train$target2 == 'Not', 
                        (1/table(var_reduced_train$target2)[2])*.6, 
                        (1/table(var_reduced_train$target2)[1])*.4)

sum(model_weights)
load('rf_all.rda')
load('pp.rda')
table(fit$predicted)

#I did not include these tree methods in the powerpoint as it was not my role to do them, 
#It was more for learning experience
require(randomForest)
fit <- randomForest(target~.-id , data = train, ntree = 50, mtry = floor(sqrt(ncol(train))))
require(ranger)
grep('11_cat', colnames(trainer))
Forester <- ranger(target2 ~. ,
                data = var_reduced_train,
                probability = TRUE,
                seed = 1103,
                mtry = 4, 
                splitrule = 'extratrees', 
                importance = 'impurity',
                case.weights = model_weights,
                
                min.node.size = 500)

#.1885
Forester
r
plot(Forester)
print.ranger.forest
preds <- predict(Forester, var_reduced_test, type = 'response') 
head(preds)
NormalizedGini(preds[[1]][,1], target)

#gini, mtry = 7, no weights, .1987
#gini, mtry = 7, .6, .4 weights, .216  
#gini, mtry = 7, .5, .5 weights, .209
#gini, mtry = 7, .7, .3 weights, .195
#gini, mtry = 5, .7, .3 weights, .209
#gini, mtry = 9, .7, .3 weights, .197
#extratrees, mtry = 5, .6, .4 weights, .2209 with reduced vars
#extratrees, mtry = 4, .6, .4 weights, .2224 with reduced vars
#extratrees, mtry = 4, .6, .4 weights, min.node.size = 500, .236 with reduced vars
                                                      #300, .234
                                                      #50,  .2277

#extra trees finds best splitting value by searching k split points, for categoricals, 
#create k subset of categorical, pick the best
#this is a very efficient version of RF
Forester <- train(target ~. -id,
                      data = train,
                      method = "ranger",
                      seed = 1103,
                      importance = 'impurity',
                      tuneGrid = expand.grid(mtry = 12, splitrule = 'extratrees', min.node.size = 500) ,
                      preProcess= 'pca',
                      weights = model_weights,
                      metric = "ROC",
                      trControl = ctrl)

require(gbm)
preds <- predict(Forester, trainer, type = 'prob') 
Forester$preProcess
head(preds)
NormalizedGini(preds[,1], target)
#.2332
#.2232 for pca
#with pca on whole trainer .00138

#Created from 147184 samples and 202 variables

#Pre-processing:
#  - centered (202)
#- ignored (0)
#- principal component signal extraction (202)
#- scaled (202)

#PCA needed 165 components to capture 95 percent of the variance

#orig rf
set.seed(1103) #20 gigs using one core
Forester <- train(target2 ~.,
                  data = wholeReduced,
                  method = "rf",
                  seed = 1103,
                  ntree = 300,
                  #importance = 'impurity',
                  tuneGrid = expand.grid(mtry = 6) ,
                  #preProcess= 'pca',
                  weights = model_weights,
                  metric = "ROC",
                  trControl = tc)



#Whole reduced is for the stratified Cross Validation
y2 <- c(var_reduced_test$target2)
x <-model.matrix(target2 ~ . ,data = var_reduced_train)[,-1]
x2 <-model.matrix(target2 ~ . ,data =  var_reduced_test)[,-1]
wholeReduced <- rbind(var_reduced_train, var_reduced_test)


model_weights <- ifelse(wholeReduced$target2 == 'Default', 
                        (1/table(wholeReduced$target2)[1])*.6, 
                        (1/table(wholeReduced$target2)[2])*.4)

sum(model_weights)

#################
##################
#I have had no training or experience working with Neural Networks
#So the next section on those topics is me playing around and attempting to learn
##################
#################



#Here averaged NNets are tried
cl <- makeCluster(2, type = 'SOCK')
registerDoSNOW(cl)
grep('car_01', colnames(var_reduced_train))
startCluster(cl)
y2 <- as.factor(y2)
table(y2)
y2 <- ifelse(y2 == 1, 'Not', 'Default')
set.seed(1103)
gc(reset = T)
neural <- train(trainTreated[,-87], ifelse(trainTreated[,87] == 1, 'Default', 'Not'),
                  method = "avNNet",
                  #preProc = c("center", "scale"),
                  trace = FALSE,
                  allowParallel = TRUE,
                  repeats = 5,
                  tuneGrid = expand.grid(size = 3, 
                                         decay = .0001, 
                                         bag = F),
                  weights = model_weights,
                  metric = "ROC",
                  trControl = tc)
neural
save(neural, file = 'cvedneural.rda')

stopCluster(cl)

plot(neural)
neural$results
preds <- predict(neural, x2, y2, type = 'prob') 
head(preds)
NormalizedGini(preds[[1]], target)
#repeats = 3, decay = .001, hidden units = 5, .2452
#                                             .226  without scaling
#                                             .2406 with pca
#repeats = 5, .2452
#repeats = 10, .2451


#NNET in R

model_weights <- ifelse(var_reduced_train$target2 == 1, 
                        (1/table(var_reduced_train$target2)[2])*.6, 
                        (1/table(var_reduced_train$target2)[1])*.4)

sum(model_weights)

y <- as.factor(y)
y2 <- ifelse(trainer$target2 == 'Not', 0, 1)
y2 <- as.numeric(y)
require(brnn)
preds <- predict(glmfit, wholeReduced, type = 'prob')
head(preds)
NormalizedGini(preds[[1]], target)
table(preds, target)
new_trainer <- var_reduced_train
new_tester <- var_reduced_test
new_trainer$target2 <- ifelse(new_trainer$target2 == 'Not', .1, .9)
new_trainer$target2 <- factor(new_trainer$target2)
new_tester$target2 <- ifelse(new_tester$target2 == 'Not', .1, .9)
summary(trainTreated$ps_calc_01_clean)
set.seed(1103)
gc(reset = T)
neural <- train(target2,
                #data = trainer,
                method = "nnet",
                maxit = 100,
                #preProc = c("center", "scale"),
                softmax = F,
                #trace = FALSE,
                #allowParallel = TRUE,
                #repeats = 3,
                tuneGrid = expand.grid(size = c(1), decay = c(.0001)),
                weights = model_weights,
                metric = "ROC",
                trControl = tc)
neural
x <- preProcess(var_reduced_train, method = c("center", "scale"))
x <- predict(x, var_reduced_train)
neural2 <- nnet(trainTreated, target, size = 1, decay = .0001, weights = model_weights, maxit = 1000)
preds
plotnn
??neuralnet
require(RCurl)
install.packages('neuralnet')
require(neuralnet)
?neuralnet()

require(statnet)
??neuralnet
plot(neural2)
neural$results
preds <- predict(neural2, x2, y2, type = 'raw')
NormalizedGini(preds[,1], target)
plot(neural)
mat <- table(preds, target)

  #size decay       ROC      Sens      Spec       ROCSD      SensSD      SpecSD
#1    1 0e+00 0.6181041 0.5010043 0.6663597 0.006945448 0.047463197 0.040031098
#2    1 1e-04 0.6251888 0.3234877 0.8167366 0.002908972 0.004426567 0.012831204
#3    1 1e-01 0.6223700 0.0000000 1.0000000 0.004909963 0.000000000 0.000000000
#4    3 0e+00 0.6109492 0.3501420 0.7829191 0.002083550 0.012278696 0.008861350
#5    3 1e-04 0.6203508 0.3274838 0.8110687 0.003539326 0.007564317 0.013983341
#6    3 1e-01 0.6223802 0.0000000 1.0000000 0.004896131 0.000000000 0.000000000
#7    5 0e+00 0.6145004 0.3547935 0.7795209 0.001904483 0.007005650 0.006836797
#8    5 1e-04 0.6199407 0.3452932 0.7968999 0.002222167 0.006906426 0.008318583
#9    5 1e-01 0.6223867 0.0000000 1.0000000 0.004891029 0.000000000 0.000000000
head(preds)
NormalizedGini(preds[[1]], target)
#.2406


devtools::install_github("gaborcsardi/pkgconfig")
devtools::install_github("igraph/rigraph")

#XML
install.packages('DiagrammeR')
install.packages('XML', dependencies = T)
install.packages('/tmp/mozilla_jthernacki0/xml2_1.1.1.tar.gz', repos = NULL, type="source")
install.packages('libxml2-dev')
installed.packages('rgexf')
#rgexf
install.packages('mxnet')
install.packages ("igraph")
require(doMC)
require(mxnet)
regis


#MXNet

t1 <- trainer[trainer$target2 == 'Default',]
t2ind <- sample(1:nrow(trainer), nrow(t1))
t2 <- trainer[t2ind,]
new_t <- rbind(t1, t2)
set.seed(1103)
len = 3
neural <- train(target2 ~ .,
                data = trainer,
                method = 'mxnet',
                preProc = c("center", "scale"),
                #trace = FALSE,
                #allowParallel = TRUE,
                #repeats = 3,
                #eval.metric = mx.metric.mlogloss, 
                ctx = mx.gpu(),
                #eval.metric=mx.metric.mlogloss,
                #device = mx.gpu(),
                tuneGrid = expand.grid(layer1 = sample(2:20, replace = F, size = len),
                layer2 = sample(2:20, replace = F, size = len),
                layer3 = sample(2:20, replace = F, size = len),
                learning.rate = runif(len),
                momentum = runif(len),
                dropout = runif(len, max = .7),
                activation = c('relu', 'sigmoid', 'tanh', 'softrelu')),
                #weights = model_weights,
                metric = "ROC",
                trControl = ctrl)

require(ggplot2)
ggplot(neural$results, aes(x = activation, y= ROC)) + 
  geom_point(size = 2, shape = 18, alpha = .9, aes(colour = layer1)) +
  theme_bw()  + scale_color_gradient(low="blue", high="green")

?scale_color_continuous
ggplot(neural$results, aes(y = ROC, x=activation)) + 
  geom_boxplot(aes(colour = layer1))+#size = 2, shape = 18, alpha = .9, aes(colour = activation)) +
  theme_bw() 
#facet_wrap(~momentum)
results <- neural$results[neural$results$ROC > .625,]
results <- results[results$activation == 'sigmoid',]
results[with(results, order(-ROC)), ]
results <- results[results$layer1 == 16,]
neural$bestTune
#took 7 hours
#about 4 hours with gpu's
save(neural, file = 'mxnet.rda')
preds <- predict(neural, tester, type = 'prob')
head(preds)
NormalizedGini(preds[[1]], target)
# orig  down   up     weighted    
#.243   .2377  .227   .243
neural$bestTune
#layer1 layer2 layer3 learning.rate momentum    dropout activation
#2     14     20       0.24479       0.20376 0.07524117   softrelu



######################
#Here we get into mxnet without the caret package as the caret package limited abilities
######################
require(mxnet)


mLogLoss.normalize = function(p, min_eta=1e-15, max_eta = 1.0){
  #min_eta
  for(ix in 1:dim(p)[2]) {
    p[,ix] = ifelse(p[,ix]<=min_eta,min_eta,p[,ix]);
    p[,ix] = ifelse(p[,ix]>=max_eta,max_eta,p[,ix]);
  }
  #normalize
  for(ix in 1:dim(p)[1]) {
    p[ix,] = p[ix,] / sum(p[ix,]);
  }
  return(p);
}

# helper function
#calculates logloss
mlogloss = function(y, p, min_eta=1e-15,max_eta = 1.0){
  class_loss = c(dim(p)[2]);
  loss = 0;
  #p = mLogLoss.normalize(p,min_eta, max_eta);
  for(ix in 1:dim(y)[2]) {
    p[,ix] = ifelse(p[,ix]>1,1,p[,ix]);
    class_loss[ix] = sum(y[,ix]*log(p[,ix]));
    loss = loss + class_loss[ix];
  }
  #return loss
  return (list("loss"=-1*loss/dim(p)[1],"class_loss"=class_loss));
}

# mxnet specific logloss metric
mx.metric.mlogloss <- mx.metric.custom("mlogloss", function(label, pred){
 p = t(pred);
  m = mlogloss(class.ind(label),p);
  gc();
  return(m$loss);
})
# mx.metric.mlogloss <- NULL
# mydat <-  var_reduced_train
# mydat$ps_car_11_cat <- NULL
# train.x <- data.matrix(mydat[,-38])
# train.y <- mydat[,38]
# 
# data <- mx.symbol.Variable("data")
# label <- mx.symbol.Variable("label")
# fc1 <- mx.symbol.FullyConnected(data, num_hidden = 14, name = "fc1")
# tanh1 <- mx.symbol.Activation(fc1, act_type = "tanh", name = "tanh1")
# fc2 <- mx.symbol.FullyConnected(tanh1, num_hidden = 1, name = "fc2")
# lro <- mx.symbol.LinearRegressionOutput(fc2, name = "lro")



#Using V treat for kind of a black box standardization 
#since I didnt have time to go more in depth with the data
install.packages('vtreat')
require(vtreat)
?vtreat
yName <- 'target2'
yTarget <-  1
wholeReduced$target2 <- ifelse(wholeReduced$target2 == 'Not', 0, 1)
varNames <- setdiff(names(wholeReduced), 'target2')
treatmentsC <- designTreatmentsC(wholeReduced, varNames, yName, yTarget, verbose= F)
trainTreated <- prepare(treatmentsC, wholeReduced, pruneSig = c(), scale = T)
testTreated <-  prepare(treatmentsC, var_reduced_test, pruneSig = c(), scale = T)

var_reduced_train$target2 <- ifelse(var_reduced_train$target2 == 'Not', 0, 1)
var_reduced_train$target2 <- as.numeric(var_reduced_train$target2)
varNames <- setdiff(names(var_reduced_train), 'target2')
treatmentsC <- designTreatmentsC(var_reduced_train, varNames, yName, yTarget, verbose= F)
trainTreated <- prepare(treatmentsC, var_reduced_train, pruneSig = c(), scale = T)
testTreated <-  prepare(treatmentsC, var_reduced_test, pruneSig = c(), scale = T)

glmfit
load('train.rda')


mx.set.seed(1103)
neural <- train(x, factor(ifelse(y == 1, 'Default', 'Not')),
                #data = trainer,N
                #eval.metric = mx.metric.gini,
                method = 'mxnet',
                preProc = c("center", "scale"),
                #trace = FALSE,
                #allowParallel = TRUE,
                #eval.data = list("data"=x2,"label"=y2),
                ctx = mx.gpu(),
                #device = mx.gpu(),
                tuneGrid = expand.grid(activation = c('sigmoid'),
                                       layer1 = (16),
                                       layer2 = 8,
                                       layer3 =  9,
                                       learning.rate = .887525,
                                       momentum = .5618,
                                       dropout = c(.562)),#, .47429)),
                #weights = model_weights,
                
                #out_activation = 'logistic', # can be rmse or softmax
                #out_node = 1, 
                #optimizer = 'sgd', #stochastic gradient descent, update for each training example
                #metric = "ROC",
                trControl = ctrl)

pred <- predict(neural, x2,y2, type = 'prob')
NormalizedGini(pred[[1]], target)

mx.set.seed(1103)
x <- (data.matrix(trainTreated[,-87]))
y <- trainTreated[,87]
x2 <- (data.matrix(testTreated[,-87]))
y2 <- testTreated[,87]
dim(x)
table(y)

mx.metric.gini <- mx.metric.custom("gini", 
                                       function(label, pred){return(1-abs(NormalizedGini(label, pred)))})




dim(trainTreated)
x <- data.matrix(trainTreated[,-87])
y <- trainTreated[,87]
model <- mx.mlp(x, y,
                hidden_node=c(16, 8, 9), out_node=1,
                ctx = mx.gpu(), activation = 'sigmoid', 
                learning.rate= 0.1, 
                dropout = .5618, optimizer = 'rmsprop', 
                array.batch.size = 128, num.round = 1000, 
                out_activation = 'logistic',
                eval.metric = mx.metric.gini)

model
#do prediction
pred <- predict(model, X = x)
NormalizedGini(pred[1,], y)
label <- mx.io.extract(x, "label")
dataX <- mx.io.extract(x, "data")
mx.io
# Predict with R's array
pred2 <- predict(model, X=dataX)

model.re
table(wholeReduced$target2)
model
graph.viz(model$symbol)
plot(model)
mx.metric
model
pred2 <- predict(model, x)
gc(reset=T)
save(y2, file = 'forlap3.rda')
 plot(neural)
plot.nnet(neural$finalModel)
head(preds)
NormalizedGini(pred2, target)
head(pred2)
#.243
#.241 for reduced model
summary(pred2)
require(RCurl)

