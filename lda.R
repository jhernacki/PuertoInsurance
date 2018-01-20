 rm(list = ls())
require(MLmetrics)
require(MASS)
require(ROCR)
require(caret)
require(doSNOW)
library(caret) # for model-building
library(DMwR) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculations
ctrl <- NULL
ctrl <- trainControl(method = "cv",
                     number = 5,#10 used when my computer could handle it
                  
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

#######################
#       LDA           #
#######################

#Here you will see LDA ran many times, 
#each run represents different methods of handling the imbalanced data

#This same strategy was used for QDA and KNN however I'm only posting some of my code

target <- as.numeric(ifelse(wholeReduced$target2 == 'Not', 0, 1))

setwd('/home/jthernacki/Documents/Puerto_kaggle_insurance/lda/')
set.seed(1103)
orig_lda <- lda(target2 ~ ., data = wholeReduced)
orig_lda <- lda(x2, y2)
orig_lda
preds <- predict(orig_lda, wholeReduced, type = 'prob') 

NormalizedGini(preds[2][[1]][,1], target)
save(orig_lda, file = 'flippedorig_lda.rda')

#original LDA
set.seed(1103)
orig1_lda <- train(target2 ~ .,
                  data = wholeReduced,
                  method = "lda",
                  #prior = c(.8, .2),
                  verbose = T,
                  #preProcess=c('pca'),
                  metric = "ROC",
                  trControl = tc)

#save(orig1_lda, file = 'orig1_lda.rda')
orig1_lda

model_weights <- ifelse(trainer$target2 == 'Default', 
                        (1/table(trainer$target)[1])*.6, 
                        (1/table(trainer$target)[2])*.4)



sum(model_weights)
set.seed(1103)

#weighted LDA
weighted_lda <- train(target2 ~ .,
                      data = trainer,
                      method = "lda",
                      #preProcess=c('scale', 'center'),
                      weights = model_weights,
                      metric = "ROC",
                      trControl = ctrl)

#save(weighted_lda, file = 'weighted_lda.rda')

ctrl$sampling = 'down'
set.seed(1103)
#Down sampling LDA
down_lda <- train(target2 ~ . ,
                  data = trainer,
                  method = "lda",
                  #preProcess=c('scale', 'center'),
                  #weights = model_weights,
                  metric = "ROC",
                  trControl = ctrl)

#save(down_lda, file = 'down_lda.rda')

ctrl$sampling = 'up'
set.seed(1103)
#Up sampling LDA
up_lda <- train(target2 ~ . ,
                  data = trainer[,c(all_vars, 'target2')],
                  method = "lda",
                  #preProcess=c('scale', 'center'),
                  #weights = model_weights,
                  metric = "ROC",
                  trControl = ctrl)

#save(up_lda, file = 'up_lda.rda')

ctrl$sampling = 'smote'
set.seed(1103)
#SMOTE sampling LDA
smote_lda <- train(target2 ~.,
                  data = trainer,
                  method = "lda",
                  #preProcess=c('scale', 'center'),
                  #weights = model_weights,
                  metric = "ROC",
                  trControl = ctrl)

#save(smote_lda, file = 'smote_lda.rda')

load('orig_lda.rda')
load('orig1_lda.rda')
load('weighted_lda.rda')
load('up_lda.rda')
load('down_lda.rda')
load('smote_lda.rda')

########
#Comparing the different models
#########

ginis_lda <- rep(0, 6)
names(ginis_lda) <- c("orig", "origC", "weight", "down", "up", "smote")
models <- c(orig_lda, orig1_lda, weighted_lda, down_lda, up_lda, smote_lda)

preds <- predict(orig_lda, tester, type = 'prob') 
target <- ifelse(tester$target2 == 'Default', 1, 0)
target <- as.numeric(tester$target)
ginis_lda[1] <- NormalizedGini(preds[2][[1]][,1], tester$target)


preds <- predict(orig1_lda, var_reduced_test, type = 'prob') 
ginis_lda[2] <- NormalizedGini(preds[,1], target)

preds <- predict(weighted_lda, tester, type = 'prob') 
ginis_lda[3] <- NormalizedGini(preds[[1]], target)

preds <- predict(down_lda, tester, type = 'prob') 
ginis_lda[4] <- NormalizedGini(preds[[1]], tester$target)

preds <- predict(up_lda, tester, type = 'prob') 
ginis_lda[5] <- NormalizedGini(preds[[1]], tester$target)

preds <- predict(smote_lda, tester, type = 'prob') 
ginis_lda[6] <- NormalizedGini(preds[[1]], tester$target)


#orig      origC     weight       down         up      smote 
#0.2377194 0.2377194 0.2377194 0.2228751 0.2359431 0.2056960 
#ROC
#          0.621     0.621     0.604     0.6198    0.602
#No categoricals
#orig      origC     weight       down         up      smote 
#0.2377194 0.2377194 0.2377194 0.2200184 0.2365420 0.1999579 
#best Vars
#                    0.2384066           0.2368277




#################
#Playing with methods I dont have experience in
#################


model_weights <- ifelse(trainer$target2 == 'Default', 
                        (1/table(trainer$target2)[1])*.6, 
                        (1/table(trainer$target2)[2])*.4)
sum(model_weights)
table(small_trainer$target2)


cl <- makeCluster(8, type = 'SOCK')
registerDoSNOW(cl)
install.packages('klaR')
require(klaR)
fit <- loclda(target2 ~ ., data = small_trainer, k = 100, weighted.apriori = T)
set.seed(1103)
weighted_lda <- train(target2 ~ .,
                      data = small_trainer,
                      method = "loclda",
                      #preProcess=c('scale', 'center'),
                      weights = model_weights,
                      metric = "ROC",
                      trControl = ctrl)
save(weighted_lda, file = 'loclda.rda')
#did terrible

preds <- predict(fit, tester, type = 'prob') 
NormalizedGini(preds[[1]], target)

#lambdas .00001, .01, .1
weighted_lda <- train(target2 ~ .,
                      data = trainer,
                      method = "dwdLinear",
                      #preProcess=c('scale', 'center'),
                      #weights = model_weights,
                      metric = "ROC",
                      trControl = ctrl)

save(weighted_lda, file = 'dwdLinear.rda')
preds <- predict(weighted_lda, tester, type = 'prob') 
NormalizedGini(preds[[1]], target)
#.184

new <- createDataPartition(trainer$target2,times = 1,p = .9, list = F)
new <- trainer[-new,]
getModelInfo(method = 'bagFDA')
weighted_fda <- train(target2 ~ .,
                      data = trainer,
                      method = "bagFDA",
                    
                      #preProcess=c('scale', 'center'),
                     # weights = model_weights,
                      metric = "ROC",
                      #tuneGrid = expand.grid(degree = 1),
                      trControl = ctrl)

require(arm)
set.seed(1103)
weighted_bayes <- train(target2 ~ .,
                      data = trainer,
                      method = "bayesglm",
                      preProcess=c('scale', 'center'),
                      #weights = model_weights,
                      metric = "ROC",
                      #tuneGrid = expand.grid(degree = 1),
                      trControl = ctrl)

target <- as.numeric(tester$target)
preds <- predict(weighted_bayes, tester, type = 'prob') 
NormalizedGini(preds[[1]], target)

preds <- predict(weighted_bayes, var_reduced_test, type = 'prob') 
NormalizedGini(preds[[1]], tester$target)
#0.2185
stop(cl)

