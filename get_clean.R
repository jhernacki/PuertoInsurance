rm(list = ls())
install.packages('randomForest')
require(randomForest)
require(gbm)
require(glmnet)
require(ggplot2)
require(RANN)
require(GGally)
require(corrplot)
setwd('/home/jthernacki/Documents/Puerto_kaggle_insurance/')
train <- read.csv('/home/jthernacki/Documents/Puerto_kaggle_insurance/train.csv', na.strings = '-1')
test <- read.csv('/home/jthernacki/Documents/Puerto_kaggle_insurance/test.csv', na.strings = '-1')
save(test, file = 'test.rda')
save(train, file = 'train.R')

load(file = 'pp.rda')
load(file = 'new_train.rda')
train <- ins_data
str(train)
###############
#  Cleaning   #
###############
#making the factor variables
binary <- grep('bin' , colnames(train))
categorical <- grep('cat',  colnames(train))
train$target <- factor(train$target)
for(i in binary){
  train[,i] <- factor(train[,i])
}

for(i in categorical){
  train[,i] <- factor(train[,i])
}


#Missing values-----------------------------------------------------
install.packages(mice)
require(mice)
md.pattern(train_1)

prop.table(table(train$target))
train <- ins_data
#find missing value issues
for(i in colnames(train)){
  temp <- sum(is.na(train[,i]))
  if(temp > length(train[,1])*.1){ #.1 represents how many it needs missing in order to be printed
    cat(i, temp)
    cat("\n")
  }
}
table(train$ps_car_07_cat)

#Since these variables had many missing values, convert them to binary predictors of missing or not missing
#For future I would attempt to use imputation methods
ps_car_03_cat_isNull <- ifelse(is.na(train$ps_car_03_cat), 1, 0)
ps_car_05_cat_isNull <- ifelse(is.na(train$ps_car_05_cat), 1, 0)
ps_car_07_cat_isNull <- ifelse(is.na(train$ps_car_07_cat), 1, 0)

train$ps_car_03_cat_isNull <- NULL
train$ps_car_05_cat_isNull <- NULL
train$ps_car_07_cat_isNull <- NULL
tester <- train$target
id <- train$id
train$target <- NULL
train$id <- NULL
#per protocal
#car 3, car 7, ind 5 are influential, maybe multiple models for different situations
#remove car 11 as well because it has too many categories
train_1 <- subset(train, select = -c(ps_car_03_cat, ps_car_05_cat, ps_car_07_cat, ps_ind_05_cat, ps_car_11_cat))
pp <- na.omit(train_1)
save(pp, file = 'pp.rda')
length(pp$target)
pp_tiny <- pp[sample(nrow(pp), 50000),]
load('train.rda')
table(ins_data$target)
table(wholeReduced$target2)

#########################
#   Correlations
#########################
load('pp.rda')
train <- pp
shrink <- sample(1:length())
#The dataset was separated into 4 unexplained groups
car <- train[,grep('car', colnames(train))]
ind <- train[,grep('ind', colnames(train))]
reg <- train[,grep('reg', colnames(train))]
calc <- train[,grep('calc', colnames(train))]

car <- sapply( car, as.numeric )
corrplot.mixed(cor(car), lower.col = "black", number.cex = .7)

ind <- sapply( ind, as.numeric )
corrplot.mixed(cor(ind), lower.col = "black", number.cex = .7)
#14 and 12 bin have .89
train$target <- factor(train$target)
ggplot(tester, aes(y = ps_ind_14, x = target)) +
  theme_bw() +
  facet_wrap(~ ps_ind_12_bin) +
  geom_boxplot() +
  labs(x = "target",
       y = "ind_14",
       title = "Target by ind_12 and ind_14")

reg <- sapply( reg, as.numeric )
corrplot.mixed(cor(reg), lower.col = "black", number.cex = .7)

calc <- sapply( calc, as.numeric )
corrplot.mixed(cor(calc), lower.col = "black", number.cex = .7)
#No very significant correlations were found other than the ones labeled above

#################
#drop vars/variable reduction
#################
fit <- glm(target~ . -id -target2,data = trainer, family = binomial)
summary(fit)
#ps_ind 9 and 14 are linear combinations of other variables
trainer$ps_ind_09_bin <- NULL
trainer$ps_ind_14 <- NULL
tester$ps_ind_09_bin <- NULL
tester$ps_ind_14 <- NULL

trainer$target <- as.numeric(trainer$target)
fit <-  lm(ps_ind_01~ . -id -target2,data = trainer)
vif(fit)
#Nothing bigger than 4
trainer$target <- factor(trainer$target)

#########
#Take the top 30 from glmnet, gbm, and random forest to reduce some of the variables
####
#glm NET
x <- model.matrix(target2 ~ . ,data = trainer)[,-1]
y <- trainer$target2
objControl <- trainControl(method='cv', number=3, 
                           summaryFunction = twoClassSummary, 
                           returnResamp='none', classProbs = T)
objModel <- train(x, y, method='glmnet',  
                  metric = "ROC", trControl=objControl)
#13 alpha values tried with all seq(0, .9, .1) using cv choosing best ROC optomizer
predictions <- predict(object=objModel, x)
vars <- as.data.frame(varImp(objModel,scale=F)$importance)
t <- rownames(vars)
s <- vars[1:202,1]
names(s) <- t
s <- sort(s, decreasing = T)
glmnetVars <- names(s[1:30])


#Generalized Boosted Models
objControl <- trainControl(method='cv', number=3, 
                           summaryFunction = twoClassSummary, 
                           returnResamp='none', classProbs = T)
objModel <- train(x, y, method='gbm',
                  preProc = c("center", "scale"),  
                  metric = "ROC", 
                  trControl=objControl)
plot(objModel)
#.1 shrinkage, 1:3 interaction.depth(how many splits per), ntrees c(50, 100, 150)
save(objModel, file = 'gbm.rda')
predictions <- predict(object=objModel, x)
names(plot(varImp(objModel,scale=F), top = 30))
glmNetVars <- as.data.frame(varImp(objModel,scale=F))
vars <- as.data.frame(varImp(objModel,scale=F)$importance)
t <- rownames(vars)
s <- vars[1:202,1]
names(s) <- t
s <- sort(s, decreasing = T)
gbmVars <- names(s[1:30])

#random forest (preRan), mtry sqrt(p), ntree = 500
load('rf_all.rda')
plot(varImp(fit,scale=T), top = 30)
varImpPlot(fit,type=2)
vars <- (fit$importance)
vars <- as.data.frame(vars)
t <- rownames(vars)
s <- vars[1:52,1]
names(s) <- t
s <- sort(s, decreasing = T)
randomForestVars <- names(s[1:30])

save(glmnetVars, file = 'glmNetvarsstring.rda')
save(randomForestVars, file = 'randomForestvarsstring.rda')


#recreate the ps_car_11_cat variable (see knn for fullx), turn 100 categories to just 7
#If I had more detail I would attempt to generalize the groups rather than deleting many

keepers <- c('ps_car_11_cat17', 'ps_car_11_cat21','ps_car_11_cat29',
             'ps_car_11_cat41',  'ps_car_11_cat86', 'ps_car_11_cat93')
creater <- fullx[,keepers]
new_car_11_cat <- ifelse(creater[,1] == 1, 'A', 
                  ifelse(creater[,2] == 1, 'B', 
                  ifelse(creater[,3] == 1, 'C', 
                  ifelse(creater[,4] == 1, 'D', 
                  ifelse(creater[,5] == 1, 'E', 
                  ifelse(creater[,6] == 1, 'F', 'G'))))))

