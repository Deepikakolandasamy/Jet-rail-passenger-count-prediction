library(xgboost)
library(lubridate)
library(dplyr)
jet <- read.csv("F:/R/main proj/JetRail Hackathon/Train_SU63ISt.csv")
jet$Datetime <- dmy_hm(jet$Datetime)
jet$year <- year(jet$Datetime)
jet$month<- month(jet$Datetime)
jet$day <- day(jet$Datetime)
jet$weekday <- wday(jet$Datetime)
jet$hour <- hour(jet$Datetime)
jet$weekend <- ifelse(jet$weekday == "7" | jet$weekday == "1",1,0)
str(jet)
jet$monthend <- ifelse(jet$day > 27,1,0)
jet$monthstart <- ifelse(jet$day <5,1,0)
jet$yearend <- ifelse(jet$month == 12,1,0)
jet$yearstart <- ifelse(jet$month == 1,1,0)
str(jet)
jet_model <- jet[,c('ID','day','month','year','hour','weekday','weekend','yearstart','yearend','monthstart','monthend','Count')]


testcheck <- read.csv("F:/R/main proj/JetRail Hackathon/Test_0qrQsBZ.csv")
testcheck$Datetime <- dmy_hm(testcheck$Datetime)
testcheck$year <- year(testcheck$Datetime)
testcheck$month<- month(testcheck$Datetime)
testcheck$day <- day(testcheck$Datetime)
testcheck$weekday <- wday(testcheck$Datetime)
testcheck$hour <- hour(testcheck$Datetime)
testcheck$weekend <- ifelse(testcheck$weekday == "7" | testcheck$weekday == "1",1,0)
str(testcheck)
testcheck$monthend <- ifelse(testcheck$day > 27,1,0)
testcheck$monthstart <- ifelse(testcheck$day <5,1,0)
testcheck$yearend <- ifelse(testcheck$month == 12,1,0)
testcheck$yearstart <- ifelse(testcheck$month == 1,1,0)
str(testcheck)
testcheck_model <- testcheck[,c('ID','day','month','year','hour','weekday','weekend','yearstart','yearend','monthstart','monthend')]
install.packages('HH')
library(HH)
vif(jet_model)
#--------------------------------------------------------------------------------------------------#

library(caret)
folds = createFolds(jet_model$Count, k=10) 
library(xgboost)
jet_model[12]

set.seed(123)
cv = lapply(folds,function(x) {
  jet_train_fold = jet_model[-x,]
  jet_test_fold = jet_model[x,]
  classifier = xgboost(
    data = as.matrix(jet_train_fold[-12]),
    label = jet_train_fold$Count, nrounds = 50)
  y_pred <- predict(classifier, as.matrix(jet_test_fold[-12]))
  RMSE <- sqrt(mean((y_pred-jet_test_fold[12])^2))
  return(RMSE)
})

Linear = lapply(folds,function(x){
  jet_train_fold = jet_model[-x,]
  jet_test_fold = jet_model[x,]
  Linear_model  = lm(Count~., data = jet_train_fold)
  y_pred_linear <- predict(Linear_model, jet_test_fold[-12])
  RMSE <- sqrt(mean((y_pred_linear-jet_test_fold[12])^2))
  return(RMSE)
})

RMSE_mean = mean(as.numeric(cv))
RMSE_Linear = mean(as.numeric(Linear))
#--------------------------------------------------------------------------------------------------#

#Fold 5 gives better RMSE Value - XGBoosting

folds$Fold05
jet_test = jet_model[(folds$Fold05),]
jet_train = jet_model[-(folds$Fold05),]

classifier_jet = xgboost(
  data = as.matrix(jet_train[-12]),
  label = jet_train$Count, nrounds = 50)
y_pred <- predict(classifier_jet, as.matrix(jet_test[-12]))


RMSE <- sqrt(mean((y_pred-jet_test[12])^2))

#--------------------------------------------------------------------------------------------------#
#Fold 7 gives better RMSE  Value - Linear Modeling

folds$Fold07
jet_linear_test = jet_model[(folds$Fold07),]
jet_linear_train = jet_model[-(folds$Fold07),]
linear_classifier = lm(Count~., data = jet_linear_train)
y_linear_pred <- predict(linear_classifier,jet_linear_test[-12])
linear_RMSE <- sqrt(mean((y_linear_pred-jet_linear_test[12])^2))

#--------------------------------------------------------------------------------------------------#

str(jet_model)
str(testcheck_model)
Datetime <- testcheck$Datetime
testcheck <- NULL
testcheck_model$Predicting <- NULL
testcheck_model$Predicting <- predict(classifier_jet,as.matrix(testcheck_model))

#--------------------------------------------------------------------------------------------------#
?as.Date.IDate
dt=as.Date()

#date <- as.Date(with(testcheck_model, paste(year, month,day,hour,,sep="-")), "%Y-%m-%d %H-%M")
date<-with(testcheck_model, ymd_h(paste(year, month, day, hour ,sep= ' ')))
#Datetime <- as.Date(with(testcheck_model,paste(date,hour,sep=" "),"%-%"))
submission_data <- testcheck_model
#submission_data <- as.data.frame(submission_data)
submission_data$date <- date
Count <- testcheck_model$Predicting
Linear_Count <- predict(linear_classifier,as.data.frame(testcheck_model))
#testcheck_model <- Count
str(testcheck_model)
str(submission_data)
submission_data$XGBCount <- (round(Count))
submission_data$LinearCount <-(round(Linear_Count))
