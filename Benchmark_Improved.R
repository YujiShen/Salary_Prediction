setwd('/Users/Yuji/Workspace/R/Salary_Prediction/data')
load("salary.RData")

library(caret)
library(randomForest)
library(tm)
library(Matrix)

GetTrainFeatures <- function(train, column.names, topk, weighting){
  features <- NULL
  for(i in 1:length(column.names)){
    corpus <- Corpus(VectorSource(train[, column.names[i]]))
    
    StopWords <- function(x) {
      removeWords(x, c(stopwords("SMART"), "england"))
    }
    funs <- list(stemDocument, StopWords, removeNumbers, stripWhitespace, removePunctuation, content_transformer(tolower))
    corpus <- tm_map(corpus, FUN = tm_reduce, tmFuns = funs)
    if(weighting == T){
      tdm <- DocumentTermMatrix(corpus, control=list(weighting = weightSMART))
    }else{
      tdm <- DocumentTermMatrix(corpus)
    }
    temp <- sparseMatrix(i=tdm$i, j=tdm$j, x=tdm$v)
    sums <- order(colSums(temp), decreasing = TRUE)
    tdm$dimnames$Terms[sums[1:topk]]
    tdm.top <- tdm[, sums[1:topk]]
    features <- cbind(features, tdm.top)
  }
  return(features)
}

GetTestFeatures <- function(test, dic, column.names, weighting){
  features <- NULL
  for(i in 1:length(column.names)){
    corpus <- Corpus(VectorSource(test[, column.names[i]]))
    
    StopWords <- function(x) {
      removeWords(x, c(stopwords("SMART"), "england"))
    }
    funs <- list(stemDocument, StopWords, removeNumbers, stripWhitespace, removePunctuation, content_transformer(tolower))
    corpus <- tm_map(corpus, FUN = tm_reduce, tmFuns = funs)
    
    if(weighting == T){
      tdm <- DocumentTermMatrix(corpus, control=list(weighting = weightSMART, dictionary=dic[[i]]))
    }else{
      tdm <- DocumentTermMatrix(corpus, control = list(dictionary=dic[[i]]))
    }
    temp <- tdm[, dic[[i]]]
    features <- cbind(features, temp)
  }
  return(features)
}

set.seed(100)
split <- createDataPartition(salary$SalaryNormalized, times = 1, p = 0.1, list = F)
salary.sub <- salary[split, ]
set.seed(100)
split.train <- createDataPartition(salary.sub$SalaryNormalized, times = 1, p = 0.7, list = F)
train <- salary.sub[split.train, ]
test <- salary.sub[-split.train, ]

column.names <- c("FullDescription", "Title", "LocationRaw")
topk <- 100

train.fea <- GetTrainFeatures(train, column.names, topk, T)
train.dic <- list(train.fea$dimnames[[2]][1:topk],
                  train.fea$dimnames[[2]][(topk+1):(2*topk)],
                  train.fea$dimnames[[2]][(2*topk+1):(3*topk)],
                  train.fea$dimnames[[2]][(3*topk+1):(4*topk)])

train.data <- as.data.frame(as.matrix(train.fea))
train.data <- cbind(SalaryNormalized = train$SalaryNormalized, train.data, train$Category)
colnames(train.data) <- paste("V", 1:(length(column.names)*topk+2), sep="")
set.seed(100)
rf <- randomForest(V1 ~ ., data = train.data, ntree = 50, importance = T)
varImpPlot(rf)
imp <- importance(rf)
rownames(imp) <- c(train.dic[[1]], train.dic[[2]], train.dic[[3]], "Category")

test.fea <- GetTestFeatures(test, train.dic, column.names, T)
test.data <- data.frame(as.matrix(test.fea))
test.data <- cbind(SalaryNormalized = test$SalaryNormalized, test.data, test$Category)
colnames(test.data) <- paste("V", 1:(length(column.names)*topk+2), sep="")
rf.pred <- predict(rf, newdata = test.data)
rf.mae <- mean(abs(rf.pred - test.data[[1]]))



#################
train.data <- as.data.frame(as.matrix(train.fea))
train.data <- cbind(SalaryNormalized = log(train$SalaryNormalized), train.data)
colnames(train.data) <- make.names(colnames(train.data), unique = T)
lm <- lm(SalaryNormalized ~., data = train.data)
test.data <- data.frame(as.matrix(test.fea))
test.data <- cbind(SalaryNormalized = log(test$SalaryNormalized), test.data)
colnames(test.data) <- colnames(train.data)
lm.pred <- predict(lm, newdata = test.data)
lm.mae <- mean(abs(exp(lm.pred) - exp(test.data[[1]])))


fmla <- formula(paste("SalaryNormalized ~", paste(names(lm$coef[summary(lm)$coef[,4] <= .05])[-1], collapse = " + ")))
lm.sig <- lm(fmla, data = train.data)
summary(lm.sig)
lm.sig.pred <- predict(lm.sig, newdata = test.data)
lm.sig.mae <- mean(abs(exp(lm.sig.pred) - exp(test.data[[1]])))

##############
train.data <- as.data.frame(as.matrix(train.fea))
test.data <- as.data.frame(as.matrix(test.fea))
pca <- prcomp(rbind(train.data, test.data), scale = T)
pc <- predict(pca)
cat("### Rotation of PCs ###")
pca$rotation
dat1.pca <- cbind(dat1.log[1], pc.crime)
apply(pca$rotation)
plot(pca, main = "Screeplot of Crime")
train.data <- as.data.frame(cbind(SalaryNormalized = log(train$SalaryNormalized), pc[1:nrow(train.data),pca$sdev^2 > 1]))
lm.pca <- lm(SalaryNormalized ~ ., train.data)
test.data <- as.data.frame(cbind(SalaryNormalized = log(test$SalaryNormalized), pc[(nrow(train.data)+1):(nrow(train.data)+nrow(test.data)),pca$sdev^2 > 1]))
lm.pca.pred <- predict(lm.pca, newdata = test.data)
lm.pca.mae <- mean(abs(exp(lm.pca.pred) - exp(test.data[[1]])))
