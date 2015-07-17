setwd('/Users/Yuji/Workspace/R/Salary_Prediction/data')
load("salary.RData")

library(caret)
library(randomForest)
library(tm)
library(Matrix)

### Mean Benchmark
salary.mean <- mean(salary$SalaryNormalized)
MAE <- mean(abs(salary$SalaryNormalized - salary.mean))

### Random Forest Benchmark

set.seed(100)
split <- createDataPartition(salary$SalaryNormalized, times = 1, p = 0.1, list = F)
salary.sub <- salary[split, ]
set.seed(100)
split.train <- createDataPartition(salary.sub$SalaryNormalized, times = 1, p = 0.7, list = F)
train <- salary.sub[split.train, ]
test <- salary.sub[-split.train, ]

GetTrainFeatures <- function(train, column.names, topk){
  features <- NULL
  for(i in 1:length(column.names)){
    corpus <- Corpus(VectorSource(train[, column.names[i]]))
    
    StopWords <- function(x) {
      removeWords(x, stopwords("SMART"))
    }
    funs <- list(StopWords, removeNumbers, stripWhitespace, removePunctuation, content_transformer(tolower))
    corpus <- tm_map(corpus, FUN = tm_reduce, tmFuns = funs)
    tdm <- DocumentTermMatrix(corpus)  
    temp <- sparseMatrix(i=tdm$i, j=tdm$j, x=tdm$v)
    sums <- order(colSums(temp), decreasing = TRUE)
    tdm$dimnames$Terms[sums[1:topk]]
    tdm.top <- tdm[, sums[1:topk]]
    features <- cbind(features, tdm.top)
  }
  return(features)
}

GetTestFeatures <- function(test, dic, column.names){
  features <- NULL
  for(i in 1:length(column.names)){
    corpus <- Corpus(VectorSource(test[, column.names[i]]))
    
    StopWords <- function(x) {
      removeWords(x, stopwords("SMART"))
    }
    funs <- list(StopWords, removeNumbers, stripWhitespace, removePunctuation, content_transformer(tolower))
    corpus <- tm_map(corpus, FUN = tm_reduce, tmFuns = funs)
    tdm <- DocumentTermMatrix(corpus, control = list(dictionary=dic[[i]]))
    temp <- tdm[, train.dic[[i]]]
    features <- cbind(features, temp)
  }
  return(features)
}

column.names <- c("FullDescription", "Title", "LocationRaw", "LocationNormalized")
topk <- 100



error <- NULL
set.seed(100)
flds <- createFolds(salary.sub$SalaryNormalized, k = 10, list = TRUE, returnTrain = FALSE)
for(i in 1:length(flds)){
  train <- salary.sub[-flds[[i]], ]
  test <- salary.sub[flds[[i]], ]
  
  train.fea <- GetTrainFeatures(train, column.names, topk)
  train.dic <- list(train.fea$dimnames[[2]][1:topk],
                    train.fea$dimnames[[2]][(topk+1):(2*topk)],
                    train.fea$dimnames[[2]][(2*topk+1):(3*topk)],
                    train.fea$dimnames[[2]][(3*topk+1):(4*topk)])
  
  train.data <- as.data.frame(as.matrix(train.fea))
  train.data <- cbind(train.data, SalaryNormalized = train$SalaryNormalized)
  colnames(train.data) <- make.names(train.data, unique = T)
  rf <- randomForest(SalaryNormalized ~ ., data = train.data,ntree = 50, nodesize = 25)
  
  test.fea <- GetTestFeatures(test, train.dic, column.names)
  test.data <- data.frame(as.matrix(test.fea))
  test.data <- cbind(test.data, SalaryNormalized = test$SalaryNormalized)
  colnames(test.data) <- make.names(test.data, unique = T)
  rf.pred <- predict(rf, newdata = test.data)
  rf.mae <- mean(abs(rf.pred - test.data$SalaryNormalized))
  mean.mae <- mean(abs(mean(test.data$SalaryNormalized) - test.data$SalaryNormalized))
  error <- rbind(error, cbind(rf.mae, mean.mae))
}

###########
tfidf.desc <- TermDocumentMatrix(corpus.desc, control=list(bounds=list(global=c(floor(length(corpus)*0.05), Inf)), weighting = weightTfIdf))
temp2 <- sparseMatrix(i=tfidf.desc$i, j=tfidf.desc$j, x=tfidf.desc$v)
sums2 <- order(rowSums(temp2), decreasing = TRUE)
tfidf.desc$dimnames$Terms[sums2[1:100]]

