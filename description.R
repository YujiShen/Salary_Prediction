
GetTrainFeatures <- function(train, column.names, topk, weighting){
  features <- NULL
  for(i in 1:length(column.names)){
    corpus <- Corpus(VectorSource(train[, column.names[i]]))
    
    StopWords <- function(x) {
      removeWords(x, c(stopwords("SMART")))
    }
    funs <- list(stemDocument, StopWords, removeNumbers, stripWhitespace, removePunctuation, content_transformer(tolower))
    corpus <- tm_map(corpus, FUN = tm_reduce, tmFuns = funs)
    if(weighting == T){
      tdm <- DocumentTermMatrix(corpus, control=list(weighting = weightBin))
    }else{
      tdm <- DocumentTermMatrix(corpus)
    }
    temp <- sparseMatrix(i=tdm$i, j=tdm$j, x=tdm$v)
    sums <- order(colMeans(temp), decreasing = TRUE)
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
      removeWords(x, c(stopwords("SMART")))
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

######
desc <- GetTrainFeatures(train, "FullDescription", 500, T)
desc$dimnames[[2]]
desc.data <- as.data.frame(as.matrix(desc))
desc.data <- cbind(SalaryNormalized = train$SalaryNormalized, desc.data)
colnames(desc.data) <- make.names(colnames(desc.data), unique = T)
set.seed(100)
desc.rf <- randomForest(SalaryNormalized ~ ., data = desc.data, ntree = 50, importance = T)
varImpPlot(desc.rf)

desc.test <- GetTestFeatures(test, list(desc$dimnames[[2]]), "FullDescription", T)
desc.test.data <- as.data.frame(as.matrix(desc.test))
desc.test.data <- cbind(SalaryNormalized = test$SalaryNormalized, desc.test.data, test$Category)
colnames(desc.test.data) <- colnames(desc.data)
desc.rf.pred2 <- predict(desc.rf, newdata = desc.test.data)
desc.rf.mae <- mean(abs(desc.rf.pred2 - desc.test.data[[1]]))


desc.project <- grepl("project", train$FullDescription)
train$FullDescription[desc.project][1:10]
desc.rf.pred <- predict(desc.rf, desc.data)
mean(desc.rf.pred[desc.project])
mean(train$SalaryNormalized[desc.project])
mean(desc.rf.pred)

desc.project <- grepl("project", train$FullDescription)
train$FullDescription[desc.project][1:10]
desc.rf.pred <- predict(desc.rf, desc.data)
desc.rf.pred2 <- predict(desc.rf, desc.test)
mean(desc.rf.pred[desc.project])
mean(desc.rf.pred)

######## dtm then split
desc.sub <- GetTrainFeatures(salary.sub, "FullDescription", 500, T)
title.sub <- GetTrainFeatures(salary.sub, "Title", 1000, T)
title.sub.data <- scale(as.data.frame(as.matrix(title.sub)))
set.seed(100)
km <- kmeans(title.sub.data, centers = 100, iter.max = 50)

title.dist <- dist(title.sub)

desc.sub.data <- as.data.frame(as.matrix(desc.sub))
colnames(desc.sub.data) <- make.names(colnames(desc.sub.data), unique = T)
desc.sub.data$Title <- km$cluster
desc.sub.data$SalaryNormalized <- salary.sub$SalaryNormalized
desc.train <- desc.sub.data[split.train, ]
desc.test <- desc.sub.data[-split.train, ]
set.seed(100)
desc.sub.rf <- randomForest(SalaryNormalized ~ ., data = desc.train, ntree = 50, importance = T)
desc.sub.pred <- predict(desc.sub.rf, newdata = desc.test)
desc.sub.mae <- mean(abs(desc.sub.pred - desc.test$SalaryNormalized))
varImpPlot(desc.sub.rf)

imp.var <- intersect(rownames(importance(desc.sub.rf))[order(importance(desc.sub.rf, type = 1), decreasing = T)][1:100], rownames(importance(desc.sub.rf))[order(importance(desc.sub.rf, type = 2), decreasing = T)][1:100])

####### kmeans
desc.sub1 <- GetTrainFeatures(salary.sub, "FullDescription", 100, T)

desc.sub.data <- scale(as.data.frame(as.matrix(desc.sub1)))
set.seed(100)
km <- kmeans(desc.sub.data, centers = 30, iter.max = 50)

desc.sub.data1 <- as.data.frame(as.matrix(desc.sub1))
colnames(desc.sub.data1) <- make.names(colnames(desc.sub.data1), unique = T)
desc.sub.data1$cluster <- km$cluster
desc.sub.data1$SalaryNormalized <- salary.sub$SalaryNormalized
desc.train1 <- desc.sub.data1[split.train, ]
desc.test1 <- desc.sub.data1[-split.train, ]
set.seed(100)
desc.sub.rf1 <- randomForest(SalaryNormalized ~ ., data = desc.train1, ntree = 50, importance = T)
desc.sub.pred1 <- predict(desc.sub.rf1, newdata = desc.test1)
desc.sub.mae1 <- mean(abs(desc.sub.pred1 - desc.test1$SalaryNormalized))
varImpPlot(desc.sub.rf1)
importance(desc.sub.rf1, type = 1)[order(importance(desc.sub.rf1, type = 1), decreasing = T),]

#####
gbm = gbm(SalaryNormalized ~., desc.train,
          n.trees=100,
          shrinkage=0.01,
          distribution="gaussian",
          interaction.depth=7,
          bag.fraction=0.5,
          cv.fold=3,
          n.minobsinnode = 50
)
summary(gbm)
desc.gbm.pred <- predict(gbm, newdata = desc.test)
desc.gbm.mae <- mean(abs(desc.gbm.pred - desc.test$SalaryNormalized))
