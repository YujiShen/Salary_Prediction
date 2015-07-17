library(topicmodels)

set.seed(100)
split.train <- createDataPartition(salary$SalaryNormalized, times = 1, p = 0.01, list = F)
train <- salary[split.train, ]
test <- salary.sub[-split.train, ]

column.names <- c("FullDescription", "Title", "LocationRaw", "LocationNormalized")


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
      dtm <- DocumentTermMatrix(corpus, control=list(weighting = weightSMART))
    }else{
      dtm <- DocumentTermMatrix(corpus)
    }
    
    if(topk == 0){
      features <- cbind(features, dtm)
    }else{
      temp <- sparseMatrix(i=dtm$i, j=dtm$j, x=dtm$v)
      sums <- order(colSums(temp), decreasing = TRUE)
      dtm$dimnames$Terms[sums[1:topk]]
      dtm.top <- dtm[, sums[1:topk]]
      features <- cbind(features, dtm.top)
    }
  }
  return(features)
}

train.fea <- GetTrainFeatures(train, "FullDescription", 0, F)

lda <- LDA(train.fea, 50)
topic <- topics(lda)
term <- terms(lda, 3)

train.topic <- as.data.frame(cbind(SalaryNormalized  = train$SalaryNormalized))
train.topic$topic <- as.factor(topic)
train.topic$loc <- as.factor(train$LocationNormalized)

train.cat <- as.data.frame(cbind(SalaryNormalized = train$SalaryNormalized))
train.cat$Category <- as.factor(train$Category)
lm.cat <- lm(SalaryNormalized ~ ., train.cat)
lm.topic <- lm(SalaryNormalized ~ ., train.topic)
summary(lm.topic)

