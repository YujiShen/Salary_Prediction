setwd('/Users/Yuji/Workspace/R/Salary_Prediction/data')
library(data.table)
library(stringdist)
library(stringr)

### Salary
salary <- fread("Train_rev1.csv", na.strings = "", colClasses=list(character=c(2:4,10), integer=c(1,11)))
salary <- as.data.frame.matrix(salary)
salary <- transform(salary, LocationNormalized = as.factor(LocationNormalized),
                    ContractType = as.factor(ContractType),
                    ContractTime = as.factor(ContractTime),
                    Company = as.factor(Company),
                    Category = as.factor(Category),
                    SourceName = as.factor(SourceName))
save(salary, file = "salary.RData")

### Company Normalization

comNorm <- function(data){
  com <- as.character(data$Company)
  # com[72663] <- NA
  
  StopWords <- function(x) {
    removeWords(x, c("ltd", "llp", "plc", "limited", "inc", "co", "uk", "group", "services", "service", "and"))
  }
  com <- tolower(com)
  com <- removePunctuation(com)
  com <- StopWords(com)
  com <- stripWhitespace(com)
  com <- str_trim(com)
}

### Title Normalization
titleNorm <- function(data){
  title <- data$Title
  title[1589] <- ""
  title <- tolower(title)
  title <- removePunctuation(title)
  title <- stripWhitespace(title)
  title <- str_trim(title)
  title <- stemDocument(title, "english")
}

### Title Clustering
title.uni <- unique(salary$TitleNormalized)
title.uni <- as.data.frame(cbind(1:length(title.uni), title.uni))
names(title.uni) <- c("id", "title")
title.dist <- stringdistmatrix(title.uni[[2]], title.uni[[2]], method = "jaccard")

titleClu <- function(data, dist, clust){
  if(dist == "edit"){
    dist <- stringdistmatrix(data$TitleNormalized, data$TitleNormalized, method = "jaccard")
  }else if(dist == "td"){
    corpus <- Corpus(VectorSource(data$TitleNormalized))
    dtm <- DocumentTermMatrix(corpus)
    dist <- as.matrix(dtm)
  }
  
  if(clust == "hc"){
    hc <- hclust(as.dist(dist))
    #plot(hc)
    #rect.hclust(hc,k=20)
    cl = cutree(hc,k=100)
  }else if(clust == "kmeans"){
    cl <- kmeans(dist, 100)$cluster
  }
  return(cl)
}
hc <- titleClu(salary.sub, "edit", "hc")

### Location Matching - Edit distance

location.tree <- read.csv("Location_Tree_rev.csv", stringsAsFactors = F,header = F)
location.tree <- location.tree[[1]]
loc.raw <- salary$LocationRaw
loc.raw <- tolower(loc.raw)
loc.raw <- removePunctuation(loc.raw)
loc.tree.norm <- tolower(location.tree)
loc.tree.norm <- gsub(";", " ", loc.tree.norm)

findLocCos <- function(loc.raw){
  loc.dist <- stringdist(loc.raw, loc.tree.norm, method = "cosine")
  location.tree[which.min(loc.dist)]
}
loc.full.cos <- simplify2array(mclapply(loc.raw, function(x) findLocCos(x)))
#View(cbind(loc.raw, loc.full))

loc.full.cos <- read.csv(text = loc.full, stringsAsFactors = F, sep = ";", col.names = paste0("V",seq_len(7)), fill = T, header = F)


### Location Matching - Search Engine

location.tree <- read.csv("Location_Tree_rev.csv", stringsAsFactors = F,header = F)
location.tree <- location.tree[[1]]
loc.raw <- salary$LocationRaw
loc.raw <- tolower(loc.raw)
loc.raw <- removePunctuation(loc.raw)
loc.tree.norm <- tolower(location.tree)
loc.tree.norm <- gsub(";", " ", loc.tree.norm)

loc.uni <- unique(loc.raw)
loc.uni <- as.data.frame(cbind(1:length(loc.uni), loc.uni))
names(loc.uni) <- c("id", "location")

searchLoc <- function(loc.raw, loc.tree.norm, weighting){
  N.doc <- length(loc.tree.norm)
  N.query <- length(loc.raw)
  location <- c(loc.tree.norm, loc.raw)
  
  corpus <- Corpus(VectorSource(location))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  
  td.mat <- TermDocumentMatrix(corpus)
  if(weighting == "tfidf"){
    doc.mat <- weightTfIdf(td.mat[,1:N.doc], normalize = T)
  }else if(weighting == "smart"){
    doc.mat <- weightSMART(td.mat[,1:N.doc])
  }else if(weighting == "tf"){
    doc.mat <- weightTf(td.mat[,1:N.doc])
  }else if(weighting == "bin"){
    doc.mat <- weightBin(td.mat[, 1:N.doc])
  }
  
  query.mat <- td.mat[, (N.doc+1):(N.doc+N.query)]
  query.sp <- sparseMatrix(i = query.mat$i, j = query.mat$j, x = query.mat$v, dims = c(query.mat$nrow, query.mat$ncol))
  
  doc.sp <- sparseMatrix(i = doc.mat$i, j = doc.mat$j, x = doc.mat$v, dims = c(doc.mat$nrow, doc.mat$ncol))  
  score <- crossprod(query.sp, doc.sp)
  tree.ind <- NULL
  for(i in 1:length(loc.raw)){
    tree.ind <- c(tree.ind, which.max(score[i, ]))
  }
  loc.full <- location.tree[tree.ind]
}

loc.full.tfidf <- searchLoc(loc.raw, loc.tree.norm, "tfidf")

#View(cbind(loc.raw, loc.full.tfidf))

### Company Differece

comDiff <- function(train, test){
  median <- median(train$SalaryNormalized)
  com.ref <- train %>% 
    group_by(CompanyNormalized) %>%
    summarise(#n = n(),
      #min = min(SalaryNormalized),
      CompanyDiff = median(as.numeric(SalaryNormalized)) - median
      #mean = mean(SalaryNormalized),
      #max = max(SalaryNormalized)
    )
  #com.ref <- arrange(company, desc(n))
  
  train <- merge(x = train, y = com.ref, all.x = T)
  test <- merge(x = test, y = com.ref, all.x = T)
  test$CompanyDiff[is.na(test$CompanyDiff)] <- median(train$CompanyDiff)
  assign("train", train, envir = .GlobalEnv)
  assign("test", test, envir = .GlobalEnv)
}

### Location Difference

locDiff <- function(train, test){
  median <- median(train$SalaryNormalized)
  loc.ref <- train %>% 
    group_by(LocationNormalized) %>%
    summarise(#n = n(),
      #min = min(SalaryNormalized),
      LocationDiff = median(as.numeric(SalaryNormalized)) - median
      #mean = mean(SalaryNormalized),
      #max = max(SalaryNormalized)
    )
  #com.ref <- arrange(company, desc(n))
  
  train <- merge(x = train, y = loc.ref, all.x = T)
  test <- merge(x = test, y = loc.ref, all.x = T)
  test$LocationDiff[is.na(test$LocationDiff)] <- median(train$LocationDiff)
  assign("train", train, envir = .GlobalEnv)
  assign("test", test, envir = .GlobalEnv)
}

### Title Difference

titleDiff <- function(train, test){
  median <- median(train$SalaryNormalized)
  title.ref <- train %>% 
    group_by(TitleCluster) %>%
    summarise(#n = n(),
      #min = min(SalaryNormalized),
      TitleDiff = median(as.numeric(SalaryNormalized)) - median
      #mean = mean(SalaryNormalized),
      #max = max(SalaryNormalized)
    )
  #com.ref <- arrange(company, desc(n))
  
  train <- merge(x = train, y = title.ref, all.x = T)
  test <- merge(x = test, y = title.ref, all.x = T)
  test$TitleDiff[is.na(test$TitleDiff)] <- median(train$TitleDiff)
  assign("train", train, envir = .GlobalEnv)
  assign("test", test, envir = .GlobalEnv)
}

salary.sub$CompanyNormalized <- comNorm(salary.sub)
salary$CompanyNormalized <- comNorm(salary)
salary.sub$TitleNormalized <- titleNorm(salary.sub)
comDiff(train, test)
locDiff(train, test)
