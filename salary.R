setwd('/Users/Yuji/Workspace/R/Salary_Prediction/data')

library(tm)
library(Matrix)
library(lsa)
library(ggplot2)

load("salary.RData")

location.tree <- read.csv("Location_Tree_rev.csv", stringsAsFactors = F, sep = ";", col.names = paste0("V",seq_len(7)), fill = T, header = F)

df_args <- c(location.tree, sep=" ")
location <- do.call(paste, df_args)
N.doc <- length(location)
N.query <- nrow(salary)
location <- c(location, salary$LocationRaw)

corpus <- Corpus(VectorSource(location))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)

inspect(corpus[(N.doc + 1):(N.doc + 10)])

td.mat <- TermDocumentMatrix(corpus)

tfidf.mat <- weightTfIdf(td.mat[,1:N.doc], normalize = T)
query.mat <- td.mat[, (N.doc+1):(N.doc+N.query)]

query.sp <- sparseMatrix(i = query.mat$i, j = query.mat$j, x = query.mat$v, dims = c(query.mat$nrow, query.mat$ncol))
doc.sp <- sparseMatrix(i = tfidf.mat$i, j = tfidf.mat$j, x = tfidf.mat$v, dims = c(tfidf.mat$nrow, tfidf.mat$ncol))

score <- crossprod(query.sp, doc.sp)
tree.ind <- apply(score, 1, which.max)

location.tree[test,]
train$LocationRaw[1:10]
#train <- read.csv("Train_rev1.csv")
#test <- read.csv("Test_rev1.csv")
#valid <- read.csv("Valid_rev1.csv")

head(train)
str(train)
head(train$Title)

## Report
colSums(is.na(train))

ggplot(aes(x = SalaryNormalized), data = train) +
  geom_histogram() +
  scale_x_log10()

ggplot(aes(x = SalaryNormalized), data = train) +
  geom_histogram() +
  scale_x_log10() +
  facet_grid(ContractType ~ .)

ggplot(aes(x = SalaryNormalized), data = train) +
  geom_histogram() +
  facet_grid(ContractTime ~ .)

ggplot(aes(x = Category, y = SalaryNormalized), data = train) +
  geom_boxplot() +
  scale_y_log10() +
  theme(axis.text.x=element_text(angle = -90, hjust = 0))

corpus1 <- Corpus(VectorSource(train$Title[1:100]))
corpus1 <- tm_map(corpus1, content_transformer(tolower))
corpus1 <- tm_map(corpus1, removePunctuation)
corpus1 <- tm_map(corpus1, function(x) removeWords(x, stopwords("english")))
corpus1 <- tm_map(corpus1, stemDocument, language = "english")
inspect(corpus[1:10])
td.mat1 <- as.matrix(TermDocumentMatrix(corpus1))
sort(cosine(td.mat1[,2:5], td.mat1), index.return = T, decreasing = T)$ix[1:5]

corpus <- Corpus(VectorSource(train$Title))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, function(x) removeWords(x, stopwords("english")))
corpus <- tm_map(corpus, stemDocument, language = "english")
inspect(corpus[1:10])
td.mat <- TermDocumentMatrix(corpus)
mat <- cosine(td.sm)
td.sm <- sparseMatrix(i = td.mat$i, j = td.mat$j, x = td.mat$v, dims = c(td.mat$nrow, td.mat$ncol))
td.sm <- drop0(td.sm)
row.norm <- sqrt(rowSums(td.sm^2))
row.norm <- t(crossprod(sign(td.sm), Diagonal(x=row.norm)))
row.norm@x <- 1/row.norm@x
td.sm.norm <- td.sm * row.norm
sim <- tcrossprod(td.sm.norm)

m <- matrix(1:12, ncol=4)
l <- sqrt(colSums(m^2))
t(m) %*% m / (l%*% t(l))
m <- t(c(-1, 1, -1, 1) * t(m))

mm <- as(m, "dgCMatrix")
ll <- as(sqrt(colSums(mm^2)), "sparseVector")
ll.m <- tcrossprod(ll)
ll.m@x <- 1/ll.m@x
sim <- crossprod(mm) * ll.m

vlength <- sqrt(colSums(td.sm^2))
vlength.m vlength %*% t(vlength)
vlength <- as(vlength, "sparseVector")
vlength.m <- tcrossprod(vlength)
vlength.m@x <- 1/vlength.m@x
csim <- crossprod(td.sm) * vlength.m 


