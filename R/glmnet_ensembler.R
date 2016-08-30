# Attempt at glmnet ensemble for multiple predictions made from classify_bayes.py
library(data.table)
library(glmnet)
library(Metrics)
library(caret)
library(e1071)
library(doParallel)
library(readr)
require(stringr)

# Load the train dataset -> forgot to add the outcome to output :-S
# Lets merge.
train <- fread('./input/act_train.csv', select = c('activity_id', 'outcome'), data.table = FALSE)
id_var = 'activity_id'
target = 'outcome'


auc <- function (actual, predicted) {
  r <- as.numeric(rank(predicted))
  
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <-
    (sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos *  n_neg)
  auc
  
}

msg <- function(mmm, ...)
{
  cat(sprintf(paste0("[%s] ", mmm), Sys.time(), ...))
  cat("\n")
}



## data ####
# list the groups
xlist_val <- dir("./metafeatures/", pattern =  "prval", full.names = T)
xlist_full <- dir("./metafeatures/", pattern = "prfull", full.names = T)

# aggregate validation set
ii <- 1
mod_type <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
mod_data <- str_split(xlist_val[[ii]], "_")[[1]][[4]]
mod_class <- paste(mod_type, mod_data, sep = '_')

print(paste("Reading data for", mod_class))
xvalid <- read_csv(xlist_val[[ii]])
xvalid$outcome <- NULL
xcols <- colnames(xvalid)[-which(colnames(xvalid) %in% c(id_var, target))]
xcols <- paste(mod_class, xcols , ii, sep = "_")
colnames(xvalid)[-which(colnames(xvalid) %in% c(id_var, target))] <- xcols
xvalid <- merge(xvalid, train, by = c(id_var))


for (ii in 2:length(xlist_val))
{
  mod_type <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
  mod_data <- str_split(xlist_val[[ii]], "_")[[1]][[4]]
  mod_class <- paste(mod_type, mod_data, sep = '_')
  print(paste("Reading data for", mod_class))
  xval <- read_csv(xlist_val[[ii]])
  xvalid$outcome <- NULL
  xcols <- colnames(xval)[-which(colnames(xval) %in% c(id_var, target))]
  xcols <- paste0(mod_class, xcols , ii, sep = "")
  colnames(xval)[-which(colnames(xval) %in% c(id_var, target))] <- xcols
  xvalid <- merge(xvalid, xval, by = c(id_var))
  msg(ii)
}

ii <- 1
mod_type <- str_split(xlist_full[[ii]], "_")[[1]][[2]]
mod_data <- str_split(xlist_full[[ii]], "_")[[1]][[4]]
mod_class <- paste(mod_type, mod_data, sep = '_')

print(paste("Reading data for", mod_class))
xfull <- read_csv(xlist_full[[ii]])
xfull$outcome <- NULL
xcols <- colnames(xfull)[-which(colnames(xfull) %in% c(id_var))]
xcols <- paste(mod_class, xcols , ii, sep = "_")
colnames(xfull)[-which(colnames(xfull) %in% c(id_var))] <- xcols

for (ii in 2:length(xlist_full))
{
  mod_type <- str_split(xlist_full[[ii]], "_")[[1]][[2]]
  mod_data <- str_split(xlist_full[[ii]], "_")[[1]][[4]]
  mod_class <- paste(mod_type, mod_data, sep = '_')
  print(paste("Reading data for", mod_class))
  xval <- read_csv(xlist_full[[ii]])
  xfull$outcome <- NULL
  xcols <- colnames(xval)[-which(colnames(xval) %in% c(id_var))]
  xcols <- paste0(mod_class, xcols , ii, sep = "")
  colnames(xval)[-which(colnames(xval) %in% c(id_var))] <- xcols
  xfull <- merge(xfull, xval, by = c(id_var))
  msg(ii)
}

rm(xval)

y = train$outcome
id_train <- xvalid$activity_id
xvalid$activity_id <- NULL
xvalid$outcome <- NULL

test_id <- xfull$activity_id

######################################################################
######################################################################
## Model. Use Caret to find best alpha and lambda

eGrid <- expand.grid(.alpha = (0:10) * 0.1, 
                     .lambda = (0:10) * 0.1)
Control <- trainControl(method = "repeatedcv", 
                        allowParallel = T,
                        number = 2,
                        repeats = 2,
                        verboseIter =TRUE,
                        classProbs = TRUE,
                        summaryFunction=twoClassSummary)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

netFit <- train(x = as.matrix(xvalid), 
                y = as.factor(make.names(y)),
                method = "glmnet",
                tuneGrid = eGrid,
                trControl = Control,
                family = "binomial",
                metric = "ROC")
stopCluster(cl)
# Check the local AUC
auc(y, predict(netFit, xvalid, type = "prob")[2])

submission <- predict(netFit, xfull, type = "prob")[2]
hist(submission$X1)
submission <- as.data.table(list("activity_id" = test_id, 
                                 'outcome' = submission$X1))

write.csv(submission, 'submissions/glmnet_ensembler.csv', row.names = F)





