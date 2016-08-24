library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)
require(caret)
require(readr)
require(stringr)

dataset_version <- "base"
seed_value <- 44556877
model_type <- "XGBHash"
todate <- str_replace_all(Sys.Date(), "-","")
source_folder <- "input"
target_folder <- "metafeatures"
target_params <- 'meta_parameters'

msg <-
  function(mmm,...) {
    cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
  }


auc <- function (actual, predicted) {
  r <- as.numeric(rank(predicted))
  
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <-
    (sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos *  n_neg)
  auc
  
}

# Pre process base data.

train = fread('./input/act_train.csv') %>% as.data.frame()
test = fread('./input/act_test.csv') %>% as.data.frame()


#people data frame
people = fread('./input/people.csv') %>% as.data.frame()
people$char_1 <- NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))] = paste0('people_',names(people)[2:length(names(people))])

p_logi <- names(people)[which(sapply(people, is.logical))]
for (col in p_logi)
  set(people, j = col, value = as.numeric(people[[col]]))

#reducing group_1 dimension
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1) ==
                                                               1))] = 'group unique'


#reducing char_10 dimension
unique.char_10 =
  rbind(select(train,people_id,char_10),
        select(train,people_id,char_10)) %>% group_by(char_10) %>%
  summarize(n = n_distinct(people_id)) %>%
  filter(n == 1) %>%
  select(char_10) %>%
  as.matrix() %>%
  as.vector()

train$char_10[train$char_10 %in% unique.char_10] = 'type unique'
test$char_10[test$char_10 %in% unique.char_10] = 'type unique'

d1 <- merge(train, people, by = "people_id", all.x = T)
d2 <- merge(test, people, by = "people_id", all.x = T)
Y <- d1$outcome
d1$outcome <- NULL

row.train = nrow(train)
gc()

D = rbind(d1,d2)
D$i = 1:dim(D)[1]


test_activity_id = test$activity_id
train_activity_id = train$activity_id
rm(train,test,d1,d2);gc()


char.cols = c(
  'activity_category','people_group_1',
  'char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10',
  'people_char_2','people_char_3','people_char_4','people_char_5','people_char_6','people_char_7','people_char_8','people_char_9'
)
for (f in char.cols) {
  if (class(D[[f]]) == "character") {
    levels <- unique(c(D[[f]]))
    D[[f]] <- as.numeric(factor(D[[f]], levels = levels))
  }
}


D.sparse =
  cBind(
    sparseMatrix(D$i,D$activity_category),
    sparseMatrix(D$i,D$people_group_1),
    sparseMatrix(D$i,D$char_1),
    sparseMatrix(D$i,D$char_2),
    sparseMatrix(D$i,D$char_3),
    sparseMatrix(D$i,D$char_4),
    sparseMatrix(D$i,D$char_5),
    sparseMatrix(D$i,D$char_6),
    sparseMatrix(D$i,D$char_7),
    sparseMatrix(D$i,D$char_8),
    sparseMatrix(D$i,D$char_9),
    sparseMatrix(D$i,D$people_char_2),
    sparseMatrix(D$i,D$people_char_3),
    sparseMatrix(D$i,D$people_char_4),
    sparseMatrix(D$i,D$people_char_5),
    sparseMatrix(D$i,D$people_char_6),
    sparseMatrix(D$i,D$people_char_7),
    sparseMatrix(D$i,D$people_char_8),
    sparseMatrix(D$i,D$people_char_9)
  )

D.sparse =
  cBind(
    D.sparse,
    D$people_char_10,
    D$people_char_11,
    D$people_char_12,
    D$people_char_13,
    D$people_char_14,
    D$people_char_15,
    D$people_char_16,
    D$people_char_17,
    D$people_char_18,
    D$people_char_19,
    D$people_char_20,
    D$people_char_21,
    D$people_char_22,
    D$people_char_23,
    D$people_char_24,
    D$people_char_25,
    D$people_char_26,
    D$people_char_27,
    D$people_char_28,
    D$people_char_29,
    D$people_char_30,
    D$people_char_31,
    D$people_char_32,
    D$people_char_33,
    D$people_char_34,
    D$people_char_35,
    D$people_char_36,
    D$people_char_37,
    D$people_char_38,
    D$binay_sum
  )

train.sparse = D.sparse[1:row.train,]
test.sparse = D.sparse[(row.train + 1):nrow(D.sparse),]


## Time to build a stacker.

# division into folds: 5-fold
xfolds <- fread("./input/5-fold.csv", data.table=F)
xfolds$fold_index <- xfolds$fold5
xfolds <- xfolds[,c("activity_id", "fold_index")]
nfolds <- length(unique(xfolds$fold_index))

## fit models ####

# storage structures
mtrain <- array(0, c(nrow(train.sparse), 1))
mtest <- array(0, c(nrow(test.sparse), 1))

# loop over folds
for (jj in 0:(nfolds-1))
{
  print(paste("running fold", jj))
  isTrain <- which(xfolds$fold_index != jj)
  isValid <- which(xfolds$fold_index == jj)
  x0 <- train.sparse[isTrain,] 
  x1 <- train.sparse[isValid,]
  y0 <- Y[isTrain] 
  y1 <- Y[isValid]
  
  # Hash train to sparse dmatrix X_train
  dtrain  <- xgb.DMatrix(x0, label = y0)
  dtest  <- xgb.DMatrix(x1)
  
  param <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    booster = "gblinear",
    eta = 0.05,
    subsample = 0.72,
    colsample_bytree = 0.72,
    min_child_weight = 0,
    max_depth = 11
  )
  
  
  set.seed(120)
  m2 <- xgb.train(data = dtrain,param, nrounds = 350, watchlist = list(train = dtrain),print_every_n = 10)
  
  # Predict
  pred_valid <- predict(m2, dtest)
  
  print(auc((y1 == 1) + 0, pred_valid))
  mtrain[isValid,1] <- pred_valid
}

# full version
# Hash train to sparse dmatrix X_train
dtrain  <- xgb.DMatrix(train.sparse, label = Y)
dtest  <- xgb.DMatrix(test.sparse)
xgb.model <- xgb.train(data = dtrain, param, nrounds = 350,print_every_n = 10)

pred_full <- predict(xgb.model, dtest)
mtest[,1] <- pred_full
msg(1)

## store complete versions ####
mtrain <- data.frame(mtrain)
mtest <- data.frame(mtest)
colnames(mtrain) <-  colnames(mtest) <- paste(model_type, 1:ncol(mtrain), sep = "")
mtrain$activity_id <- train_activity_id
mtest$activity_id <- test_activity_id 
mtrain$outcome <- Y


write_csv(mtrain, path = paste("./",target_folder, "/prval_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = ""  ))
write_csv(mtest, path = paste("./",target_folder,"/prfull_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = ""))


