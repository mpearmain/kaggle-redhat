# Geo rank

library(data.table)
x1 <- fread("./Submission.csv", data.table = F)
x2 <- fread("./model_sub.csv", data.table = F)

x1[,2] <- rank(x1[,2])/nrow(x1)
x2[,2] <- rank(x2[,2])/nrow(x2)
# check that the index ordering matches :-)
x1 <- x1[with(x1, order(activity_id)), ]
x2 <- x2[with(x2, order(activity_id)), ]

xfor <- x1

xfor[,2] <- exp((0.9* log(x1[,2]) + (0.1 * log(x2[,2]))))
# save, submit, (hopefully) smile
write.csv(xfor, "./xmix_hacky.csv", row.names = F)
