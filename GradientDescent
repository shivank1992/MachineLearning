##  Install packages
list.of.packages <- c("data.table","GGally","corrplot")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
# 

library(data.table)
library(GGally)
library(caret)
library(corrplot)


#####NOTE : Save this file in a folder where only Traning Features_Variant_1.csv + all 10 Test Cases
###############################################################################
##Merge Training Variant-1 and all 10 Testing Dataset Files
# (1) Make sure where your files are located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
csv_files <- list.files (path       = "./fb_dataset", 
                         pattern    = "*.csv", 
                         full.names = T)

library (data.table)
library(dplyr)
facebook.df <- as_tibble(rbindlist (lapply (csv_files, fread)))
####Dimensions of the dataset should be 41949 Rows and 54 Columns
#dim(facebook.df)

###Explore the dataset
# head(facebook.df)
# str(facebook.df)
# summary(facebook.df)
# dim(facebook.df)
#ggpairs(data=facebook.df, columns=1:54, title="FB Page Data")

#Data pre-processing
##Check for NA values
sapply(facebook.df,function(x){sum(is.na(x))})
##Rename column names
names(facebook.df)[1:4] <- c('likes',
                             'checkins',
                             'talking',
                             'category')

names(facebook.df)[30:39] <- c('CC1',
                               'CC2',
                               'CC3',
                               'CC4',
                               'CC5',
                               'base_time',
                               'post_length',
                               'post_share_count',
                               'post_promotion_status',
                               'hours_taken')

names(facebook.df)[40:46] <- c('p_sun',
                               'p_mon',
                               'p_tue',
                               'p_wed',
                               'p_thu',
                               'p_fri',
                               'p_sat')

names(facebook.df)[47:53] <- c('s_sun',
                               's_mon',
                               's_tue',
                               's_wed',
                               's_thu',
                               's_fri',
                               's_sat')

names(facebook.df)[54] <- c('target')
facebook.df2 <- facebook.df

#facebook.df2 <- facebook.df[,-(5:29)]
#facebook.df2 <- facebook.df2[,-"CC5"]
summary(facebook.df2)
str(facebook.df2)
sapply(facebook.df2,function(x){sum(is.na(x))})
#boxplot(facebook.df2,ylim=c(0.01,200))

###Feature Scaling
target <- as.numeric(as.character(facebook.df2$target))
scaledFB.df<-as.data.frame(scale(facebook.df2[,-54]))
scaledFB.df<-as.data.frame(scale(facebook.df2[,-54], 
                                 center = TRUE, 
                                 scale = apply(facebook.df2[,-54], 2, sd, na.rm = TRUE)))

summary(scaledFB.df)
scaledFB.df <- cbind(target, scaledFB.df)
str(scaledFB.df)
summary(scaledFB.df)
sum(is.na(scaledFB.df))

# ##Remove near zero variance variable
# set.seed(13)
# nzrv <- nearZeroVar(scaledFB.df[,-1], saveMetrics = T)
# discard <- rownames(nzrv[nzrv$zeroVar,])
# keep <- setdiff(names(scaledFB.df), discard)
# cleanFB.df <- scaledFB.df[,keep]
# cat((discard), "is a zero variance variable.")

##Remove near zero variance variable
set.seed(13)
nzrv <- nearZeroVar(scaledFB.df[,-1], saveMetrics = T)
discard <- rownames(nzrv[nzrv$zeroVar,])
keep <- setdiff(names(scaledFB.df), discard)
cleanFB.df <- scaledFB.df[,keep]
cat((discard), "is a zero variance variable.")

##Partition data randomly into train and test set using a 70/30 split
set.seed(13)
train.index <- createDataPartition(cleanFB.df$target, p = 0.7, list = FALSE)
train.df <- cleanFB.df[train.index, ]
test.df <- cleanFB.df[-train.index, ]

correlationMatrix <- cor(train.df[,2:53])
#visualize the matrix, clustering features by correlation index.
col<- colorRampPalette(c("red", "white", "blue"))(20)
corrplot(correlationMatrix, order = "hclust",type="lower",tl.col="black", tl.srt=45,tl.cex = 0.7)
# find attributes that are highly corrected (ideally >0.7)

highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7)
print(highlyCorrelated)
names(train.df[highlyCorrelated])

datMyFiltered.scale <- train.df[,-highlyCorrelated]
corMatMy <- cor(datMyFiltered.scale)
corrplot(corMatMy, order = "hclust",type="lower",tl.col="black", tl.srt=45,tl.cex = 0.7)




###set up the cost function for least square linear regression:
computeCost <- function(matrixA, matrixB, beta_matrix){
  m <- length(matrixB)
  predictions<-matrixA%*%beta_matrix
  #print (predictions)
  squaredErrors<-(predictions-matrixB)^2
  J=(1/(2*m))*sum(squaredErrors)
  #print (J)
  return (J)
}

###set up the gradient descent
gradientDescent_nonconv <- function(train, test, target_var, alpha, iterations){
  start.time <- Sys.time()
  print("Initialize parameters...")
  beta_matrix<-matrix(0,NCOL(train),1)
  keep1 <- setdiff(names(train), target_var)
  train_x <- as.matrix.data.frame(train[,keep1])
  train_x1 <- cbind(a=1, train_x)
  train_y <- as.matrix(train[,target_var])
  m1 <- length(train_y)
  beta_history <- matrix(rep(0, len=iterations*length(train_y)), nrow = length(train_y), ncol = iterations)
  J_history <- matrix(rep(0, len=iterations), nrow = iterations)
  beta_history <- list(iterations)
  for(i in 1:iterations){
    hypothesis <- train_x1 %*% beta_matrix  # guess(y-.hat) = b0 + b1.x1 + b2.x2 +.... + bn.xn for all observations(no of rows)
    error <- hypothesis - train_y   #guess(y.hat) - actual(y) for all observations(no of rows)
    updates <- t(train_x1) %*% error   #Summation of [(b0 + b1.x1 + b2.x2 +.... + bn.xn) - actual(y)] * xn to for all fetures
    beta_matrix <- beta_matrix - alpha * (1/m1) * updates  #Old beta - (alpha * (1/m) * Updates * Summation of errors for every feature)
    J_history[i] <- computeCost(train_x1, train_y, beta_matrix)
    beta_history[[i]] <- beta_matrix
  }
  keep2 <- setdiff(names(test), target_var)
  test_x <- as.matrix.data.frame(test[,keep2])
  test_x1 <- cbind(a=1, test_x)
  test_y <- as.matrix(test[,target_var])
  J_test <- computeCost(test_x1 , test_y, beta_matrix)
  t<- Sys.time() - start.time
  print(paste("Algorithm converged in ",i," iterations"))
  print(paste("Final train data cost is",J_history[i]))
  print(paste("Final test data cost is",J_test))
  print(paste("Time taken: ",round(as.numeric(t, units = "secs"),2)," seconds"))
  values<-list("final_beta" = beta_matrix, "train_cost_history" = J_history[1:i],"beta_history" = beta_history,"cost_test" = J_test)
  return(values)
}



########################
#Exp-1 : Experiment with various values of learning rate

###alpha = 0.001
target_var <- "target"
iterations <- 2000
alpha <- 0.001
GD_result_001 <- gradientDescent_nonconv(train.df, test.df, target_var, alpha, iterations)
beta_history <- GD_result_001$beta_history
beta_zero_001 <- sapply(beta_history,'[[',1)
beta_final_001 <- round(GD_result_001[[1]],4)
cost_hist_001 <- GD_result_001[[2]]
test_cost_001 <- GD_result_001[[4]]
#print(beta_final)

#plot beta_zero history
plot(x= beta_zero_001,y=cost_hist_001,main='Cost Vs Beta_Zero, alpha = 0.001', ylab='cost', xlab='Beta_0')
#Plot cost hist
plot(cost_hist_001, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations, alpha = 0.001',
     ylab='cost', xlab='Iterations')

###alpha = 0.003
target_var <- "target"
iterations <- 2000
alpha <- 0.003
GD_result_003 <- gradientDescent_nonconv(train.df, test.df, target_var, alpha, iterations)
beta_history <- GD_result_003$beta_history
beta_zero_003 <- sapply(beta_history,'[[',1)
beta_final_003 <- round(GD_result_003[[1]],4)
cost_hist_003 <- GD_result_003[[2]]
test_cost_003 <- GD_result_003[[4]]
#print(beta_final)
plot(x= beta_zero_003,y=cost_hist,main='Cost Vs Beta_Zero, alpha = 0.003', ylab='cost', xlab='Beta_0')
plot(cost_hist_003, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations, alpha = 0.003',
     ylab='cost', xlab='Iterations')

###alpha = 0.01
target_var <- "target"
iterations <- 2000
alpha <- 0.01
GD_result_01 <- gradientDescent_nonconv(train.df, test.df, target_var, alpha, iterations)
beta_history <- GD_result_01$beta_history
beta_zero_01 <- sapply(beta_history,'[[',1)
beta_final_01 <- round(GD_result_01[[1]],4)
cost_hist_01 <- GD_result_01[[2]]
test_cost_01 <- GD_result_01[[4]]
#print(beta_final)
plot(x= beta_zero_01,y=cost_hist,main='Cost Vs Beta_Zero, alpha = 0.01', ylab='cost', xlab='Beta_0')
plot(cost_hist_01, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations, alpha = 0.01',
     ylab='cost', xlab='Iterations')

###alpha = 0.03
target_var <- "target"
iterations <- 2000
alpha <- 0.03
GD_result_03 <- gradientDescent_nonconv(train.df, test.df, target_var, alpha, iterations)
beta_history <- GD_result_03$beta_history
beta_zero_03 <- sapply(beta_history,'[[',1)
beta_final_03 <- round(GD_result_03[[1]],4)
cost_hist_03 <- GD_result_03[[2]]
test_cost_03 <- GD_result_03[[4]]
#print(beta_final)
plot(x= beta_zero_03,y=cost_hist,main='Cost Vs Beta_Zero, alpha = 0.03', ylab='cost', xlab='Beta_0')
plot(cost_hist_03, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations, alpha = 0.03',
     ylab='cost', xlab='Iterations')

###alpha = 0.1
target_var <- "target"
iterations <- 2000
alpha <- 0.1
GD_result_1 <- gradientDescent_nonconv(train.df, test.df, target_var, alpha, iterations)
beta_history <- GD_result_1$beta_history
beta_zero_1 <- sapply(beta_history,'[[',1)
beta_one_1 <- sapply(beta_history,'[[',2)
beta_final_1 <- round(GD_result_1[[1]],4)
cost_hist_1 <- GD_result_1[[2]]
test_cost_1 <- GD_result_1[[4]]
#print(beta_final)
#Plot beta_zero
plot(x= beta_zero_1,y=cost_hist_1,main='Training Data Cost Vs Beta_Zero, alpha = 0.1', ylab='cost', xlab='Beta_0')
#Plot cost history
plot(cost_hist_1, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations, alpha = 0.1',
     ylab='cost', xlab='Iterations')


###alpha = 0.118
target_var <- "target"
iterations <- 2000
alpha <- 0.118
GD_result_30 <- gradientDescent_nonconv(train.df, test.df, target_var, alpha, iterations)
beta_history <- GD_result_30$beta_history
beta_zero_30 <- sapply(beta_history,'[[',1)
beta_final_30 <- round(GD_result_30[[1]],4)
cost_hist_30 <- GD_result_30[[2]]
test_cost_30 <- GD_result_30[[4]]
#print(beta_final)
plot(x= beta_zero_30,y=cost_hist,main='Cost Vs Beta_Zero, alpha = 0.0.118', ylab='cost', xlab='Beta_0')
plot(cost_hist_30, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations, alpha = 0.0.118',
     ylab='cost', xlab='Iterations')


# define 6 data sets
y1 <- cost_hist_001
y2 <- cost_hist_003
y3 <- cost_hist_01
y4 <- cost_hist_03
y5 <- cost_hist_1
y6 <- cost_hist_30

# plot the first curve by calling plot() function
# First curve is plotted
plot(y1, type='line',lwd=2, main='Training Data Cost Vs Iterations', ylab='cost', xlab='Iterations',
     col="blue",lty=1 )

points(y2, col="red",lwd=2,type='line')
lines(y2, col="red",lty=2)

points(y3, type='line',lwd=2,col="gold")
lines(y3, col="gold", lty=3)

points(y4, type='line',lwd=2,col="green")
lines(y4, col="green", lty=4)

points(y5, type='line',lwd=2,col="black")
lines(y5, col="black", lty=5)

points(y6, type='line',lwd=2,col="purple")
lines(y6, col="purple", lty=6)

# Adding a legend inside box at the location (1350,570) in graph coordinates.
# Note that the order of plots are maintained in the vectors of attributes.
legend(1400,720,title = "Alpha value",
       legend=c("0.001","0.003","0.01","0.03","0.1","0.118"), 
       col=c("blue","red","gold","green","black","purple"),
       lty=c(1,2,3,4,5,6), ncol=1,cex=0.8)


# define minimum cost of the training and testing dataset for different alphas

alpha_cost_train <- c((tail(cost_hist_001, n=1)),(tail(cost_hist_003, n=1)),
                      (tail(cost_hist_01, n=1)),(tail(cost_hist_03, n=1)),
                      (tail(cost_hist_1, n=1)),(tail(cost_hist_30, n=1)))
alpha_cost_test <- c(test_cost_001,test_cost_003,
                     test_cost_01,test_cost_03,
                     test_cost_1,test_cost_30)

plot(alpha_cost_test, type='line',lwd=2, main='Cost vs alpha value', ylab='cost', xlab='alpha value',
     col="blue",lty=1,
     ylim = c(470,700),xaxt = 'n')
axis(side=1,at=c(1,2,3,4,5,6),labels=c("0.001","0.003","0.01","0.03","0.1","0.118"))
points(alpha_cost_test, col="blue",pch = 15,cex=1.2)

points(alpha_cost_train, col="red",pch = 16,cex=1.2)
lines(alpha_cost_train, col="red",lty=1, lwd=2)

# Adding a legend inside box at the location (1350,570) in graph coordinates.
# Note that the order of plots are maintained in the vectors of attributes.
legend(2,700,
       legend=c("Train","Test"), 
       col=c("red","blue"),
       lty=c(1,1), ncol=1,cex=0.8)



########################
#Exp-2 : Experiment with various thresholds for convergence
##set up the gradient descent function, running for iterations with convergence threshold:
gradientDescent <- function(train, test, target_var, alpha, iterations, conv_crit){
  start.time <- Sys.time()
  print("Initialize parameters...")
  beta_matrix<-matrix(0,NCOL(train),1)
  keep1 <- setdiff(names(train), target_var)
  train_x <- as.matrix.data.frame(train[,keep1])
  train_x1 <- cbind(a=1, train_x)
  train_y <- as.matrix(train[,target_var])
  m1 <- length(train_y)
  beta_history <- matrix(rep(0, len=iterations*length(train_y)), nrow = length(train_y), ncol = iterations)
  J_history <- matrix(rep(0, len=iterations), nrow = iterations)
  beta_history <- list(iterations)
  for(i in 1:iterations){
    hypothesis <- train_x1 %*% beta_matrix  # guess(y-.hat) = b0 + b1.x1 + b2.x2 +.... + bn.xn for all observations(no of rows)
    error <- hypothesis - train_y   #guess(y.hat) - actual(y) for all observations(no of rows)
    updates <- t(train_x1) %*% error   #Summation of [(b0 + b1.x1 + b2.x2 +.... + bn.xn) - actual(y)] * xn to for all fetures
    beta_matrix <- beta_matrix - alpha * (1/m1) * updates  #Old beta - (alpha * (1/m) * Updates * Summation of errors for every feature)
    J_history[i] <- computeCost(train_x1, train_y, beta_matrix)
    beta_history[[i]] <- beta_matrix
    if ((i>1) && (J_history[i-1]-J_history[i]) < conv_crit) {
      print (J_history[i-1]-J_history[i])
      break
    }
  }
  keep2 <- setdiff(names(test), target_var)
  test_x <- as.matrix.data.frame(test[,keep2])
  test_x1 <- cbind(a=1, test_x)
  test_y <- as.matrix(test[,target_var])
  J_test <- computeCost(test_x1 , test_y, beta_matrix)
  t<- Sys.time() - start.time
  print(paste("Algorithm converged in ",i," iterations"))
  print(paste("Final train data cost is",J_history[i]))
  print(paste("Final test data cost is",J_test))
  print(paste("Time taken: ",round(as.numeric(t, units = "secs"),2)," seconds"))
  values<-list("final_beta" = beta_matrix, "train_cost_history" = J_history[1:i],"beta_history" = beta_history,"cost_test" = J_test)
  return(values)
}



###convergence_criteria <- 0.1
target_var <- "target"
iterations <- 2000
alpha <- 0.1
convergence_criteria <- 0.1
GD_result_001 <- gradientDescent(train.df, test.df, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_001$beta_history
beta_zero_001 <- sapply(beta_history,'[[',1)
beta_final_001 <- round(GD_result_001[[1]],4)
cost_hist_001 <- GD_result_001[[2]]
test_cost_001 <- GD_result_001[[4]]
#print(beta_final)

#convergence_criteria <- 0.01
target_var <- "target"
iterations <- 4000
alpha <- 0.1
convergence_criteria <- 0.01
GD_result_003 <- gradientDescent(train.df, test.df, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_003$beta_history
beta_zero_003 <- sapply(beta_history,'[[',1)
beta_final_003 <- round(GD_result_003[[1]],4)
cost_hist_003 <- GD_result_003[[2]]
test_cost_003 <- GD_result_003[[4]]
#print(beta_final)

#convergence_criteria <- 0.001
target_var <- "target"
iterations <- 6000
alpha <- 0.1
convergence_criteria <- 0.001
GD_result_01 <- gradientDescent(train.df, test.df, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_01$beta_history
beta_zero_01 <- sapply(beta_history,'[[',1)
beta_final_01 <- round(GD_result_01[[1]],4)
cost_hist_01 <- GD_result_01[[2]]
test_cost_01 <- GD_result_01[[4]]
#print(beta_final)

#convergence_criteria <- 0.0001
target_var <- "target"
iterations <- 8000
alpha <- 0.1
convergence_criteria <- 0.0001
GD_result_03 <- gradientDescent(train.df, test.df, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_03$beta_history
beta_zero_03 <- sapply(beta_history,'[[',1)
beta_final_03 <- round(GD_result_03[[1]],4)
cost_hist_03 <- GD_result_03[[2]]
test_cost_03 <- GD_result_03[[4]]
#print(beta_final)

#convergence_criteria <- 0.00001
target_var <- "target"
iterations <- 10000
alpha <- 0.1
convergence_criteria <- 0.00001
GD_result_1 <- gradientDescent(train.df, test.df, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_1$beta_history
beta_zero_1 <- sapply(beta_history,'[[',1)
beta_final_1 <- round(GD_result_1[[1]],4)
cost_hist_1 <- GD_result_1[[2]]
test_cost_1 <- GD_result_1[[4]]
#print(beta_final)
plot(cost_hist_1, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations',
     ylab='cost', xlab='Iterations')


# define 5 data sets
y1 <- cost_hist_1
y2 <- cost_hist_03
y3 <- cost_hist_01
y4 <- cost_hist_003
y5 <- cost_hist_001

# plot the first curve by calling plot() function
# First curve is plotted
plot(y1, type='line',lwd=2, main='Exp 2 : Training Data Threshold Value', ylab='cost', xlab='Iterations',
     col="blue",lty=1)

points(y2, col="red",lwd=2,type='line')
lines(y2, col="red",lty=1)

points(y3, type='line',lwd=2,col="gold")
lines(y3, col="gold", lty=1)

points(y4, type='line',lwd=2,col="green")
lines(y4, col="green", lty=1)

points(y5, type='line',lwd=2,col="black")
lines(y5, col="black", lty=1)


# Adding a legend inside box at the location (1350,570) in graph coordinates.
# Note that the order of plots are maintained in the vectors of attributes.
legend(6000,600,title = "Threshold value",
       legend=c("0.00001","0.0001","0.001","0.01","0.1"), 
       col=c("blue","red","gold","green","black"),
       lty=c(1,1,1,1,1), ncol=1,cex=0.8)

training_threshold_00001 <- cost_hist_01

##############
threshold_cost_train <- c((tail(cost_hist_1, n=1)),(tail(cost_hist_03, n=1)),
                      (tail(cost_hist_01, n=1)),(tail(cost_hist_003, n=1)),
                      (tail(cost_hist_001, n=1)))
threshold_cost_test <- c(test_cost_1,test_cost_03,
                     test_cost_01,test_cost_003,
                     test_cost_001)

plot(threshold_cost_test, type='line',lwd=2, main='Exp 2 : Cost vs Convergence Threshold', 
     ylab='cost', xlab='Convergence Threshold',
     col="blue",lty=1,
     ylim = c(470,650),xaxt = 'n')
axis(side=1,at=c(1,2,3,4,5),labels=c("0.00001","0.0001","0.001","0.01","0.1"))
points(threshold_cost_test, col="blue",pch = 15,cex=1.2)

points(threshold_cost_train, col="red",pch = 16,cex=1.2)
lines(threshold_cost_train, col="red",lty=1, lwd=2)

# Note that the order of plots are maintained in the vectors of attributes.
legend(3,600,title = "Fixed aplha = 0.1",
       legend=c("Test","Train"), 
       col=c("blue","red"),
       lty=c(1,1), ncol=1,cex=0.8)


##Set up gradient descent for exp2 part 2
gradientDescent.exp2 <- function(train, test, target_var, alpha, iterations, conv_crit){
  start.time <- Sys.time()
  print("Initialize parameters...")
  beta_matrix<-matrix(0,NCOL(train),1)
  keep1 <- setdiff(names(train), target_var)
  train_x <- as.matrix.data.frame(train[,keep1])
  train_x1 <- cbind(a=1, train_x)
  train_y <- as.matrix(train[,target_var])
  m1 <- length(train_y)
  keep2 <- setdiff(names(test), target_var)
  test_x <- as.matrix.data.frame(test[,keep2])
  test_x1 <- cbind(a=1, test_x)
  test_y <- as.matrix(test[,target_var])
  beta_history <- matrix(rep(0, len=iterations*length(train_y)), nrow = length(train_y), ncol = iterations)
  J_history <- matrix(rep(0, len=iterations), nrow = iterations)
  J_test <- matrix(rep(0, len=iterations), nrow = iterations)
  beta_history <- list(iterations)
  for(i in 1:iterations){
    hypothesis <- train_x1 %*% beta_matrix  # guess(y-.hat) = b0 + b1.x1 + b2.x2 +.... + bn.xn for all observations(no of rows)
    error <- hypothesis - train_y   #guess(y.hat) - actual(y) for all observations(no of rows)
    updates <- t(train_x1) %*% error   #Summation of [(b0 + b1.x1 + b2.x2 +.... + bn.xn) - actual(y)] * xn to for all fetures
    beta_matrix <- beta_matrix - alpha * (1/m1) * updates  #Old beta - (alpha * (1/m) * Updates * Summation of errors for every feature)
    J_history[i] <- computeCost(train_x1, train_y, beta_matrix)
    J_test[i] <- computeCost(test_x1 , test_y, beta_matrix)
    beta_history[[i]] <- beta_matrix
    if ((i>1) && (J_history[i-1]-J_history[i]) < conv_crit) {
      #print (J_history[i-1]-J_history[i])
      break
    }
  }
  t<- Sys.time() - start.time
  print(paste("Algorithm converged in ",i," iterations"))
  print(paste("Final train data cost is",J_history[i]))
  print(paste("Final test data cost is",J_test[i]))
  print(paste("Time taken: ",round(as.numeric(t, units = "secs"),2)," seconds"))
  values<-list("final_beta" = beta_matrix, "train_cost_history" = J_history[1:i],"beta_history" = beta_history,"cost_test" = J_test[1:i])
  return(values)
}



#convergence_criteria <- 0.001
target_var <- "target"
iterations <- 6000
alpha <- 0.1
convergence_criteria <- 0.001
GD_result_exp2 <- gradientDescent.exp2(train.df, test.df, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_exp2$beta_history
beta_zero_exp2 <- sapply(beta_history,'[[',1)
beta_final_exp2 <- round(GD_result_exp2[[1]],4)
cost_hist_exp2 <- GD_result_exp2[[2]]
test_cost_exp2 <- GD_result_exp2[[4]]

##Training vs Testing Data as a function of iterations
# First curve is plotted
plot(cost_hist_exp2, type='line',lwd=2, main='Exp 2 Part2 : Training vs Testing Data', 
     ylab='cost', xlab='Iterations',
     col="blue",lty=1,
     ylim = c(470,800), xlim = c(0,1500))

points(test_cost_exp2,lwd=2,type='line',col = "red")
lines(test_cost_exp2, col="red",lty=2)

legend(500,800,title = "Threshold value = 0.001",
       legend=c("Traning Dataset","Testing Dataset"), 
       col=c("blue","red"),
       lty=c(1,2), ncol=1,cex=0.8)
##############################################################################################3
################################################################################################33
###Final alpha = 0.1, converge_criteria=0.001
#Experiment-3 : Random 5 Dataset
##Initialize data
exp3.train <- train.df[,c(1,47,25,8,9,52)]
exp3.test <- test.df[,c(1,47,25,8,9,52)]
names(train.df[,c(1,47,25,8,9,52)])


###alpha = 0.1
target_var <- "target"
iterations <- 2000
alpha <- 0.1
convergence_criteria <- 0.001
GD_result_1 <- gradientDescent(exp3.train, exp3.test, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_1$beta_history
beta_zero_1 <- sapply(beta_history,'[[',1)
beta_final_1 <- round(GD_result_1[[1]],4)
cost_hist_exp3_train <- GD_result_1[[2]]
test_cost_exp3 <- GD_result_1[[4]]
#print(beta_final)
plot(cost_hist_exp3_train, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations',
     ylab='cost', xlab='Iterations')


####Exp : 32 variables Data
#convergence_criteria <- 0.001
datMyFiltered.train <- train.df[,-highlyCorrelated]
datMyFiltered.test <- test.df[,-highlyCorrelated]
target_var <- "target"
iterations <- 2000
alpha <- 0.1
convergence_criteria <- 0.001
GD_result_01 <- gradientDescent(datMyFiltered.train, datMyFiltered.test, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_01$beta_history
beta_zero_01 <- sapply(beta_history,'[[',1)
beta_final_01 <- round(GD_result_01[[1]],4)
cost_hist_exp3_train32 <- GD_result_01[[2]]
test_cost_exp3_32 <- GD_result_01[[4]]
#print(beta_final)
plot(cost_hist_exp3_train32, type='line', col='blue', lwd=2, 
     main='Cost Vs Iterations',
     ylab='cost', xlab='Iterations')

##Compare diff variable data

plot(cost_hist_exp3_train32, type='line',lwd=2, main='Exp 3 : Random 5 vs 32 Variables', 
     ylab='cost', xlab='Iterations',
     col="blue",lty=1,
     ylim = c(450,750), xlim = c(0,500))

points(cost_hist_exp3_train,lwd=2,type='line',col = "red")
lines(cost_hist_exp3_train, col="red",lty=1)

legend(200,700,
       legend=c("32 Variables","5 Random Variable"), 
       col=c("blue","red"),
       lty=c(1,1), ncol=1,cex=0.8)


########### Barchart

heights<- c(test_cost_exp3_32,test_cost_exp3)
barchart.exp3 <- barplot(heights, names.arg = c("32","5"),
        xlab = "No. of training variables", ylab = "Cost Value", ylim = c(0 ,850),
        main = "Exp 3 : Test Data Cost Value",
        col = c("blue","red"))
text(barchart.exp3, heights+50, labels=round(heights, 1), cex = 1)


################################################################################################
###Exp-4########################
###1.Hours Taken, 2.Post Share Count, 3.Talking, 4.CC2, 5.base_time
exp4.train <- train.df[,c("target","hours_taken","post_share_count","talking","CC2","base_time")]
exp4.test <- test.df[,c("target","hours_taken","post_share_count","talking","CC2","base_time")]

###alpha = 0.1
iterations <- 2000
alpha <- 0.1
convergence_criteria <- 0.001
GD_result_1 <- gradientDescent(exp4.train, exp4.test, target_var, alpha, iterations, convergence_criteria)
beta_history <- GD_result_1$beta_history
beta_zero_1 <- sapply(beta_history,'[[',1)
beta_final_1 <- round(GD_result_1[[1]],4)
cost_hist_exp4_best5 <- GD_result_1[[2]]
test_cost_exp4_best5 <- GD_result_1[[4]]

# First curve is plotted
plot(cost_hist_exp3_train32, type='line',lwd=3, main='Exp 4 : 32 Variables vs 5 Random vs Best 5', 
     ylab='cost', xlab='Iterations',
     col="blue",lty=1,
     ylim = c(450,850), xlim = c(0,500))

points(cost_hist_exp3_train,lwd=3,type='line',col = "red")
lines(cost_hist_exp3_train, col="red",lty=1)

points(cost_hist_exp4_best5,lwd=3,type='line',col = "green")
lines(cost_hist_exp4_best5, col="green",lty=1)

legend(200,850,
       legend=c("32 Variables","5 Random Variable","5 Best Variables"), 
       col=c("blue","red","green"),
       lty=c(1,1,1), ncol=1,cex=0.8)
text(barchart.exp4, heights.exp4+50, labels=round(heights.exp4, 1), cex = 1)


########### Barchart

heights.exp4<- c(test_cost_exp3_32,test_cost_exp4_best5,test_cost_exp3)
barchart.exp4 <- barplot(heights.exp4, names.arg = c("32 Variables","5 Best","5 Random"),
                         xlab = "No. of training variables", ylab = "Cost Value", ylim = c(0 ,850),
                         main = "Exp 4 : Test Data Cost Value",
                         col = c("blue","green","red"))
text(barchart.exp4, heights.exp4+50, labels=round(heights.exp4, 1), cex = 1)




####################################
####################################
####################################
##########-------END-------#########
####################################
####################################
####################################
