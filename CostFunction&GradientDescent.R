###Set up the cost function for least square linear regression:
computeCost <- function(matrixA, matrixB, beta_matrix){
  m <- length(matrixB)
  predictions<-matrixA%*%beta_matrix
  squaredErrors<-(predictions-matrixB)^2
  J=(1/(2*m))*sum(squaredErrors)
  return (J)
}

###Create multivariable gradient descent function
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


##Check gradient descent algorithm
#target_var <- "target"
#iterations <- 2000
#alpha <- 0.1
#convergence_criteria <- 0.001
#gradientDescent(train.df, test.df, target_var, alpha, iterations, convergence_criteria)
