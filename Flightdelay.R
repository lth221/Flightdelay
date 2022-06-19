#install.packages("fastDummies")           # for Dummies
#install.packages("randomForest")          # for random forest
#install.packages("dplyr")                 # for easy coding
#install.packages("tree")                  # for tree classification
#install.packages("e1071")                 # for naive bayse
#install.packages("class")                 # for kNN
#install.packages("ggplot2")               # for plotting

library(fastDummies)
library(dplyr)
library(randomForest)
library(tree)
library(e1071)
library(class)
library(ggplot2)


################################################################################
### Read dataset
airline <- read.csv("JFKLGA.csv")
str(airline)


################################################################################
### Check the data and convert types

# Convert Month to dummies 
airline <- dummy_cols(airline, select_columns = "MONTH")
airline <- subset (airline, select = -c(MONTH))

# Convert Day of Week to dummies
airline <- dummy_cols(airline, select_columns = "DAY_OF_WEEK")
airline <- subset (airline, select = -c(DAY_OF_WEEK))

# Convert Departing Airport to dummies
airline <- dummy_cols(airline, select_columns = "DEPARTING_AIRPORT")
airline <- subset(airline, select = -c(DEPARTING_AIRPORT))

# Convert Departure Time Block into 4 bins and to dummies
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "0001-0559"] <- "Midnight"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "0600-0659"] <- "Morning"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "0700-0759"] <- "Morning"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "0800-0859"] <- "Morning"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "0900-0959"] <- "Morning"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1000-1059"] <- "Morning"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1100-1159"] <- "Morning"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1200-1259"] <- "Afternoon"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1300-1359"] <- "Afternoon"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1400-1459"] <- "Afternoon"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1500-1559"] <- "Afternoon"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1600-1659"] <- "Afternoon"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1700-1759"] <- "Afternoon"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1800-1859"] <- "Evening"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "1900-1959"] <- "Evening"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "2000-2059"] <- "Evening"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "2100-2159"] <- "Evening"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "2200-2259"] <- "Evening"
airline["DEP_TIME_BLK"][airline["DEP_TIME_BLK"] == "2300-2359"] <- "Evening"

airline <- dummy_cols(airline, select_columns = "DEP_TIME_BLK")
airline <- subset(airline, select = -c(DEP_TIME_BLK))

# Check correlation
cor(airline$SNOW, airline$SNWD)
# High correlation with snow and ground snow - delete SNWD
airline <- subset(airline, select = -c(SNWD))

# Convert weather into binary variables
airline$PRCP <- ifelse(airline$PRCP != 0, 1, 0)
airline$SNOW <- ifelse(airline$SNOW != 0, 1, 0)
airline$AWND <- ifelse(airline$AWND > 10, 1, 0) #10mph may start causing problem

# Have less impact variables -> delete
airline <- subset(airline, select = -c(DISTANCE_GROUP))
airline <- subset(airline, select = -c(SEGMENT_NUMBER))
airline <- subset(airline, select = -c(CONCURRENT_FLIGHTS))
airline <- subset(airline, select = -c(NUMBER_OF_SEATS))
airline <- subset(airline, select = -c(PLANE_AGE))
airline <- subset(airline, select = -c(TMAX))

# Convert target variable into factor
airline$DEP_DEL15 = as.factor(airline$DEP_DEL15)

# Change column name
colnames(airline)[24] <- "JFK"
colnames(airline)[25] <- "LaGuardia"


################################################################################
### Set seed!
set.seed(123)


################################################################################
### Check the ratio of delayed/not delayed

table(airline$DEP_DEL15)
prop.table(table(airline$DEP_DEL15))

# not delayed : 77.56%
# delayed     : 22.43%
# unbalanced!


################################################################################
### Split training and test set

# Divide the dataset into 50, 50% 
delayed <- subset(airline, airline$DEP_DEL15 == 1)
not_delayed <- subset(airline, airline$DEP_DEL15 == 0)

train.delayed <- sample(1:nrow(delayed),8000)
train.not_delayed <- sample(1:nrow(not_delayed),8000)
dat.train <- rbind(delayed[train.delayed,],not_delayed[train.not_delayed,])
table(dat.train$DEP_DEL15)

# Create a test set out of the remaining deposits 
new_not_delayed <- not_delayed[-train.not_delayed,]
test_not_delayed <- new_not_delayed[sample(1:nrow(new_not_delayed),2543),]
dat.test <- rbind(delayed[-train.delayed,],test_not_delayed)
table(dat.test$DEP_DEL15)

# Check the ratio of training and test set
5086/21086
# training : 75.88%
# test     : 24.12% 
rm(delayed, not_delayed, test_not_delayed, new_not_delayed, train.not_delayed,
   train.delayed)
prop.table(table(dat.test$DEP_DEL15))


################################################################################
### Run Logistic Regression

airline.lr <- subset(airline, select = -c(DEP_TIME_BLK_Evening))
airline.lr <- subset(airline.lr, select = -c(DAY_OF_WEEK_3)) # lowest
airline.lr <- subset(airline.lr, select = -c(MONTH_10))
airline.lr <- subset(airline.lr, select = -c(23))

delayedlr <- subset(airline.lr, airline.lr$DEP_DEL15 == 1)
not_delayedlr <- subset(airline.lr, airline.lr$DEP_DEL15 == 0)

train.delayedlr <- sample(1:nrow(delayedlr),8000)
train.not_delayedlr <- sample(1:nrow(not_delayedlr),8000)
dat.train.lr <- rbind(delayedlr[train.delayedlr,],
                      not_delayedlr[train.not_delayedlr,])

new_not_delayedlr <- not_delayedlr[-train.not_delayedlr,]
test_not_delayedlr <- new_not_delayedlr[sample(1:nrow(new_not_delayedlr),2543),]
dat.test.lr <- rbind(delayedlr[-train.delayedlr,],test_not_delayedlr)

rm(delayedlr, not_delayedlr, test_not_delayedlr, new_not_delayedlr,
   train.not_delayedlr, train.delayedlr)
prop.table(table(dat.test$DEP_DEL15))
str(airline.lr)

lr <- glm(DEP_DEL15 ~ ., data = dat.train.lr, family = "binomial")
summary(lr)

# Create confusion matrix and compute the error on the test data
yhat.dat.test <- predict(lr, dat.test.lr, type = "response")
yhat.dat.test.cl <- ifelse(yhat.dat.test > 0.5, 1, 0)
tab.dat.test <- table(dat.test.lr$DEP_DEL15, yhat.dat.test.cl, 
                      dnn = c("Actual","Predicted"))
tab.dat.test

1586/(1586+957)*100 # stratified accuracy(not delayed): 62.36%
1659/(1659+884)*100 # stratified accuracy(delayed)    : 65.23%

dat.test.err <- mean(dat.test.lr$DEP_DEL15 != yhat.dat.test.cl)
dat.test.err # 36.20%

odds <- data.frame((exp(coef(lr))-1)*100)
rownames <- rownames(odds)
odds <- cbind(rownames, odds)
odds <- odds[order(odds$X.exp.coef.lr.....1....100, decreasing = T), ]
odds

# MONTH
odds_month <- odds %>% slice(-c(4,9:11,15:19,21,22,23,24,25))
odds_month
barplot(odds_month$X.exp.coef.lr.....1....100,
        main= "Month (baseline : Oct)",
        names.arg=c("Aug","Jul","Jan","Jun","Dec","Apr",
                    "Feb","Sep","Mar","May","Nov"))

# DAY
odds_day <- odds %>% slice(-c(1:9,12:14,16,20:25))
odds_day
barplot(odds_day$X.exp.coef.lr.....1....100,
        main= "Day (baseline : Wednesday)",
        names.arg=c("Fri","Thu","Sun","Mon","Sat","Tue"))

# Time Block
odds_timeblock <- odds %>% slice(-c(1:21,23))
odds_timeblock
barplot(odds_timeblock$X.exp.coef.lr.....1....100,
        main= "Time Block (baseline : Evening)",
        names.arg=c("Afternoon","Morning","Midnight"))

# Others
odds_others <- odds %>% slice(-c(1:3,5:8,10:15,17:22,24,25))
odds_others
barplot(odds_others$X.exp.coef.lr.....1....100,
        main = "Weather & Airport",
        names.arg=c("Snow","Rain","Wind","JFK"))


################################################################################
### Run Random Forest

rf = randomForest(DEP_DEL15~., data = dat.train, importance=TRUE)
print(rf)
# Increasing the tree size doesn't give substantial change

# Now make predictions on the test data and compute error
yhat.rf <- predict(rf, dat.test)
tab.rf <- table(dat.test$DEP_DEL15, yhat.rf)
tab.rf

1712/(1712+831)*100  # stratified accuracy(not delayed): 67.32%
1635/(1635+908)*100  # stratified accuracy(delayed)    : 64.29%

err.rf <- mean(dat.test$DEP_DEL15 != yhat.rf)
err.rf # 34.19%

# Important variables
importance(rf)
varImpPlot(rf, main = "Variable Importance Plot")


################################################################################
### Run classification tree

tree <- tree(DEP_DEL15~., data = dat.train)
summary(tree)
plot(tree)
text(tree, pretty = 0)

tree

tree.pred.tst <- predict(tree, dat.test, type = "class")
table(dat.test$DEP_DEL15, tree.pred.tst,
      dnn = c("Actual", "Predicted"))
mean(dat.test$DEP_DEL15 != tree.pred.tst) # 40.58%

1045/(1045+1498)*100  # stratified accuracy(not delayed): 41.09%
1977/(1977+566)*100   # stratified accuracy(delayed)    : 77.74%


################################################################################
### Run Naive Bayes model

nb.fit <- naiveBayes(DEP_DEL15 ~ ., data = dat.train)
nb.fit

# Make predictions on the test data and build confusion matrix
nb.class <- predict(nb.fit, newdata = dat.test)
tab.nb <- table(dat.test$DEP_DEL15, nb.class,
                dnn = c("Actual", "Predicted"))
tab.nb
1494/(1494+1049) # stratified accuracy(not delayed): 58.75%
1653/(1653+890)  # stratified accuracy(delayed)    : 65.00%
err.nb <- mean(dat.test$DEP_DEL15 != nb.class)
err.nb # 38.12%


################################################################################
### Run KNN

dat.test$DEP_DEL15 = as.numeric(dat.test$DEP_DEL15)-1
dat.train$DEP_DEL15 = as.numeric(dat.train$DEP_DEL15)-1
str(dat.test)

dat.train.x <- dat.train[,2:29]
dat.train.y <- dat.train[,1]
dat.test.x <- dat.test[,2:29]
dat.test.y <- dat.test[,1]

# Find the best k
knn.err <- 1:20
xrange <- 1:20
for (j in 1:39) {
  if (j %% 2 != 0) {
    xrange[(j+1)/2] <- j
    out <- knn(dat.train.x, dat.test.x, 
               dat.train.y, j)
    knn.err[(j+1)/2] <- mean(out != dat.test.y)
  }
}

plot(xrange, knn.err, xlab = "Value of K (K odd)",
     ylab = "Error from KNN")
# Best k is 3

knn_best <- knn(dat.train.x, dat.test.x, dat.train.y, k=3)
tab.knn_best <- table(dat.test.y, knn_best,
                      dnn = c("Actual", "Predicted"))
tab.knn_best

1642/(1642+901)*100  # stratified accuracy(not delayed): 64.57%
1663/(1663+880)*100  # stratified accuracy(delayed)    : 65.40%

knn_best.err <- mean(dat.test.y != knn_best)
knn_best.err # 35.02%


################################################################################
### Using present data for prediction

# For logistic
new <- data.frame(
  "PRCP" = 0,
  "SNOW" = 0,
  "AWND" = 0,
  "MONTH_1" = 0,
  "MONTH_2" = 0,
  "MONTH_3" = 1,
  "MONTH_4" = 0,
  "MONTH_5" = 0,
  "MONTH_6" = 0,
  "MONTH_7" = 0,
  "MONTH_8" = 0,
  "MONTH_9" = 0,
  "MONTH_11" = 0,
  "MONTH_12" = 0,
  "DAY_OF_WEEK_1" = 0,
  "DAY_OF_WEEK_2" = 0,
  "DAY_OF_WEEK_4" = 1,
  "DAY_OF_WEEK_5" = 0,
  "DAY_OF_WEEK_6" = 0,
  "DAY_OF_WEEK_7" = 0,
  "DEP_TIME_BLK_Afternoon" = 0,
  "DEP_TIME_BLK_Midnight" = 0,
  "DEP_TIME_BLK_Morning" = 1,
  "JFK" = 1
)

# For random forest and kNN
new1 <- data.frame(
  "PRCP" = 1,
  "SNOW" = 0,
  "AWND" = 0,
  "MONTH_1" = 0,
  "MONTH_2" = 0,
  "MONTH_3" = 1,
  "MONTH_4" = 0,
  "MONTH_5" = 0,
  "MONTH_6" = 0,
  "MONTH_7" = 0,
  "MONTH_8" = 0,
  "MONTH_9" = 0,
  "MONTH_10" = 0,
  "MONTH_11" = 0,
  "MONTH_12" = 0,
  "DAY_OF_WEEK_1" = 0,
  "DAY_OF_WEEK_2" = 0,
  "DAY_OF_WEEK_3" = 0,
  "DAY_OF_WEEK_4" = 1,
  "DAY_OF_WEEK_5" = 0,
  "DAY_OF_WEEK_6" = 0,
  "DAY_OF_WEEK_7" = 0,
  "DEP_TIME_BLK_Evening" = 0,
  "DEP_TIME_BLK_Afternoon" = 0,
  "DEP_TIME_BLK_Midnight" = 0,
  "DEP_TIME_BLK_Morning" = 1,
  "JFK" = 0,
  "LaGuardia" = 1
)

# Prediction for logistic
prediction <- predict(lr, new, type = "response")
prediction.cl <- ifelse(prediction > 0.5, "Delay", "Not Delay")
prediction.cl

# Prediction for random forest
prediction <- predict(rf, new1, type = "response")
prediction.cl <- ifelse(prediction == 1 , "Delay", "Not Delay")
prediction.cl

# Prediction for kNN
prediction <- predict(knn_best, new1, type = "response")
prediction.cl <- ifelse(prediction == 1 , "Delay", "Not Delay")
prediction.cl

