# Project: Random Forest Regression

# Importing the dataset
dataset = read.csv('Bike_Sharing.csv')

# Categorical variables as factor
dataset$season = as.factor(dataset$season)
dataset$workingday = as.factor(dataset$workingday)
dataset$weathersit = as.factor(dataset$weathersit)


# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$cnt, SplitRatio = 3/4)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Regression to the whole dataset
library(randomForest)
regressor = randomForest(formula = cnt ~ .,
                         data = training_set,
                         ntree = 850)

levels(test_set$season) <- levels(test_set$season)
levels(test_set$workingday) <- levels(test_set$workingday)
levels(test_set$weathersit) <- levels(test_set$weathersit)

y_pred = data.frame(predict(regressor, newdata = test_set))


# Visualizing the Random Forest Regression results (for higher resolution and smoother curve)
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$temp, y = training_set$cnt),
             colour = 'red') +
  ggtitle('Random Forest Regression (Training Set') +
  xlab('Temperature') +
  ylab('Bike Count')

# Visualizing the Random Forest Regression results (for higher resolution and smoother curve)
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$temp, y = y_pred$predict.regressor..newdata...test_set.),
             colour = 'red') +
  ggtitle('Random Forest Regression (Test set)') +
  xlab('Temperature') +
  ylab('Bike Count')