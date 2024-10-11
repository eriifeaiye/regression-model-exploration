# Project Polynomial Regression

# Importing the dataset
dataset = read.csv('Bike_Sharing.csv')

# Categorical variables as factor
dataset$season = as.factor(dataset$season)
dataset$workingday = as.factor(dataset$workingday)
dataset$weathersit = as.factor(dataset$weathersit)


# Fitting Polynomial Regression to the whole dataset
# In R, if we specify column "Level2" and it doesn't exist, a new column is created.
dataset$temp2 = dataset$temp^2
dataset$hum2 = dataset$hum^2
# dataset$windspeed2 = dataset$windspeed^2
#attempt to add degree 3 but p values exceeded 5%
# dataset$temp3 = dataset$temp^3
# dataset$hum3 = dataset$hum^3


# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$cnt, SplitRatio = 3/4)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


regressor = lm(formula = cnt ~ .,
               data = training_set)
summary(regressor)

# Visualizing the Training Polynomial Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$temp, y = training_set$cnt),
             colour = 'red') +
  ggtitle('Polynomial Regression (Training Set)') +
  xlab('Temperature') +
  ylab('Bike count')


# Visualizing the Polynomial Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$temp, y = test_set$cnt),
             colour = 'red') +
  ggtitle('Polynomial Regression (Test Set)') +
  xlab('Temperature (Test Set)') +
  ylab('Bike count')

