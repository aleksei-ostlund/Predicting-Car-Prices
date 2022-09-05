#load packages

install.packages("tidyverse")
install.packages("caret")
library(tidyverse)
library(caret)

#import data file
cars_data <- read.table("imports-85.data", sep = ",")

#tibble overview
as_tibble(cars_data)

#name columns according to description file
cars_data_cleaned <- rename(cars_data, 
                    "symboling" = "V1",
                    "normalized_losses" = "V2",
                    "make" = "V3",
                    "fuel_type" = "V4", 
                    "aspiration" = "V5",
                    "num_of_doors" = "V6",
                    "body_style" = "V7",
                    "drive_wheels" = "V8",
                    "engine_location" = "V9",
                    "wheel_base" = "V10",
                    "length" = "V11",
                    "width" = "V12",
                    "height" = "V13",
                    "curb_weight" = "V14",
                    "engine_type" = "V15",
                    "num_of_cylinders" = "V16",
                    "engine_size" = "V17",
                    "fuel_system" = "V18",
                    "bore" = "V19",
                    "stroke" = "V20",
                    "compression_ratio" = "V21",
                    "horsepower" = "V22",
                    "peak_rpm" = "V23",
                    "city_mpg" = "V24",
                    "highway_mpg" = "V25",
                    "price" = "V26"
)

#select quantitative columns
cars_numeric <- cars_data_cleaned %>%
  select(-make, -fuel_type, -aspiration, -body_style, -drive_wheels,
         -engine_location, -engine_type, -fuel_system)

#check unique values

print(unique(cars_numeric$num_of_doors))
print(unique(cars_numeric$num_of_cylinders))

#change written numbers to digits
cars_numeric <- cars_numeric %>%
  mutate(num_of_doors= recode(cars_numeric$num_of_doors, 
                               "two" = 2, 
                               "four" = 4)) %>%
  mutate(num_of_cylinders= recode(cars_numeric$num_of_cylinders, 
                                   "four" = 4,
                                   "six" = 6,
                                   "five" = 5,
                                   "three" = 3,
                                   "twelve" = 12,
                                   "two" = 2,
                                   "eight" = 8
                                   ))

#remove NA rows
cars_numeric_clean <- cars_numeric %>%
  na_if("?") %>%
  na.omit()

#check column types
as_tibble(str(cars_numeric_clean))

#convert characters to numeric
cars_numeric_clean <- cars_numeric_clean %>%
  transform(num_of_doors = as.numeric(num_of_doors),
           normalized_losses = as.numeric(normalized_losses),
           bore = as.numeric(bore),
           stroke = as.numeric(stroke),
           horsepower = as.numeric(horsepower),
           peak_rpm = as.numeric(peak_rpm),
           price = as.numeric(price)
             )

#lattice plots for correlation
featurePlot(cars_numeric_clean[,c('horsepower','engine_size', 'curb_weight', 'city_mpg')], cars_numeric_clean$price)

#partition data with and 80/20 split
train_indices <- createDataPartition(y=cars_numeric_clean[["price"]],
                                     p = 0.8,
                                     list=FALSE)

train_listings <- cars_numeric_clean[train_indices,]
test_listings <- cars_numeric_clean[-train_indices,]

#produce 10 test sets using 10% of the TRAINING data -> estimate model error

ten_fold_control <- trainControl(method = "cv", number = 10)

#set up k nearest neighbors models with diff num of features
#normalize data with preProcess (center = subtract mean, scale = divide by sdv)

cars_knn_model_4f <- train(price ~ horsepower + engine_size + curb_weight + city_mpg,
                        data = train_listings,
                        method = "knn",
                        trControl = ten_fold_control,
                        preProcess = c("center", "scale"))

cars_knn_model_3f <- train(price ~ horsepower + engine_size + curb_weight,
                        data = train_listings,
                        method = "knn",
                        trControl = ten_fold_control,
                        preProcess = c("center", "scale"))

cars_knn_model_2f <- train(price ~ horsepower + engine_size,
                        data = train_listings,
                        method = "knn",
                        trControl = ten_fold_control,
                        preProcess = c("center", "scale"))

#4 factor model RMSE=2366
test_predictions_4f <- predict(cars_knn_model_4f, newdata = test_listings)
print(as_tibble(test_predictions_4f))

rmse_4f <- postResample(pred = test_predictions_4f, obs = test_listings$price) 
print(rmse_4f)

#3 factor model RMSE=2432
test_predictions_3f <- predict(cars_knn_model_3f, newdata = test_listings)
print(as_tibble(test_predictions_3f))

rmse_3f <- postResample(pred = test_predictions_3f, obs = test_listings$price) 
print(rmse_3f)

#2 factor model RMSE=3162
test_predictions_2f <- predict(cars_knn_model_2f, newdata = test_listings)
print(as_tibble(test_predictions_2f))

rmse_2f <- postResample(pred = test_predictions_2f, obs = test_listings$price)
print(rmse_2f)

#4 factor model has lowest RMSE
model <- c(1,2,3)
rmse <- c(rmse_4f[[1]],rmse_3f[[1]],rmse_2f[[1]])
compared_rmse <- tibble(model,rmse) %>%
  arrange(rmse)
  print(compared_rmse)
  
#check for final k value used in this model
  print(cars_knn_model_4f)
  
  

#double check with table mutations method
  
#generate new columns for predictions
test_listings <- test_listings %>%
  mutate(predictions_4f = predict(cars_knn_model_4f, newdata = test_listings),
         predictions_3f = predict(cars_knn_model_3f, newdata = test_listings),
         predictions_2f = predict(cars_knn_model_2f, newdata = test_listings))

#round up predictions
test_listings <- test_listings %>%
  mutate(across(19:21, round, 0))
         
#add squared error to test listings
test_listings <- test_listings %>%
  mutate(sq_error_4f = (price - predictions_4f)^2,
         sq_error_3f = (price - predictions_3f)^2,
         sq_error_2f = (price - predictions_2f)^2
         )

#mean squared error grouped by model
pivoted_test_listings <- test_listings %>%
  pivot_longer(cols = sq_error_4f:sq_error_2f, 
               names_to = "model", values_to = "sq_error")

test_listings_by_model <- pivoted_test_listings %>%
  group_by(model) %>%
  summarise(mse = mean(sq_error)) %>%
  arrange(mse) %>%
  mutate(rmse = sqrt(mse))
print(test_listings_by_model)



#create a test case for the 4 factor model (horsepower, engine_size, curb_weight, city_mpg)
car_test <- test_listings[3,c("horsepower", "engine_size", "curb_weight", "city_mpg")]
car_test[1, ] <- c(200,170,2000,20)
car_test[2,] <- c(400,260,3500,15)
car_test[3,] <- c(100, 120, 1500, 30)

#use model to predict prices
car_test <- car_test %>%
  mutate(price= predict(cars_knn_model_4f, newdata = car_test))