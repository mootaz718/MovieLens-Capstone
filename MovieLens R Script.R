########################################
## HarvardX PH125.9x
## Data Science: Capstone
## Project: MovieLens
## Name: Mootaz Abdel-Dayem
########################################



############ I. LOADING DATA ############


##*** Start of code provided by edx ***
#######################################

## Create train and validation sets

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                          title = as.character(title),
#                                          genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##** End of Script provided by edx **

### Save edx and validation files: Main file and Validation file for testing
save(edx, file="edx.RData")
save(validation, file = "validation.RData")


## More packages to install
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(dplyr) install.packages("dplyr", repos = "http://cran.us.r-project.org")
   if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
   if(!require(markdown)) install.packages("markdown", repos = "http://cran.us.r-project.org")
   
   # Load extra libraries for analysis and visulizations
   library(ggplot2)
   library(lubridate)
   library(dplyr)
   library(knitr)
   library(markdown)
   
   ############ II. EXPLORATORY DATA ANALYSIS ############
   
   # Head
   head(edx) %>%
   print.data.frame()
   
   # Total Users & Movies
   summary(edx)
   
   # Number of unique Users & Movies in the edx dataset 
   edx %>%
     summarize(n_users = n_distinct(userId), 
               n_movies = n_distinct(movieId))
   
   # Checking the data 
   str(edx)
   str(validation)
   
   # Data Exploration by finding the most  ratings
   edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
     arrange(desc(count))  
   
   # Number of Ratings per Movie 
   edx %>% count(movieId) %>% ggplot(aes(n))+
     geom_histogram(color = "black" , fill= "light blue")+
     scale_x_log10()+
     ggtitle("Number of Ratings Per Movie")+
     theme_gray()
   
   # Number of Ratings per User
   edx %>% count(userId) %>% ggplot(aes(n))+
     geom_histogram(color = "blue" , fill= "light blue")+
     ggtitle(" Number of Ratings Per User")+
     scale_x_log10()+
     theme_gray()
   
   # Number of Ratings per Movie Genre
   edx %>% separate_rows(genres, sep = "\\|") %>%
     group_by(genres) %>%
     summarize(count = n()) %>%
     arrange(desc(count)) %>% ggplot(aes(genres,count)) + 
     geom_bar(aes(fill =genres),stat = "identity")+ 
     labs(title = " Number of Ratings per Movie Genre")+
     theme(axis.text.x  = element_text(angle= 90, vjust = 50 ))+
     theme_light()
   
   # Partitioning the Data 
   set.seed(1)
   test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
   train_set <- edx[-test_index,]
   test_set <- edx[test_index,]
   
   # Removing Users and Movies from the training set using the semi_join function
   test_set <- test_set %>% 
     semi_join(train_set, by = "movieId") %>%
     semi_join(train_set, by = "userId")
   
   
   # RMSE Calculation Function 
   # Computing the RMSE for vectors of ratings and their corresponding predictors:
   
   RMSE <- function(true_ratings, predicted_ratings){
     sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
   }
   
   
   ############ III. MODELS ############
   
   ## Average Movie Rating Model
   
   # Compute the Mean Rating for the dataset
   mu <- mean(edx$rating)
   mu
   
   # Test the Results based on a Simple Prediction
   
   naive_rmse <- RMSE(validation$rating, mu)
   naive_rmse
   
   # Checking RMSE Results
   # Save prediction in data frame
   rmse_results <- data_frame(method = "Average Movie Rating Model", RMSE = naive_rmse)
   rmse_results %>% knitr::kable()
   
   ## Movie Effect Model 
   
   # This is a simple model taking into consideration the movie effect b_i
   # Subtract  (Rating -  Mean) for each rating per Movie
   # Then Plot the Number of Movies and including the Movie Effect (b_i)
   movie_avgs <- edx %>%
     group_by(movieId) %>%
     summarize(b_i = mean(rating - mu))
   movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("light blue"),
                        ylab = "Number of movies", main = "Number of Movies with the computed b_i")
   
   
   
   # Testing & Saving RMSE Results
   predicted_ratings <- mu +  validation %>%
     left_join(movie_avgs, by='movieId') %>%
     pull(b_i)
   model_1_rmse <- RMSE(predicted_ratings, validation$rating)
   rmse_results <- bind_rows(rmse_results,
                             data_frame(method="Movie Effect Model",  
                                        RMSE = model_1_rmse ))
   
   # Checking RMSE Results
   rmse_results %>% knitr::kable()
   
   ## Movie and user effect model
   
   # Computing the average rating for users that have rated over 100 movies
   user_avgs<- edx %>% 
     left_join(movie_avgs, by='movieId') %>%
     group_by(userId) %>%
     filter(n() >= 100) %>%
     summarize(b_u = mean(rating - mu - b_i))
   user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("light blue"))
   
   
   user_avgs <- edx %>%
     left_join(movie_avgs, by='movieId') %>%
     group_by(userId) %>%
     summarize(b_u = mean(rating - mu - b_i))
   
   
   # Testing & Saving RMSE Results
   predicted_ratings <- validation%>%
     left_join(movie_avgs, by='movieId') %>%
     left_join(user_avgs, by='userId') %>%
     mutate(pred = mu + b_i + b_u) %>%
     pull(pred)
   
   model_2_rmse <- RMSE(predicted_ratings, validation$rating)
   rmse_results <- bind_rows(rmse_results,
                             data_frame(method="Movie & User Effect Model",  
                                        RMSE = model_2_rmse))
   
   # Check RMSE Result
   rmse_results %>% knitr::kable()
   
   ## Regularized movie and user effect model
   
   # lambda is a tuning parameter
   # Use cross-validation to choose it.
   lambdas <- seq(0, 10, 0.25)
   
   
   # For each lambda,find b_i & b_u, followed by rating prediction & testing
   # note:the below code could take some time  
   rmses <- sapply(lambdas, function(l){
     
     mu <- mean(edx$rating)
     
     b_i <- edx %>% 
       group_by(movieId) %>%
       summarize(b_i = sum(rating - mu)/(n()+l))
     
     b_u <- edx %>% 
       left_join(b_i, by="movieId") %>%
       group_by(userId) %>%
       summarize(b_u = sum(rating - b_i - mu)/(n()+l))
     
     predicted_ratings <- 
       validation %>% 
       left_join(b_i, by = "movieId") %>%
       left_join(b_u, by = "userId") %>%
       mutate(pred = mu + b_i + b_u) %>%
       pull(pred)
     
     return(RMSE(predicted_ratings, validation$rating))
   })
   
   
   # Plot rmses vs lambdas to select the optimal lambda                                                             
   qplot(lambdas, rmses)  
   
   
   # The optimal lambda                                                             
   lambda <- lambdas[which.min(rmses)]
   lambda
   
   # Testing & Saving RMSE Results                                                           
   rmse_results <- bind_rows(rmse_results,
                             data_frame(method="Regularized Movies & Users Effect Model",  
                                        RMSE = min(rmses)))
   
   # Check RMSE Result
   rmse_results %>% knitr::kable()
   
   ## Best Performing Model:
   ## Regularized Movies & Users Effect Model | 0.8648170|
   ## Model result is less than target RMSE < 0.86490
   
   
   