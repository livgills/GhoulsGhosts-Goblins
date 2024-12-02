library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)
library(discrim)
library(embed)
library(dplyr)
library(naivebayes)
library(themis)


missingtrain <- vroom("trainWithMissingValues.csv")
train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(type ~ . , data=missingtrain) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(rotting_flesh, hair_length, has_soul), neighbors = 7) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(bone_length, hair_length, has_soul), neighbors = 7) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(rotting_flesh, bone_length, has_soul), neighbors = 7)

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = missingtrain)

rmse_vec(train[is.na(missingtrain)],
         baked[is.na(missingtrain)])
 ######################
nn_recipe <- recipe(type~., data = train) %>% 
  update_role(id, new_role = "id") %>% 
  step_range(all_numeric_predictors(), min = 0 , max = 1)

nn_model <- mlp(hidden_units = tune(),
                 epochs = 50) %>% 
  set_engine("nnet") %>% 
  set_mode("classification")

nn_wf <- workflow() %>% 
  add_recipe(nn_recipe) %>% 
  add_model(nn_model)
folds <- vfold_cv(train, v = 10, repeats = 1)

nn_tuneGrid <- grid_regular(hidden_units(range = c(1, 50)),
                            levels = 10)
tuned_nn <- nn_wf %>% 
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))
tuned_nn %>% collect_metrics() %>% 
  filter(.metric=="accuracy") %>% 
  ggplot(aes(x = hidden_units, y = mean)) + geom_line()
bestTune<- tuned_nn %>%  select_best("accuracy")

final_wf <- nn_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data= train)
NN_preds <- final_wf %>% 
  predict(new_data = test, type = "class")

Neura1N_preds <- tibble(id = test$id,
                        type = NN_preds$.pred_class)
vroom_write(x=Neura1N_preds, file= "./NN_preds.csv", delim=",")


### BOOSTED MODEL

boosted_recipe <- recipe(type~., data = train) %>% 
  update_role(id, new_role = "id") %>% 
  step_range(all_numeric_predictors(), min = 0 , max = 1)

prep <- prep(boosted_recipe)
baked <- bake(prep, new_data = train)

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

boosted_wf <- workflow() %>% 
  add_recipe(boosted_recipe) %>% 
  add_model(boost_model)

boosted_tunegrid <- grid_regular(tree_depth(),
                                 trees(),
                                 learn_rate(),
                                 levels = 3)

folds_boost <- vfold_cv(train, v=5, repeats = 1)

cv_results <- boosted_wf %>% 
  tune_grid(resamples = folds_boost,
            grid = boosted_tunegrid,
            metrics = metric_set(accuracy))

besttune <- cv_results %>% 
  select_best()

final_wf <- boosted_wf %>% 
  finalize_workflow(besttune) %>% 
  fit(data = train)

boosted_preds <- final_wf %>% 
  predict(new_data = test, type = "class")

boost_preds <- tibble(id = test$id,
                        type = boosted_preds$.pred_class)
vroom_write(x=boost_preds, file= "./boosted_preds.csv", delim=",")


#####################

Final_recipe <- recipe(type ~. , data = train) %>% 
  
  step_mutate(color = as.factor(color)) %>% 
  step_mutate(id, features = id) %>% 
  
  step_normalize(all_numeric_predictors())

prep <- prep(Final_recipe)
baked <- bake(prep, new_data = train)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(Final_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace
tunegrid <- grid_regular(smoothness(range = c(.1,2)),
                         Laplace(),
                         levels = 30)
CV_folds <- vfold_cv(train, v = 30, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=CV_folds,
            grid=tunegrid,
            metrics=metric_set(roc_auc)) 

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

nb_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)
## Predict
nb_pred <- predict(nb_wf, new_data=test, type="class")

boost_preds <- tibble(id = test$id,
                      type = nb_pred$.pred_class)
vroom_write(x=boost_preds, file= "./NaiveBayes3_preds.csv", delim=",")


