library(tidyverse)
library(tidymodels)
library(jsonlite)
library(tidycensus)
library(here)
library(vip)
library(sf)
library(tigris)
theme_set(theme_minimal())
options(scipen = 2012,
        tigris_use_cache = TRUE)

set.seed(2015)

# use inspect option to find AJAX data source
url <- "https://covid.cdc.gov/covid-data-tracker/COVIDData/getAjaxData?id=vaccination_data"

# read in json data and grab vaccination_data first element
vac_cdc <- read_json(url)['vaccination_data'][[1]]

# transform into a tibble and rid of first row
df <- vac_cdc %>%
  map_df(as_tibble) %>%
  slice(-1) %>%
  mutate(LongName = if_else(LongName == "New York State", "New York", LongName)) 
#%>%
  janitor::clean_names()



glimpse(df)

# Updating.. --------------------------------------------------------------
# When scraping updated data check if columns are all the same -------------


colnames_df <- tibble(colnames = df %>%
                        colnames() %>%
                        sort())


# If the field types aren’t the same as those in the cache, then stop the script
if (isTRUE(all_equal(colnames_df, (read_csv(here("data", "fields.csv")))))) {
  # Write out new fields into a cache (ie a csv)
  colnames_df %>%
    write_csv(here("data", "fields.csv"))
  # Stop the script
  stop("CDC API schema has changed. Please see `data/fields.csv`")
}


# End Updating ------------------------------------------------------------



# Census Data -------------------------------------------------------------

my_vars <- c(
  tot_pop = "B02001_001",
  white = "B02001_002",
  black = "B02001_003",
  ame_ind = "B02001_004",
  asian = "B02001_005",
  alask_haw = "B02001_006",
  hispanic = "B03001_003",
  other_race = "B02001_007",
  income = "B19013_001"
)

acs_data <- get_acs(geography = "state",
                    variables = my_vars,
                    output = "wide",
                    geometry = TRUE) %>%
  tigris::shift_geometry()



df_merged <- acs_data %>%
  select(
    GEOID,
    NAME,
    tot_pop = tot_popE,
    white = whiteE,
    black = blackE,
    ame_ind = ame_indE,
    asian = asianE,
    hispanic = hispanicE,
    alask_haw = alask_hawE,
    other_race = other_raceE,
    income = incomeE
  ) %>%
  mutate(other_race = other_race + alask_haw + ame_ind) %>%
  select(-c(ame_ind, alask_haw)) %>%
  left_join(df, by = c("NAME" = "LongName")) %>%
  mutate(p_white = white / tot_pop,
         p_black = black / tot_pop,
         p_asian = asian / tot_pop,
         p_hispanic = hispanic / tot_pop,
         p_other_race = other_race / tot_pop) 




# Plot Vaccination Rates --------------------------------------------------

ggplot(df_merged, aes(fill = Admin_Per_100k_12Plus)) +
  geom_sf(lwd = 0.2) +
  scale_fill_distiller(type = "div",
                       palette = "RdBu") + 
  ggthemes::theme_map() +
  labs(fill = "Administered Per 100k",
       title = "CDC Doses Administered Per 100K.",
       subtitle = "Data Scraped from\n https://covid.cdc.gov/covid-data-tracker") +
  theme(plot.title = element_text(family = "sans", size = 18, hjust = 0.5, face = "bold"),
        legend.text = element_text(family = "sans", size = 9),
        legend.title = element_text(family = "sans", size = 11))

library(tmap)

tm_shape(df_merged) +
  tm_fill(col = "Admin_Per_100k_12Plus", 
          style = "quantile",
          n = 5) +
  tm_borders()



#ggsave("doses_admin_plot.jpg", dpi = 400)





# Modelings ---------------------------------------------------------------


# select variables for modeling
df <- df_merged %>% 
  sf::st_drop_geometry() %>%
  select(Admin_Per_100K, p_white, p_black, p_asian, p_other_race, p_hispanic, income, NAME)


train_fold <- df %>%
  vfold_cv(v = 5)



# Glmnet Linear Regression.
reg_lm_spec <-
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine('glmnet')


metric <- metric_set(rmse)

grid_control <- control_grid(verbose = TRUE,
                             save_pred = TRUE,
                             save_workflow = TRUE,
                             extract = extract_model)


lin_rec <- recipe(Admin_Per_100K ~ p_white + p_black + p_asian + p_hispanic + income,
                  data = df) %>%
  #step_interact(terms = ~ income*starts_with("p_")) %>%
  step_ns(income, deg_free = 2) %>%
  step_normalize(all_numeric(), -all_outcomes())


# create workflow
lin_wk <- workflow() %>%
  add_recipe(lin_rec) %>%
  add_model(reg_lm_spec)


# train model
lin_tune <- lin_wk %>%
  tune_grid(train_fold, 
            metrics = metric,
            control = control_grid(save_pred = TRUE,
                                   verbose = TRUE))

lin_tune %>%
  collect_metrics() %>%
  arrange(mean)

# plot of tuning parameters
lin_tune %>%
  collect_metrics() %>%
  mutate(penalty = log(penalty)) %>%
  pivot_longer(c(penalty, mixture), names_to = "param", values_to = "value") %>%
  ggplot(aes(value, mean, color = param)) +
  geom_point() +
  geom_line() + 
  facet_wrap(~param, scales = "free") 

# easier plot of tuning params but not as beautiful 
autoplot(lin_tune)



lin_best <- lin_wk %>%
  finalize_workflow(select_best(lin_tune)) %>%
  fit(df)

  
# get ceofs
lin_best %>%
  extract_fit_parsnip() %>%
  pluck("fit") %>%
  coef(s = 0.1)

# another way
lin_best %>%
  extract_fit_parsnip() %>%
  tidy()
  
  
# plot variable importance
lin_best %>%
  extract_fit_parsnip() %>% 
  vip()

# attach data with predicitons
df_pred <- lin_best %>%
  augment(df)

df_pred %>%
  ggplot(aes(Admin_Per_100K, .pred)) +
  geom_point() +
  geom_smooth(method = 'lm') +
  labs(title = "True vs Predicted Values",
       x = "Administerd Shots per 100k",
       y = "predicted values")



df_merged %>%
  left_join(df_pred, by = c("NAME" = "NAME")) %>%
  ggplot(aes(fill = .pred)) +
  geom_sf() +
  scale_fill_gradient()

  

# Xgboost Time ------------------------------------------------------------

set.seed(2021)

# Xgboost Regression
xgboost_spec <-
  boost_tree(
    tree_depth = tune(),
    trees = 1200,
    learn_rate = tune(),
    min_n = tune(), 
    mtry = tune()
  )  %>%
  set_engine('xgboost') %>%
  set_mode('regression')

## xg boost recipe
xg_rec <- recipe(Admin_Per_100K ~ p_white + p_black + p_asian + p_hispanic + income + NAME,
                 data = df) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


xg_wf <- workflow() %>%
  add_recipe(xg_rec) %>%
  add_model(xgboost_spec)


xg_tune <- xg_wf %>%
  tune_grid(train_fold,
            metrics = metric,
            control = grid_control)

# plot tuning params
xg_tune %>%
  collect_metrics() %>%
  mutate(learn_rate = log(learn_rate)) %>%
  pivot_longer(c(mtry, min_n, tree_depth, learn_rate), names_to = "param", values_to = "value") %>%
  ggplot(aes(value, mean, color = param)) +
  geom_point() +
  geom_line() + 
  facet_wrap(~param, scales = "free") 


autoplot(xg_tune)

# table for metrics of best models 
xg_tune %>%
  collect_metrics() %>%
  arrange(mean)

xg_wf_best <- xg_wf %>%
  finalize_workflow(select_best(xg_tune))

xg_wf_best <- xg_wf_best %>%
  fit(df) 



## attaching predictions on df because when splitting data not all states were present
pred_df <- xg_wf_best %>%
  augment(df) %>%
  select(NAME, .pred)


df_merged %>%
  left_join(pred_df, by = c("NAME" = "NAME")) %>%
  ggplot() +
  geom_sf(aes(fill = .pred), lty = 1) +
  scale_fill_gradient() +
  labs(title = "XG Boost prediction of Vaccine shots per 100k.",
       fill = "Prediction") +
  ggeasy::easy_center_title()


xg_wf_best %>%
  extract_fit_parsnip() %>%
  vip(geom = "col") +
  theme_minimal()



