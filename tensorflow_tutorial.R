library(tensorflow)
library(keras)
library(tfdatasets)
library(coro)

### Download dataset
url <- "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset <- get_file(
  "aclImdb_v1",
  url,
  untar = TRUE,
  cache_dir = '.',
  cache_subdir = ''
)

dataset_dir <- file.path("aclImdb")


train_dir <- file.path(dataset_dir, 'train')
list.files(train_dir)

sample_file <- file.path(train_dir, 'pos/1181_9.txt')
readr::read_file(sample_file)

remove_dir <- file.path(train_dir, 'unsup')
unlink(remove_dir, recursive = TRUE)




batch_size <- 32
seed <- 42

raw_train_ds <- text_dataset_from_directory(
  'aclImdb/train',
  batch_size = batch_size,
  validation_split = 0.2,
  subset = 'training',
  seed = seed
)

batch <- raw_train_ds %>%
  reticulate::as_iterator() %>%
  coro::collect(n = 1)

cat("Label 0 corresponds to", raw_train_ds$class_names[1])
cat("Label 1 corresponds to", raw_train_ds$class_names[2])

raw_val_ds <- text_dataset_from_directory(
  'aclImdb/train',
  batch_size = batch_size,
  validation_split = 0.2,
  subset = 'validation',
  seed = seed
)

raw_test_ds <- text_dataset_from_directory(
  'aclImdb/test',
  batch_size = batch_size
)



### Prepare the dataset for training

# creating a regex with all punctuation characters for replacing.
re <- reticulate::import("re")

punctuation <- c("!", "\\", "\"", "#", "$", "%", "&", "'", "(", ")", "*",
                 "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[",
                 "\\", "\\", "]", "^", "_", "`", "{", "|", "}", "~")

punctuation_group <- punctuation %>%
  sapply(re$escape) %>%
  paste0(collapse = "") %>%
  sprintf("[%s]", .)

custom_standardization <- function(input_data) {
  lowercase <- tf$strings$lower(input_data)
  stripped_html <- tf$strings$regex_replace(lowercase, '<br />', ' ')
  tf$strings$regex_replace(
    stripped_html,
    punctuation_group,
    ""
  )
}


max_features <- 10000
sequence_length <- 250

vectorize_layer <- layer_text_vectorization(
  standardize = custom_standardization,
  max_tokens = max_features,
  output_mode = "int",
  output_sequence_length = sequence_length
)

# Make a text-only dataset (without labels), then call adapt
train_text <- raw_train_ds %>%
  dataset_map(function(text, label) text)
vectorize_layer %>% adapt(train_text)


vectorize_text <- function(text, label) {
  text <- tf$expand_dims(text, -1L)
  list(vectorize_layer(text), label)
}


# retrieve a batch (of 32 reviews and labels) from the dataset
batch <- reticulate::as_iterator(raw_train_ds) %>%
  reticulate::iter_next()
first_review <- as.array(batch[[1]][1])
first_label <- as.array(batch[[2]][1])
cat("Review:\n", first_review)

cat("Label: ", raw_train_ds$class_names[first_label+1])
cat("Vectorized review: \n")

train_ds <- raw_train_ds %>% dataset_map(vectorize_text)
val_ds <- raw_val_ds %>% dataset_map(vectorize_text)
test_ds <- raw_test_ds %>% dataset_map(vectorize_text)



AUTOTUNE <- tf$data$AUTOTUNE

train_ds <- train_ds %>%
  dataset_cache() %>%
  dataset_prefetch(buffer_size = AUTOTUNE)
val_ds <- val_ds %>%
  dataset_cache() %>%
  dataset_prefetch(buffer_size = AUTOTUNE)
test_ds <- test_ds %>%
  dataset_cache() %>%
  dataset_prefetch(buffer_size = AUTOTUNE)



## Create the model
embedding_dim <- 16

model <- keras_model_sequential() %>%
  layer_embedding(max_features + 1, embedding_dim) %>%
  layer_dropout(0.2) %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(0.2) %>%
  layer_dense(1)

summary(model)


model %>% compile(
  loss = loss_binary_crossentropy(from_logits = TRUE),
  optimizer = 'adam',
  metrics = metric_binary_accuracy(threshold = 0)
)


epochs <- 20
history <- model %>%
  fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
  )


# Evaluate model
model %>% evaluate(test_ds)

# Export model
export_model <- keras_model_sequential() %>%
  vectorize_layer() %>%
  model() %>%
  layer_activation(activation = "sigmoid")

export_model %>% compile(
  loss = loss_binary_crossentropy(from_logits = FALSE),
  optimizer = "adam",
  metrics = 'accuracy'
)

# Test it with `raw_test_ds`, which yields raw strings
export_model %>% evaluate(raw_test_ds)


examples <- c(
  "Why is this great",
  "The movie was one of the worst movies of all times",
  "The movie was terrible..."
)

predict(export_model, examples)
