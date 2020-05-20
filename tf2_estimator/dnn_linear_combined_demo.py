import tensorflow as tf


from urllib.request import urlretrieve


gender = tf.feature_column.categorical_column_with_vocabulary_list(
     "gender",
     ["Female","Male"]
)
race = tf.feature_column.categorical_column_with_vocabulary_list(
    "race",
    ["Amer-Indian-Eskimo","Asian-Pac-Islander","Black","Other","White"]
)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=100)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=100)
occupation =tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country",hash_bucket_size=1000)

age = tf.feature_column.numeric_column("age")
age_buckets = tf.feature_column.bucketized_column(
    age,
    boundaries = [18,25,30,35,40,45,50,55,60,65]
)
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

wide_columns = [
    gender, native_country, education, occupation, workclass,
    relationship, age_buckets,
    tf.feature_column.crossed_column(["education","occupation"],hash_bucket_size=int(1e4)),
    tf.feature_column.crossed_column(["native_country","occupation"],hash_bucket_size=int(1e4)),
    tf.feature_column.crossed_column([age_buckets,"education","occupation"],hash_bucket_size=int(1e6))
]

deep_columns = [
    tf.feature_column.embedding_column(workclass,dimension=8),
    tf.feature_column.embedding_column(education,dimension=8),
    tf.feature_column.embedding_column(gender,dimension=8),
    tf.feature_column.embedding_column(relationship,dimension=8),
    tf.feature_column.embedding_column(native_country,dimension=8),
    tf.feature_column.embedding_column(occupation,dimension=8),
    age,education_num,capital_gain,capital_loss,hours_per_week
]



import pandas as pd
import urllib

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race","gender",
    "capital_gain","capital_loss","hours_per_week","native_country",
    "income_bracket"
]

DEFAULT_COLUMNS_VALUE = [
    [0],[''],[''],[''],[0]
    ,[''],[''],[''],[''],['']
    ,[0],[0],[0],['']
    ,['']
]

LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = [
    "workclass","education","marital_status","occupation",
    "relationship","race","gender","native_country"
]
CONTINUOUS_COLUMNS = [
    "age", "education_num","capital_gain","capital_loss","hours_per_week"
]


from tensorflow_core.python import string_split, string_to_number, decode_csv

train_file = "./data/train"
test_file = "./data/test"

import os

if not os.path.isfile(test_file):
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",train_file)
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",test_file)

# df_train = pd.read_csv(train_file,names=COLUMNS,skipinitialspace=True)
# df_test = pd.read_csv(test_file,names=COLUMNS,skipinitialspace=True,skiprows=1)
#
# df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x:'>50k' in x)).astype(int)
# df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x:'>50k' in x)).astype(int)
#
# def input_fn(df):
#     continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
#     categorical_cols = {
#         k: tf.SparseTensor(
#             indices = [[i,0] for i in range(df[k].size)],
#             values = df[k].values,
#             dense_shape = [df[k].size,1]
#         )
#         for k in CATEGORICAL_COLUMNS
#     }
#
#     feature_cols = {}
#     feature_cols.update(continuous_cols)
#     feature_cols.update(categorical_cols)
#
#     label = tf.constant(df[LABEL_COLUMN].values)
#     return feature_cols,label



def dataset_input_fn(filename, num_epochs, batch_size):
    def parse_line(line):
        data = decode_csv(line, record_defaults=DEFAULT_COLUMNS_VALUE)
        feature_cols = dict(zip(COLUMNS,data))
        label = tf.equal(feature_cols["income_bracket"],">50K")
        feature_cols.pop("income_bracket")
        return feature_cols , label
    dataset = tf.data.TextLineDataset(filename).skip(1)
    dataset = dataset.map(parse_line)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

def train_input_fn():
    return dataset_input_fn(train_file,5,32)

def eval_input_fn():
    return dataset_input_fn(test_file,5,32)

model_dir = "./model/dlcd"
wdmodel = tf.estimator.DNNLinearCombinedClassifier(
    model_dir = model_dir,
    linear_feature_columns = wide_columns,
    linear_optimizer='Ftrl',
    dnn_feature_columns = deep_columns,
    dnn_optimizer='Adagrad',
    dnn_hidden_units = [100,50]
)
wdmodel.train(input_fn = train_input_fn,steps=200)
results = wdmodel.evaluate(input_fn=eval_input_fn,steps=20)
for key in sorted(results):
    print("%s: %s" % (key,results[key]))
