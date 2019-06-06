import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner

os.environ["CUDA_VISIBLE_DEVICES"] = ""
train = pd.read_csv('../data/feature.csv')
y = pd.read_csv('../data/label.csv')
# train = train[['gy_z.15', 'pitch.15', 'pitch.3', 'gy_z.3', 'acc_z.15', 'acc_z.13', 'magnetic.2', 'acc_z.3', 'acc_z.2',
#                'pitch.6', 'magnetic.6', 'm_z.15', 'gy_y.13', 'gy_y.8', 'acc_z.6', 'm_z.14', 'm_z.3', 'l_z.11',
#                'gy_z.14',
#                'acc_y.3', 'l_x.3', 'g_z.1', 'magnetic.3', 'l_y.3', 'gy_x.14', 'gy_y.2', 'magnetic.14', 'magnetic.15',
#                'm_x.11',
#                'roll.2', 'acc_z.5', 'roll.6', 'gy_x.4', 'l_y.11', 'gy_z.6', 'l_x.11', 'acc_y.14', 'm_x.15', 'acc_z'
#                ]]
train_x = train
train_y = y
test_x = pd.read_csv('../test/feature.csv')

test_y = pd.read_csv("../test/label.csv")

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

# Parameters
batch_size = 2048  # The number of samples per batch
num_classes = 9  # 分类的种类
num_features = train_x.shape[1]
max_steps = 1000  # 最大的轮次

# GBDT Parameters
learning_rate = 0.005  # 0.005 87.13%
l1_regul = 0.
l2_regul = 1.
examples_per_layer = 5000  #5000 86
num_trees = 75  # type: int # 75 86%
max_depth = 10

# Fill GBDT parameters into the config proto
learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER

learner_config.growing_mode = growing_mode
run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN

# Create a TensorFlor GBDT Estimator
gbdt_model = GradientBoostedDecisionTreeClassifier(

    model_dir=None,  # No save directory specified
    learner_config=learner_config,
    n_classes=num_classes,
    examples_per_layer=examples_per_layer,
    num_trees=num_trees,
    center_bias=False,
    config=run_config

)

tf.logging.set_verbosity(tf.logging.INFO)  # 打印出来训练的信息

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"i": np.asarray(train_x.astype("float32"))},
    y=np.asarray(train_y.astype("int")),
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True
)

# Train the Model

print(np.asarray(test_x.astype("float32")).dtype)
gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"i": np.asarray(test_x.astype("float32"))},
    y=np.asarray(test_y.astype("int")),
    batch_size=batch_size,
    shuffle=False
)
# Use the Estimator 'evaluate' method
e = gbdt_model.evaluate(input_fn=input_fn)

print("Testing Accuracy:", e['accuracy'])
# print(classification_report(test_y, e, digits=5))
