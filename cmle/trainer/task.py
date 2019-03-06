from trainer import model
import tensorflow as tf
import argparse


def run_it(args):
  dp = model.data_pipeline(args.feature_file, args.label_file)

  train_input = lambda: dp.training()
  eval_input = lambda: dp.testing()

  train_spec = tf.estimator.TrainSpec(train_input, max_steps=10000)
  eval_spec = tf.estimator.EvalSpec(eval_input, steps=999)

  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=args.model_dir)

  estimator = model.build_estimator(config=run_config)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--feature-file', required=True)
  parser.add_argument('--label-file', required=True)
  parser.add_argument('--model-dir', required=True)

  args = parser.parse_args()

  run_it(args)
