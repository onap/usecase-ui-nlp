
#!/usr/bin/env python
# coding: utf-8

# auther = 'liuzhiyong'
# date = 20201204


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import datetime
import threading
import time
from flask import Flask, abort, request, jsonify
from concurrent.futures import ThreadPoolExecutor

import collections
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import sys
from api_squad import *
from global_setting import *
from global_setting import FLAGS_bert_config_file, FLAGS_vocab_file, FLAGS_init_checkpoint_squad, questions

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)

app = Flask(__name__)

def serving_input_fn():
    input_ids = tf.placeholder(tf.int32, [None, FLAGS_max_seq_length], name='input_ids')
    unique_id = tf.placeholder(tf.int32,[None])
    input_mask = tf.placeholder(tf.int32, [None, FLAGS_max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS_max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'unique_ids': unique_id,
        })()
    return input_fn

def main(FLAGS_output_dir, FLAGS_init_checkpoint_squad, FLAGS_export_dir, FLAGS_predict_file=None, FLAGS_train_file=None, FLAGS_do_predict=False,
         FLAGS_do_train=False, FLAGS_train_batch_size=16, FLAGS_predict_batch_size=8, FLAGS_learning_rate=5e-5, FLAGS_num_train_epochs=3.0,
         FLAGS_max_answer_length=100, FLAGS_max_query_length=64, FLAGS_version_2_with_negative=False):
    
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS_bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS_output_dir)


    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS_vocab_file, do_lower_case=FLAGS_do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS_use_tpu and FLAGS_tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS_tpu_name, zone=FLAGS_tpu_zone, project=FLAGS_gcp_project)
   
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS_master,
        model_dir=FLAGS_output_dir,
        save_checkpoints_steps=FLAGS_save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS_iterations_per_loop,
            num_shards=FLAGS_num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS_do_train:
        train_examples = read_squad_examples(
            input_file=FLAGS_train_file, is_training=True,questions = questions,FLAGS_version_2_with_negative = FLAGS_version_2_with_negative)
     
        num_train_steps = int(
            len(train_examples) / FLAGS_train_batch_size * FLAGS_num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS_warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS_init_checkpoint_squad,
        learning_rate=FLAGS_learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS_use_tpu,
        use_one_hot_embeddings=FLAGS_use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS_use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS_train_batch_size,
        predict_batch_size=FLAGS_predict_batch_size)

    if FLAGS_do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS_output_dir, "train.tf_record"),
            is_training=True)
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS_max_seq_length,
            doc_stride=FLAGS_doc_stride,
            max_query_length=FLAGS_max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS_train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS_max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        estimator._export_to_tpu = False
        estimator.export_savedmodel(FLAGS_export_dir, serving_input_fn)
    return 'success'


class AI2Flask:

    def __init__(self, port=5000, workers=4):
        self.app = app
        self.port = port
        p = ThreadPoolExecutor(max_workers=workers)
        threads_mapping = {}

        def check_threads():
            flag = False
            pop_keys = set()
            if len(threads_mapping) >= workers:
                for k, v in threads_mapping.items():
                    if v.running():
                        flag = True
                    else:
                        pop_keys.add(k)

            for k in pop_keys:
                threads_mapping.pop(k)

            return flag

        @app.route('/api/offline/train', methods=['POST'])
        def text_analyse():
            if not request.json or not 'task_id' in request.json:
                abort(400)
            if check_threads():
                return jsonify({"Des": "Task list is full. Can not submit new task! ", "Result": "Failed to submit the training task ", "Status": "ERROR"})

            else:
                try:
                    FLAGS_train_batch_size = request.json['FLAGS_train_batch_size']
                except:
                    FLAGS_train_batch_size = 16
                try:
                    FLAGS_learning_rate = request.json['FLAGS_learning_rate']
                except:
                    FLAGS_learning_rate = 5e-5
                try:
                    FLAGS_num_train_epochs = request.json['FLAGS_num_train_epochs']
                except:
                    FLAGS_num_train_epochs = 3.0
                try:
                    FLAGS_max_answer_length = request.json['FLAGS_max_answer_length']
                except:
                    FLAGS_max_answer_length = 100
                try:
                    FLAGS_max_query_length = request.json['FLAGS_max_query_length']
                except:
                    FLAGS_max_query_length = 64
                try:
                    FLAGS_version_2_with_negative = request.json['FLAGS_version_2_with_negative']
                except:
                    FLAGS_version_2_with_negative = True

                try:
                    FLAGS_predict_file = None
                    FLAGS_predict_batch_size = 8
                    FLAGS_do_predict = False
                    FLAGS_do_train = True
                    FLAGS_output_dir = request.json['FLAGS_output_dir']
                    FLAGS_train_file = request.json['FLAGS_train_file']
                    FLAGS_export_dir = request.json['FLAGS_export_dir']
                    task_id = request.json['task_id']

                    task = p.submit(main, FLAGS_output_dir, FLAGS_init_checkpoint_squad, FLAGS_export_dir, FLAGS_predict_file, FLAGS_train_file, FLAGS_do_predict,
                                    FLAGS_do_train, FLAGS_train_batch_size, FLAGS_predict_batch_size, FLAGS_learning_rate, FLAGS_num_train_epochs,
                                    FLAGS_max_answer_length, FLAGS_max_query_length, FLAGS_version_2_with_negative)
                    threads_mapping[task_id] = task

                    return jsonify({"message": "Task submitted successfully", "status": "0"})

                except KeyError as e:
                    return jsonify({"Des": 'KeyError: {}'.format(str(e)), "Result": 'None', "Status": "Error"})
                except Exception as e:
                    return jsonify({"Des": str(e), "Result": 'None', "Status": "Error"})

       

        @app.route('/api/offline/status', methods=['POST'])
        def todo_status():
            task_id = request.json['task_id']
            task = threads_mapping.get(task_id, None)
            try:
                if task is None:
                    return jsonify({'Des': 'The task was not found', 'Status': 'ERROR'})
                else:
                    if task.done():
                        print(task.result)
                        if task.result() == 'success':
                            return jsonify({'Des': 'DONE', 'Status': 'OK'})
                        else:
                            return jsonify({'Des': 'Program execution error. Please check the execution log ', 'Status': 'ERROR'})

                    else:
                        return jsonify({'Des': 'RUNNING', 'Status': 'OK'})
            except Exception as e:
                return jsonify({'Des': str(e), 'Status': 'ERROR'})

    def start(self):
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)


if __name__ == '__main__':
    port = sys.argv[1]
    AI2Flask(port=port).start()
