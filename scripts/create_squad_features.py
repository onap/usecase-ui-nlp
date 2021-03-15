#!/usr/bin/env python
# coding: utf-8

# auther = 'liuzhiyong'
# date = 20201204


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json

import collections
import math
import tokenization
import six
import tensorflow as tf
import requests

from global_setting import _improve_answer_span

version_2_with_negative = True


def get_squad_feature_result(title, text, tokenizer, question, url):

    def make_json(title, text, question):
        res = {}
        res['data'] = []
        data_inside = {}

        data_inside['title'] = title
        data_inside['paragraphs'] = []
        paragraphs_inside = {}
        paragraphs_inside['context'] = text
        paragraphs_inside['qas'] = []
        for ques in question:
            qas_inside = {}
            qas_inside['answers'] = []

            answer_inside = {}

            qas_inside['id'] = ques
            qas_inside['question'] = ques
            qas_inside['answers'].append(answer_inside.copy())
            paragraphs_inside['qas'].append(qas_inside.copy())
        data_inside['paragraphs'].append(paragraphs_inside.copy())

        res['data'].append(data_inside.copy())
        return json.dumps(res)

    def _compute_softmax(scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def get_final_text(pred_text, orig_text, do_lower_case):

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if 0:
                tf.logging.info(
                    "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if 0:
                tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                                orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if 0:
                tf.logging.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if 0:
                tf.logging.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def _get_best_indexes(logits, n_best_size):

        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

    def write_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case):
        """Write final predictions to the json file and log-odds of null if needed."""

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = _get_best_indexes(result.start_logits, n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, n_best_size)
                # if we could have irrelevant answers, get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = result.start_logits[0] + result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))

            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit))

            # if we didn't inlude the empty option in the n-best, inlcude it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(
                        _NbestPrediction(
                            text="", start_logit=null_start_logit,
                            end_logit=null_end_logit))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            if not version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > 0:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text

            all_nbest_json[example.qas_id] = nbest_json
        return all_predictions

    def create_int_feature(values):

        feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))
        return feature

    class InputFeatures(object):
        """A single set of features of data."""

        def __init__(self,
                     unique_id,
                     example_index,
                     doc_span_index,
                     tokens,
                     token_to_orig_map,
                     token_is_max_context,
                     input_ids,
                     input_mask,
                     segment_ids,
                     start_position=None,
                     end_position=None,
                     is_impossible=None):
            self.unique_id = unique_id
            self.example_index = example_index
            self.doc_span_index = doc_span_index
            self.tokens = tokens
            self.token_to_orig_map = token_to_orig_map
            self.token_is_max_context = token_is_max_context
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids
            self.start_position = start_position
            self.end_position = end_position
            self.is_impossible = is_impossible

    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                     doc_stride, max_query_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000
        result = []

        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_position = None
                end_position = None
                if is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0

                if example_index < 20:
                    tf.logging.info("*** Example ***")
                    tf.logging.info("unique_id: %s" % (unique_id))
                    tf.logging.info("example_index: %s" % (example_index))
                    tf.logging.info("doc_span_index: %s" % (doc_span_index))
                    tf.logging.info("tokens: %s" % " ".join(
                        [tokenization.printable_text(x) for x in tokens]))
                    tf.logging.info("token_to_orig_map: %s" % " ".join(
                        ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                    tf.logging.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    tf.logging.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    tf.logging.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training and example.is_impossible:
                        tf.logging.info("impossible example")
                    if is_training and not example.is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        tf.logging.info("start_position: %d" % (start_position))
                        tf.logging.info("end_position: %d" % (end_position))
                        tf.logging.info(
                            "answer: %s" % (tokenization.printable_text(answer_text)))

                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible)

                # Run callback

                result.append(feature)
                unique_id += 1
        return result

    class SquadExample(object):

        def __init__(self,
                     qas_id,
                     question_text,
                     doc_tokens,
                     orig_answer_text=None,
                     start_position=None,
                     end_position=None,
                     is_impossible=False):
            self.qas_id = qas_id
            self.question_text = question_text
            self.doc_tokens = doc_tokens
            self.orig_answer_text = orig_answer_text
            self.start_position = start_position
            self.end_position = end_position
            self.is_impossible = is_impossible

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            s = ""
            s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
            s += ", question_text: %s" % (
                tokenization.printable_text(self.question_text))
            s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
            if self.start_position:
                s += ", start_position: %d" % (self.start_position)
            if self.start_position:
                s += ", end_position: %d" % (self.end_position)
            if self.start_position:
                s += ", is_impossible: %r" % (self.is_impossible)
            return s

    def read_squad_examples(input_file, is_training):
        """Read a SQuAD json file into a list of SquadExample."""

        input_data = json.loads(input_file)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:

                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                tokenization.whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                                   actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        return examples

    def get_result(title, text, question, url):

        data = make_json(title, text, question)

        examples = read_squad_examples(data, False)

        predict_files = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=512,
            doc_stride=128,
            max_query_length=100,
            is_training=False,
        )

        headers = {"content-type": "application/json"}
        all_results = []
        for predict_file in predict_files:
            features = {}
            features["unique_ids"] = predict_file.unique_id
            features["input_mask"] = predict_file.input_mask
            features["segment_ids"] = predict_file.segment_ids
            features["input_ids"] = predict_file.input_ids
            data_list = []
            data_list.append(features)

            data = json.dumps({"instances": data_list})

            json_response = requests.post(url, data=data, headers=headers)

            x = json.loads(json_response.text)

            all_results.append(
                RawResult(
                    unique_id=predict_file.unique_id,
                    start_logits=x['predictions'][0]['start_logits'],
                    end_logits=x['predictions'][0]['end_logits']))

        result = write_predictions(examples, predict_files, all_results, 20, 64, True)
        return result

    return get_result(title, text, question, url)
