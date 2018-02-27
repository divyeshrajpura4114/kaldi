#!/usr/bin/env python3

# Copyright  2017  Ke Li 
# License: Apache 2.0.

import os
import argparse
import sys
from collections import defaultdict

parser = argparse.ArgumentParser(description="This script gets the word-count pairs from conversations.",
                                 epilog="E.g. " + sys.argv[0] + " --vocab-file=data/rnnlm/vocab/words.txt "
                                        "--data-weights-file=exp/rnnlm/data_weights.txt data/rnnlm/data "
                                        "> exp/rnnlm/train.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--vocab-file", type=str, default='', required=True,
                    help="Specify the vocab file.")
parser.add_argument("--unk-word", type=str, default='',
                    help="String form of unknown word, e.g. <unk>.  Words in the counts "
                    "but not present in the vocabulary will be mapped to this word. "
                    "If the empty string, we act as if there is no unknown-word, and "
                    "OOV words are treated as an error.")
parser.add_argument("--data-weights-file", type=str, default='', required=True,
                    help="File that specifies multiplicities and weights for each data source: "
                    "e.g. if <text_dir> contains foo.txt and bar.txt, then should have lines "
                    "like 'foo 1 0.5' and 'bar 5 1.5'.  These "
                    "don't have to sum to on.")
parser.add_argument("--smooth-unigram-counts", type=float, default=1.0,
                    help="Specify the constant for smoothing. We will add "
                         "(smooth_unigram_counts * num_words_with_non_zero_counts / vocab_size) "
                         "to every unigram counts.")
parser.add_argument("--output-path", type=str, default='', required=True,
                    help="Specify the location of output files.")
parser.add_argument("text_dir",
                    help="Directory in which to look for data")

args = parser.parse_args()


SPECIAL_SYMBOLS = ["<eps>", "<s>", "<brk>"]

# get the name with txt and counts file path for all data sources except dev
# return a dict with key is the name of data_source,
#                    value is a tuple (txt_file_path, counts_file_path)
def get_all_data_sources(text_dir):
    data_sources = {}
    for f in os.listdir(text_dir):
        full_path = text_dir + "/" + f
        # if f == 'dev.txt' or f == 'dev.counts' or os.path.isdir(full_path):
        #     continue
        if f.endswith(".txt"):
            name = f[0:-4]
            if name in data_sources:
                data_sources[name] = (full_path, data_sources[name][1])
            else:
                data_sources[name] = (full_path, None)
        elif f.endswith(".counts"):
            name = f[0:-7]
            if name in data_sources:
                data_sources[name] = (data_sources[name][0], full_path)
            else:
                data_sources[name] = (None, full_path)
        else:
            sys.exit(sys.argv[0] + ": Text directory should not contain files with suffixes "
                     "other than .txt or .counts: " + f)

    for name, (txt_file, counts_file) in data_sources.items():
        if txt_file is None or counts_file is None:
            sys.exit(sys.argv[0] + ": Missing .txt or .counts file for data source: " + name)

    return data_sources


# read the data-weights for data_sources from weights_file
# return a dict with key is name of a data source,
#                    value is a tuple (repeated_times_per_epoch, weight)
def read_data_weights(weights_file, data_sources):
    data_weights = {}
    with open(weights_file, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                fields = line.split()
                assert len(fields) == 3
                if fields[0] in data_weights:
                    raise Exception("duplicated data source({0}) specified in "
                                    "data-weights: {1}".format(fields[0], weights_file))
                data_weights[fields[0]] = (int(fields[1]), float(fields[2]))
            except Exception as e:
                sys.exit(sys.argv[0] + ": bad data-weights line: '" +
                         line.rstrip("\n") + "': " + str(e))


    for name in data_sources.keys():
        if name not in data_weights:
            sys.exit(sys.argv[0] + ": Weight for data source '{0}' not set".format(name))

    return data_weights


# read the voab
# return the vocab, which is a dict mapping the word to a integer id.
def read_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            assert len(fields) == 2
            if fields[0] in vocab:
                sys.exit(sys.argv[0] + ": duplicated word({0}) in vocab: {1}"
                                       .format(fields[0], vocab_file))
            vocab[fields[0]] = int(fields[1])

    # check there is no duplication and no gap among word ids
    sorted_ids = sorted(vocab.values())
    for idx, id in enumerate(sorted_ids):
        assert idx == id
    if args.unk_word != '' and args.unk_word not in vocab:
        sys.exit(sys.argv[0] + "--unk-word={0} does not appear in vocab file {1}".format(
            args.unk_word, vocab_file))
    return vocab

def get_conv_counts(data_sources, data_weights, vocab):
    fisher = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    fisher_sub = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    dev_fisher = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    swbd = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    swbd_sub = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    dev_swbd = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    for name, (text_file, _) in data_sources.items():
        weight = data_weights[name][0] * data_weights[name][1]
        if weight == 0.0:
            continue

        if name == "swbd":
            swbd = get_conv_swbd(text_file, weight, vocab)
        if name == "swbd_sub":
            swbd_sub = get_conv_swbd(text_file, weight, vocab)
        if name == "dev_swbd":
            dev_swbd = get_conv_swbd(text_file, weight, vocab)
        if name == "fisher":
            fisher = get_conv_fisher(text_file, weight, vocab)
        if name == "fisher_sub":
            fisher_sub = get_conv_fisher(text_file, weight, vocab)
        if name == "dev_fisher":
            dev_fisher = get_conv_fisher(text_file, weight, vocab)

    return fisher, fisher_sub, dev_fisher, swbd, swbd_sub, dev_swbd

def get_conv_fisher(filename, weight, vocab):
    counts = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    total_counts = defaultdict(lambda:defaultdict())
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 2:
                continue
            if len(fields[0]) < 13:
                continue
            conv_id = fields[0][:11] # conversation id is "fe_0x_xxxxx"
            # print(conv_id)
            speaker = fields[0][-2] # fisher id is like "fe_03_00001_a:"
            # print(speaker)
            for word in fields[1:]:
                if word not in vocab:
                    if args.unk_word == '':
                        sys.exit(sys.argv[0] + ": error: an OOV word {0} is present in the "
                             "text file {1} but you have not specified an unknown word to "
                            "map it to (--unk-word option).".format(word, filename))
                    else:
                        word = args.unk_word
                if word in counts[conv_id][speaker]:
                    counts[conv_id][speaker][word] += weight
                else:
                    counts[conv_id][speaker][word] = weight
                if speaker == 'a':
                    if speaker in total_counts[conv_id]:
                        total_counts[conv_id][speaker] += 1
                    else:
                        total_counts[conv_id][speaker] = 1 
    # normalize training data with data_weight
    for conv_id, speaker_map in counts.items():
        for speaker, word_counts in speaker_map.items():
            if speaker == 'a':
                for word, count in word_counts.items():
                    counts[conv_id][speaker][word] = count * 1.0 / total_counts[conv_id][speaker]
    return counts

def get_conv_swbd(filename, weight, vocab):
    counts = defaultdict(lambda:defaultdict(lambda:defaultdict()))
    total_counts = defaultdict(lambda:defaultdict())
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            assert len(fields) >= 2
            conv_id = fields[0][:7] # conversation id is "swxxxxx" where xxx are numbers
            speaker = fields[0][8].lower() # swbd id is like "sw04940-B_029056-029800"
            for word in fields[1:]:
                if word not in vocab:
                    if args.unk_word == '':
                        sys.exit(sys.argv[0] + ": error: an OOV word {0} is present in the "
                                 "counts file {1} but you have not specified an unknown word to "
                                 "map it to (--unk-word option).".format(word, filename))
                    else:
                        word = args.unk_word
                if word in counts[conv_id][speaker]:
                    counts[conv_id][speaker][word] += weight 
                else:
                    counts[conv_id][speaker][word] = weight 
                if speaker == 'a':
                    if speaker in total_counts[conv_id]:
                        total_counts[conv_id][speaker] += 1 
                    else:
                        total_counts[conv_id][speaker] = 1 
    # normalize training data with data_weight but not labels
    for conv_id, speaker_map in counts.items():
        for speaker, word_counts in speaker_map.items():
            if speaker == 'a':
                for word, count in word_counts.items():
                    counts[conv_id][speaker][word] = count * 1.0 / total_counts[conv_id][speaker]
    return counts

def write_data_to_file(data_name, data):
    with open(args.output_path + "/" + data_name + ".txt", 'w', encoding='utf-8') as f, \
    open(args.output_path + "/" + data_name + ".label.txt", 'w', encoding='utf-8') as f1:
        for conv_id, speaker in data.items():
            for id, word_probs in speaker.items():
                if id == 'a':
                    f.write("[ ")
                    for word, prob in word_probs.items():
                        f.write(str(vocab[word]) + " " + str(prob) + " ")
                    f.write("] ")
                if id == 'b':
                    f1.write("[ ")
                    for word, prob in word_probs.items():
                        f1.write(str(vocab[word]) + " " + str(prob) + " ")
                    f1.write("] ")

if os.system("rnnlm/ensure_counts_present.sh {0}".format(args.text_dir)) != 0:
    print(sys.argv[0] + ": command 'rnnlm/ensure_counts_present.sh {0}' failed.".format(
        args.text_dir))

data_sources = get_all_data_sources(args.text_dir)
data_weights = read_data_weights(args.data_weights_file, data_sources)
vocab = read_vocab(args.vocab_file)

fisher_input, fisher_sub_input, dev_fisher_input, swbd_input, swbd_sub_input, dev_swbd_input = get_conv_counts(data_sources, data_weights, vocab)

for name, (_, _) in data_sources.items():
    if name == "fisher":
        data = fisher_input
    if name == "fisher_sub":
        data = fisher_sub_input
    if name == "dev_fisher":
        data = dev_fisher_input
    if name == "swbd":
        data = swbd_input
    if name == "swbd_sub":
        data = swbd_sub_input
    if name == "dev_swbd":
        data = dev_swbd_input 
    write_data_to_file(name, data)

print(sys.argv[0] + ": generated swbd and fisher data.", file=sys.stderr)
