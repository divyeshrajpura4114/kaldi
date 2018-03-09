#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2018  Ke Li

# Begin configuration section.

dir=exp/dnn
embedding_dim=512
stage=-10
train_stage=-10
num_epochs=30
lr=0.01
flr=0.001
minibatch_size=8

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

text=data/train/text
text_dir=data/dnn/text
mkdir -p $dir/config
set -e

for f in $text; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/swbd1_data_prep.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  # prepare swbd data
  mkdir -p $text_dir
  cp $text $text_dir/swbd_all.txt
  # get ~ 1/50 swbd data as dev data
  head -n 5112 $text_dir/swbd_all.txt > $text_dir/dev_swbd.txt
  tail -n +5113 $text_dir/swbd_all.txt > $text_dir/swbd.txt
  # get ~ 1/50 swbd data as training diagonistic data
  tail -n 5905 $text_dir/swbd.txt > $text_dir/swbd_sub.txt

  rm $text_dir/swbd_all.txt

  # prepare fisher data
  fisher_dirs="/export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"
  [ -f $text_dir/fisher_all.txt ] && rm $text_dir/fisher_all.txt
  [ -f $text_dir/fisher_all.txt~ ] && rm $text_dir/fisher_all.txt~
  for x in ${fisher_dirs[@]}; do
    [ ! -d $x/data/trans ] \
      && "$0: Cannot find transcripts in Fisher directory $x" && exit 1;
    for file in $x/data/trans/*/*.txt; do
      name=`echo $file | awk -F'[/.]' '{print $11}'`
      # construct keys for fisher data and remove unnecessary columns (2, 3, 4)
      cat $file | grep -v ^# | grep -v ^$ | local/fisher_map_words.pl | \
        awk -v key=$name '{$1=key"_"$3" "$1}1' | cut -d ' ' -f-1,5- >> $text_dir/fisher_all.txt
    done
  done
  # get 1/100 fisher data as dev data
  head -n 22213 $text_dir/fisher_all.txt > $text_dir/dev_fisher.txt 
  tail -n +22214 $text_dir/fisher_all.txt > $text_dir/fisher.txt
  # get 1/100 fisher data as training diagnositic data
  tail -n 22770 $text_dir/fisher.txt > $text_dir/fisher_sub.txt 

  rm $text_dir/fisher_all.txt
fi

if [ $stage -le 1 ]; then
  # cp data/lang/words.txt $dir/config/
  wordlist=/export/b03/hxu/tf-pr/kaldi/egs/swbd/s5_8/data/lang/words.txt # 40k
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
swbd   1   1.0
swbd_sub    1   1.0
dev_swbd   1    1.0
fisher   1   1.0
fisher_sub    1   1.0
dev_fisher   1    1.0
EOF
  echo "$0 Generate train and diagnostical data"
  vocab_size=$(tail -n 1 $dir/config/words.txt | awk '{print $NF + 1}')
  mkdir -p $dir/egs
  
  # map words in train and dev texts into integers
  # read conversational counts into a dictionary
  # print out conversational word-count pairs into text files as input of get-eg. 
  rnnlm/get_conv_egs_2side.py --vocab-file=$dir/config/words.txt \
                        --unk-word="<unk>" \
                        --data-weights-file=$dir/config/data_weights.txt \
                        --output-path=$dir/egs \
                        $text_dir 


  # get dev and train_subset (contain both swbd and fisher data)
  cat $dir/egs/swbd_sub.txt $dir/egs/fisher_sub.txt > $dir/egs/train_subset.txt
  cat $dir/egs/swbd_sub.label.txt $dir/egs/fisher_sub.label.txt > $dir/egs/train_subset.label.txt

  cat $dir/egs/dev_swbd.txt $dir/egs/dev_fisher.txt > $dir/egs/dev.txt
  cat $dir/egs/dev_swbd.label.txt $dir/egs/dev_fisher.label.txt > $dir/egs/dev.label.txt

  cat $dir/egs/swbd.txt $dir/egs/fisher.txt > $dir/egs/train.txt
  cat $dir/egs/swbd.label.txt $dir/egs/fisher.label.txt > $dir/egs/train.label.txt
  
  cat >$dir/config/xconfig <<EOF
input dim=$vocab_size name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0)
relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0)
output-layer name=output dim=$vocab_size
EOF
fi

if [ $stage -le 2 ]; then
  echo "$0: initializing neural net"
  mkdir -p $dir/config/nnet

  steps/nnet3/xconfig_to_configs.py --xconfig-file=$dir/config/xconfig \
    --config-dir=$dir/config/nnet

  # initialize the neural net.
  nnet3-init $dir/config/nnet/ref.config $dir/0.raw
fi

if [ $stage -le 3 ]; then
  rnnlm/train_dnn.sh --num-jobs-initial 1 --num-jobs-final 1 \
                     --stage $train_stage \
                     --num-epochs $num_epochs \
                     --initial_effective_lrate $lr \
                     --final-effective-lrate $flr \
                     --minibatch-size $minibatch_size \
                     --cmd "$train_cmd" \
                     $dir
fi


exit 0
