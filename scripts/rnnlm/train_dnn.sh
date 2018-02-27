#!/usr/bin/env bash

# This script does the generation of egs and DNN training.

#num-jobs-initial, num-jobs-final, max-change, embedding-max-change [initial,final?],
#num-samples, minibatch-size, chunk-length, [and the same for dev data]...
#initial-effective-learning-rate, final-effective-learning-rate, ...
#embedding-learning-rate-factor, num-epochs


stage=0
num_jobs_initial=1
num_jobs_final=1
chunk_length=32
num_epochs=100  # maximum number of epochs to train.  later we
                # may find a stopping criterion.
initial_effective_lrate=0.01
final_effective_lrate=0.001
cmd=run.pl  # you might want to set this to queue.pl

nj=1 # maximum number of jobs you want to set
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of
# the samples on each iter. You could set it to 0 or to a large value for
# complete randomization, but this would both consume memory and cause spikes in
# dist I/O. Smaller is easier on disk and memory but less random. 
# It's not a huge deal though, as samples are anyway randomized right at the start.
minibatch_size=8 # This default is suitable for GPU-based training.
                   # Set it to 128 for multi-threaded CPU-based training.
max_param_change=0.5  # max param change per minibatch
# some options passed into rnnlm-get-egs, relating to sampling.
num_egs_threads=10  # number of threads used for sampling, if we're using
                    # sampling.  the actual number of threads that runs at one
                    # time, will be however many is needed to balance the
                    # sampling and the actual training, this is just the maximum
                    # possible number that are allowed to run
use_gpu=true  # use GPU for training
use_gpu_for_diagnostics=false  # set true to use GPU for compute_prob_*.log

trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM
. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <rnnlm-dir>"
  echo "Trains an DNN, assuming the things needed for training have already been"
  echo "set up."
  exit 1
fi


dir=$1


set -e
. ./path.sh


for f in $dir/config/{words,data_weights}.txt \
              $dir/0.raw \
              ; do
  [ ! -f $f ] && echo "$0: expected $f to exist" && exit 1
done

# set some variables and check more files.
# num_splits=$(cat $dir/text/info/num_splits)
# num_repeats=$(cat $dir/text/info/num_repeats)
num_splits=1
num_repeats=1
text_files=$(for n in $(seq $num_splits); do echo $dir/text/$n.txt; done)
vocab_size=$(tail -n 1 $dir/config/words.txt | awk '{print $NF + 1}')

if [ $num_jobs_initial -gt $num_splits ] || [ $num_jobs_final -gt $num_splits ]; then
  echo -n "$0: number of initial or final jobs $num_jobs_initial/$num_jobs_final"
  echo "exceeds num-splits=$num_splits; reduce number of jobs"
  exit 1
fi

num_splits_to_process=$[($num_epochs*$num_splits)/$num_repeats]
num_split_processed=0
num_iters=$[($num_splits_to_process*2)/($num_jobs_initial+$num_jobs_final)]

echo "$0: will train for $num_iters iterations"

# recording some configuration information
cat >$dir/info.txt <<EOF
num_iters=$num_iters
num_epochs=$num_epochs
num_jobs_initial=$num_jobs_initial
num_jobs_final=$num_jobs_final
max_param_change=$max_param_change
chunk_length=$chunk_length
initial_effective_lrate=$initial_effective_lrate
final_effective_lrate=$final_effective_lrate
EOF

#frames_per_eg=8
#frames_per_eg_principal=8
#samples_per_iter=400000 # this is the target number of egs in each archive of egs
                        # (prior to merging egs).  We probably should have called
                        # it egs_per_iter. This is just a guideline; it will pick
                        # a number that divides the number of samples in the    
                        # entire data.                                          
# echo "$0: working out number of frames of training data"
# num_frames=$(steps/nnet2/get_num_frames.sh $data) #TODO: data dir is ?

# num_archives=$[$num_frames/($frames_per_eg_principal*$samples_per_iter)+1]
num_archives=1 # 
# get training egs
egs_list=
for n in $(seq $num_archives); do
  egs_list="$egs_list ark:$dir/egs/egs.JOB.$n.ark"
done
echo "$0: Generating training examples on disk"
eg_srand=0 # random seed for nnet3-get-egs-adaptation and nnet3-copy-egs
# The examples will go round-robin to egs_list.
$cmd JOB=1:$nj $dir/egs/log/get_egs.JOB.log \
  nnet3-get-egs-adaptation --num-words=$vocab_size $dir/egs/train.txt $dir/egs/train.label.txt ark:- \| \
  nnet3-copy-egs --random=true --srand=\$[JOB+$eg_srand] ark:- $egs_list || exit 1;
echo "$0: Generating train and validation examples for diagnostics on disk"
$cmd $dir/egs/log/get_train_diagnositcs.log \
  nnet3-get-egs-adaptation --num-words=$vocab_size $dir/egs/train_subset.txt $dir/egs/train_subset.label.txt ark:- \| \
  nnet3-copy-egs --random=true --srand=$eg_srand ark:- ark:$dir/egs/train_diagnostic.ark || exit 1;
$cmd $dir/egs/log/get_valid_diagnositcs.log \
  nnet3-get-egs-adaptation --num-words=$vocab_size $dir/egs/dev.txt $dir/egs/dev.label.txt ark:- \| \
  nnet3-copy-egs --random=true --srand=$eg_srand ark:- ark:$dir/egs/valid_diagnostic.ark || exit 1;

x=0
num_splits_processed=0
while [ $x -lt $num_iters ]; do

  this_num_jobs=$(perl -e "print int(0.5+$num_jobs_initial+($num_jobs_final-$num_jobs_initial)*$x/$num_iters);")
  ilr=$initial_effective_lrate; flr=$final_effective_lrate; np=$num_splits_processed; nt=$num_splits_to_process;
  this_learning_rate=$(perl -e "print (($x + 1 >= $num_iters ? $flr : $ilr*exp($np*log($flr/$ilr)/$nt))*$this_num_jobs);");
    
  echo "On iteration $x, learning rate is $this_learning_rate."

  if [ $stage -le $x ]; then
    
    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet3-compute-prob "nnet3-copy $dir/$x.raw - |" \
            "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/egs/valid_diagnostic.ark ark:- |" &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet3-compute-prob "nnet3-copy $dir/$x.raw - |" \
            "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/egs/train_diagnostic.ark ark:- |" &

    if [ $x -gt 0 ]; then
      $cmd $dir/log/progress.$x.log \
        nnet3-show-progress --use-gpu=no $dir/$[$x-1].raw $dir/$x.raw '&&' \
          nnet3-info $dir/$x.raw &
    fi

    echo "Training neural net (pass $x)"

    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.

      # We can't easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      [ -f $dir/.train_error ] && rm $dir/.train_error
      for n in $(seq $this_num_jobs); do
        echo "this num jobs is $this_num_jobs"
        k=$[$num_splits_processed + $n - 1]; # k is a zero-based index that we'll derive
                                               # the other indexes from.
        # split is archive in nnet3 acoustic model training script
        split=$[($k%$num_splits)+1]; # work out the 1-based split index.
        # frame=$[(($k/$num_splits)%$frames_per_eg)]; # work out the 0-based frame
        # index; this increases more slowly than the split index because the
        # same split with different frame indexes will give similar gradients,
        # so we want to separate them in time.
        
        src_dnn="nnet3-copy --learning-rate=$this_learning_rate $dir/$x.raw -|"
        if $use_gpu; then gpu_opt="--use-gpu=yes"; queue_gpu_opt="--gpu 1";
        else gpu_opt="--use-gpu=no"; queue_gpu_opt=""; fi
        if [ $this_num_jobs -gt 1 ]; then dest_number=$[x+1].$n
        else dest_number=$[x+1]; fi

        # Run the training job or jobs.
        $cmd $queue_gpu_opt $dir/log/train.$x.$n.log \
           nnet3-train $gpu_opt \
           --max-param-change=$max_param_change "$src_dnn" \
           "ark:nnet3-copy-egs --frame="" ark:$dir/egs/egs.$n.$split.ark ark:- | nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-| nnet3-merge-egs --minibatch-size=$minibatch_size --discard-partial-minibatches=false ark:- ark:- |" \
           $dir/$dest_number.raw || touch $dir/.train_error &
      done
      wait # wait for just the training jobs.
      [ -f $dir/.train_error ] && \
        echo "$0: failure on iteration $x of training, see $dir/log/train.$x.*.log for details." && exit 1
      if [ $this_num_jobs -gt 1 ]; then
        # average the models and the embedding matrces.  Use run.pl as we don\'t
        # want this to wait on the queue (if there is a queue).
        src_models=$(for n in $(seq $this_num_jobs); do echo $dir/$[x+1].$n.raw; done)
        run.pl $dir/log/average.$[x+1].log \
          nnet3-average $src_models $dir/$[x+1].raw
      fi
    )

    num_splits_processed=$[num_splits_processed+this_num_jobs]
  fi
  x=$[x+1]
done

wait # wait for diagnostic jobs in the background.
ln -sf $num_iters.raw $dir/final.raw

exit 1;

if [ $stage -le $num_iters ]; then
  # link the best model we encountered during training (based on
  # dev-set probability) as the final model.
  best_iter=$(rnnlm/get_best_model.py $dir)
  echo "$0: best iteration (out of $num_iters) was $best_iter, linking it to final iteration."
  train_best_log=$dir/log/train.$best_iter.1.log
  obj_train=`grep 'Overall average' $train_best_log | awk '{printf("%.1f",$8)}'`
  dev_best_log=$dir/log/compute_prob_valid.$best_iter.log
  obj_dev=`grep 'Overall log-likelihood' $dev_best_log | awk '{printf("%.1f", $6)}'`
  echo "$0: train/dev log-likelihood was $obj_train / $obj_dev."
  ln -sf $best_iter.raw $dir/final.raw
fi

# Now get some diagnostics about the evolution of the objective function.
if [ $stage -le $[num_iters+1] ]; then
  (
    logs=$(for iter in $(seq 0 $[$num_iters-1]); do echo -n $dir/log/train.$iter.1.log ''; done)
    # in the non-sampling case the exact objf is printed and we plot that
    # in the sampling case we print the approximated objf for training.
    grep 'Overall average' $logs | awk 'BEGIN{printf("Train objf: ")} /exact/{printf("%.2f ", $NF);next} {printf("%.2f ", $8)} END{print "";}'
    logs=$(for iter in $(seq 0 $[$num_iters-1]); do echo -n $dir/log/compute_prob_valid.$iter.log ''; done)
    grep 'Overall log-likelihood' $logs | awk 'BEGIN{printf("Dev objf:   ")} {printf("%.2f ", $NF)} END{print "";}'
  ) > $dir/report.txt
  cat $dir/report.txt
fi
