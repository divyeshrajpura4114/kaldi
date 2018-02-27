// nnet3bin/nnet3-get-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar
//                2018  Ke Li

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <sstream>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {


static bool ProcessFile(const GeneralMatrix &feats,
                        const Posterior &pdf_post,
                        const std::string key,
                        bool compress,
                        int32 num_words,
                        NnetExampleWriter *example_writer) {
    NnetExample eg;
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", 0, feats));

    eg.io.push_back(NnetIo("output", num_words, 0, pdf_post));
    
    if (compress)
      eg.Compress();

    example_writer->Write(key, eg);

    return true;
}

} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3 neural network training.\n"
        "Essentially this is a format change from features and posteriors\n"
        "into a special frame-by-frame format.  This program handles the\n"
        "common case where you have some input features, possibly some\n"
        "iVectors, and one set of labels.  If people in future want to\n"
        "do different things they may have to extend this program or create\n"
        "different versions of it for different tasks (the egs format is quite\n"
        "general)\n"
        "\n"
        "Usage:  nnet3-get-egs [options] <features-rspecifier> "
        "<pdf-post-rspecifier> <egs-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet3-get-egs --num-pdfs=2658 --left-context=12 --right-context=9 --num-frames=8 \"$feats\"\\\n"
        "\"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:- | ali-to-post ark:- ark:- |\" \\\n"
        "   ark:- \n"
        "See also: nnet3-chain-get-egs, nnet3-get-egs-simple\n";


    bool compress = true;
    int32 num_words = -1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    ParseOptions po(usage);

    po.Register("compress", &compress, "If true, write egs with input features "
                "in compressed format (recommended).  This is "
                "only relevant if the features being read are un-compressed; "
                "if already compressed, we keep we same compressed format when "
                "dumping egs.");
    po.Register("num-words", &num_words, "Number of output words.");
    eg_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (num_words <= 0)
      KALDI_ERR << "--num-words options is required.";

    eg_config.ComputeDerived();

    std::string feature_rspecifier = po.GetArg(1),
        pdf_post_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    // Read input data into GeneralMatrix; Input data are sequences of word-count
    // pairs representing training conversations
    PosteriorHolder input;
    std::ifstream feat_in(feature_rspecifier);
    Posterior feats;
    if (input.Read(feat_in)) {
       feats = input.Value();
    }

    // Read output data into Posterior; Output data are sequences of word-count
    // pairs representing "test" conversations
    // Method 1: read posteriors directly 
    PosteriorHolder holder;
    std::ifstream post_in(pdf_post_rspecifier);
    Posterior pdf_posts;
    if (holder.Read(post_in))
      pdf_posts = holder.Value();

    // wrap input type Posterior into GeneralMatrix
    // std::vector<std::vector<int32, float> >::const_iterator iter_input = feats.begin();
    NnetExampleWriter example_writer(examples_wspecifier);

    int32 num_err = 0;
    int32 feat_range = feats.size();
    for(int32 index = 0; index < feat_range; index++) {
      std::string key = std::to_string(index);
      std::vector<std::vector<std::pair<int32, BaseFloat> > > feat_;
      feat_.push_back(feats[index]);
      // const Posterior &feat_p = feat_;
      const SparseMatrix<BaseFloat> feat_s(num_words, feat_);
      const GeneralMatrix feat(feat_s);
      std::vector<std::vector<std::pair<int32, BaseFloat> > > pdf_post_;
      pdf_post_.push_back(pdf_posts[index]);
      const Posterior &pdf_post = pdf_post_;
      
      if(!ProcessFile(feat, pdf_post, key, compress, num_words, &example_writer))
        num_err++;
    }
     
    if (num_err > 0)
      KALDI_WARN << num_err << " conversations had errors and could "
          "not be processed.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
