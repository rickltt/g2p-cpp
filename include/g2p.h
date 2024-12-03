#ifndef G2P_H
#define G2P_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <string>
#include <regex>
#include <cassert>
#include <glog/logging.h>
// #include <boost/tokenizer.hpp>
#include "NumCpp.hpp"
#include "tokenize.h"
#include "data.h"
#include "viterbi.h"

struct HomographFeatures {
    std::string pronunciation1;
    std::string pronunciation2;
    std::string part_of_speech;
};

class G2P {

public:
    G2P() {};
    G2P(const std::string &model_path);
    ~G2P();
    void load_dataset(const std::string &model_path);
    void load_model(const std::string &model_path);
    void construct_homograph_dictionary(const std::string& model_path);

    std::string processText(std::string text);
    std::string call(const std::string &input_text);
    std::string predict_oov(std::string str);

    std::unordered_map<std::string, std::string> pos_tag(const std::string &input_text, Dataset dataset);

    nc::NdArray<float> encode(std::string str);
    nc::NdArray<float> gru(nc::NdArray<float> x, size_t steps, nc::NdArray<float> w_ih, nc::NdArray<float> w_hh, nc::NdArray<float>b_ih, nc::NdArray<float> b_hh, nc::NdArray<float> h0);
    nc::NdArray<float> gru_cell(nc::NdArray<float> x, nc::NdArray<float> h, nc::NdArray<float> w_ih, nc::NdArray<float> w_hh, nc::NdArray<float>b_ih, nc::NdArray<float> b_hh);


private:
    Dataset dataset;
    std::unordered_map<std::string, HomographFeatures> homograph2features;

    std::vector<std::string> graphemes;
    std::vector<std::string> phonemes;

    std::vector<std::string> model_name;

    std::unordered_map<std::string, size_t> g2idx;
    std::unordered_map<size_t, std::string> idx2g;
    std::unordered_map<std::string, size_t> p2idx;
    std::unordered_map<size_t, std::string> idx2p;

    std::unordered_map<std::string, std::string> cmudict;
    std::string cmudict_path;

    nc::NdArray<float> enc_emb;
    nc::NdArray<float> enc_w_ih;
    nc::NdArray<float> enc_w_hh;
    nc::NdArray<float> enc_b_ih;
    nc::NdArray<float> enc_b_hh;

    nc::NdArray<float> dec_emb;
    nc::NdArray<float> dec_w_ih;
    nc::NdArray<float> dec_w_hh;
    nc::NdArray<float> dec_b_ih;
    nc::NdArray<float> dec_b_hh;

    nc::NdArray<float> fc_w;
    nc::NdArray<float> fc_b;
    

};

#endif

