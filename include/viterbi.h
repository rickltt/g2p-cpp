#ifndef VITERBI_H
#define VITERBI_H

#include "tokenize.h"

void initialization(std::vector<std::string> words, Dataset &dataset);
void forward_pass(std::vector<std::string> words, Dataset &dataset);
void backward_pass(Dataset &dataset);
void print_result(std::vector<std::string> words, Dataset dataset);

#endif