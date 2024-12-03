#ifndef TOKENIZE_H
#define TOKENIZE_H

#include "data.h"

std::vector<std::string> tokenize(const std::string& s);
std::vector<std::string> preprocess(std::vector<std::string> words, Dataset dataset);

#endif