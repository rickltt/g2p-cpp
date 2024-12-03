#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <vector>
#include <fstream>
#include <set>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <string>
#include <limits>
#include <math.h>


struct pair_hash {
    template <typename T1, typename T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
    }
};

class Dataset
{
public:
    std::string path;
    std::vector<std::pair<std::string, std::string>> pairs;
    std::set<std::string> vocab;
    std::set<std::string> POS;
    std::unordered_map<std::string, int> tag_freq;
    std::unordered_map <std::pair<std::string, std::string>, int, pair_hash> transition_freq;
    std::unordered_map <std::pair<std::string, std::string>, int, pair_hash> emission_freq;
    std::unordered_map <std::pair<std::string, std::string>, int, pair_hash> prior_freq;
    std::unordered_map <std::string, int> prior_tag_freq;
    std::unordered_map <std::pair<std::string, std::string>, double, pair_hash> transition_probs;
    std::unordered_map <std::pair<std::string, std::string>, double, pair_hash> emission_probs;
    std::unordered_map <std::pair<std::string, std::string>, double, pair_hash> prior_probs;
    std::unordered_map <std::string, int> tag_to_idx;
    std::unordered_map <int, std::string> idx_to_tag;
    std::vector<std::vector<double>> dp;
    std::vector<std::vector<int>> tags;
    std::vector<std::string> answer;

    std::unordered_map<std::string, std::string> tagDictionary =
    {
        {"cc", "conjunction, coordinating"},
        {"cd", "cardinal number"},
        {"dt", "determiner"},
        {"ex", "existential there"},
        {"fw", "foreign word"},
        {"in", "conjunction, subordinating or preposition"},
        {"jj", "adjective"},
        {"jjr", "adjective, comparative"},
        {"jjs", "adjective, superlative"},
        {"ls", "list item marker"},
        {"md", "verb, modal auxillary"},
        {"nn", "noun, singular or mass"},
        {"nns", "noun, plural"},
        {"nnp", "noun, proper singular"},
        {"nnps", "noun, proper plural"},
        {"pdt", "predeterminer"},
        {"pos", "possessive ending"},
        {"prp", "pronoun, personal"},
        {"prp$", "pronoun, possessive"},
        {"rb", "adverb"},
        {"rbr", "adverb, comparative"},
        {"rbs", "adverb, superlative"},
        {"rp", "adverb, particle"},
        {"sym", "symbol"},
        {"to", "infinitival to"},
        {"uh", "interjection"},
        {"vb", "verb, base form"},
        {"vbz", "verb, 3rd person singular present"},
        {"vbp", "verb, non-3rd person singular present"},
        {"vbd", "verb, past tense"},
        {"vbn", "verb, past participle"},
        {"vbg", "verb, gerund or present participle"},
        {"wdt", "wh-determiner"},
        {"wp", "wh-pronoun, personal"},
        {"wp$", "wh-pronoun, possessive"},
        {"wrb", "wh-adverb"},
        {".", "punctuation mark, sentence closer"},
        {",", "punctuation mark, comma"},
        {":", "punctuation mark, colon"},
        {"(", "contextual separator, left paren"},
        {")", "contextual separator, right paren"}};
        
    Dataset()
    {
    }
    Dataset(std::string s)
    {
        path = s;
    }
    void load_dataset();
    std::pair<std::string, std::string> process_line(std::string line);
    void create_vocabulary();
    void count_frequencies();
    void calculate_probs();
};

#endif