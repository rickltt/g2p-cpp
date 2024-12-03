#include "data.h"

void Dataset::load_dataset()
{
    std::ifstream dataset;
    dataset.open(path);
    std::string s;

    // adding start tag to represent start of text
    pairs.push_back(std::make_pair("--s--", "--s--"));

    if (dataset.is_open())
    {
        std::cout << "Processing Dataset..." << std::endl;
        while (getline(dataset, s))
        {
            // convert vocabulary to lowercase
            for (char &c : s) 
            {
                c = tolower(c);
            }
            std::pair<std::string, std::string> p = process_line(s);
            pairs.push_back(p);
        }
        std::cout << "Processing Dataset Complete" << std::endl;
    }
    else
    {
        std::cout << "Unable to open file";
        exit(1);
    }

    dataset.close();
}

std::pair<std::string, std::string> Dataset::process_line(std::string line)
{
    if (line.length() == 0)
    {
        // adding start tag to represent start of text
        return std::make_pair("--s--", "--s--");
    }
    std::string word1, word2;
    int i = 0;
    while (line[i] != '\t')
    {
        word1 += line[i];
        i++;
    }
    i++;
    while (i != line.length())
    {
        word2 += line[i];
        i++;
    }
    return make_pair(word1, word2);
}

void Dataset::create_vocabulary()
{
    std::unordered_map <std::string, int> vocab_freq;

    for (auto p : pairs)
    {
        if (p.first == "--s--")
        {
            // do nothing for start tag
            continue;
        }
        vocab_freq[p.first]++;
        POS.insert(p.second);
    }

    for (auto p : pairs)
    {
        if (p.first == "--s--")
        {
            // do nothing for start tag
            continue;
        }
        else if (vocab_freq[p.first] > 1)
        {
            vocab.insert(p.first);
        }
        else
        {
            p.first = "UNK";
        }
        vocab_freq[p.first]++;
    }

    vocab.insert("UNK");

    std::cout << "There are " << vocab.size() << " Unique words in dataset" << std::endl;
    std::cout << "There are " << POS.size() << " Unique POS tags in dataset" << std::endl;
}


void Dataset::count_frequencies()
{
    bool isStart = false;
    for (int i = 0; i < pairs.size()-1; i++)
    {
        if (pairs[i].first == "--s--")
        {
            isStart = true;
            continue;
        }

        if (i < pairs.size()-2 && pairs[i+1].first!= "--s--")
        {
            transition_freq[make_pair(pairs[i].second, pairs[i+1].second)] += 1;
        }

        emission_freq[make_pair(pairs[i].second, pairs[i].first)] += 1;
        tag_freq[pairs[i].second] += 1;

        // increment frequencies for prior tag
        if (isStart)
        {
            prior_freq[make_pair(pairs[i].second, pairs[i].first)] += 1;
            prior_tag_freq[pairs[i].second] += 1;
            isStart = false;
        }
    }
}

void Dataset::calculate_probs()
{
    for (auto &entry : transition_freq)
    {
        transition_probs[entry.first] = log(entry.second) - log(tag_freq[entry.first.first]);
    }

    for (auto &entry : emission_freq)
    {
        emission_probs[entry.first] = log(entry.second) - log(tag_freq[entry.first.first]);
    }

    for (auto &entry : prior_freq)
    {
        prior_probs[entry.first] = log(entry.second) - log(prior_tag_freq[entry.first.first]);
    }
}