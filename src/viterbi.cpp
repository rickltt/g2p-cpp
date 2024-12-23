#include "viterbi.h"

void initialization(std::vector <std::string> words, Dataset &dataset)
{
    double epsilon = std::numeric_limits<double>::min();
    int i = 0;

    for (auto p: dataset.POS)
    {
        dataset.tag_to_idx[p] = i;
        dataset.idx_to_tag[i] = p;
        i++;
    }

    dataset.dp.resize(dataset.tag_to_idx.size(), std::vector<double>(words.size()));
    dataset.tags.resize(dataset.tag_to_idx.size(), std::vector<int>(words.size()));

    for (int i = 0; i < dataset.POS.size(); i++)
    {
        std::string word = words[0];
        if (dataset.prior_probs.find(make_pair(dataset.idx_to_tag[i], word)) != dataset.prior_probs.end())
        {
            dataset.dp[i][0] = dataset.prior_probs[make_pair(dataset.idx_to_tag[i], word)];
        }
        else
        {
            dataset.dp[i][0] = log(epsilon);
        }

        dataset.tags[i][0] = 0;
    }
}

void forward_pass(std::vector <std::string> words, Dataset &dataset)
{
    double epsilon = std::numeric_limits<double>::min();
    
    // i represents column
    for (int i = 1; i < dataset.dp[0].size(); i++)
    {
        // j represents row
        for (int j = 0; j < dataset.dp.size(); j++)
        {
            double max = std::numeric_limits<double>::lowest();
            int max_idx = -1;
            // k represents row of previous column
            for (int k = 0; k < dataset.dp.size(); k++)
            {
                double curr_transition_prob, curr_emission_prob;

                if (dataset.transition_probs.find(make_pair(dataset.idx_to_tag[j], dataset.idx_to_tag[k])) != dataset.transition_probs.end())
                {
                    curr_transition_prob = dataset.transition_probs[make_pair(dataset.idx_to_tag[j], dataset.idx_to_tag[k])];
                }
                else
                {
                    curr_transition_prob = log(epsilon);
                }

                if (dataset.emission_probs.find(make_pair(dataset.idx_to_tag[j], words[i])) != dataset.emission_probs.end())
                {
                    curr_emission_prob = dataset.emission_probs[make_pair(dataset.idx_to_tag[j], words[i])];
                }
                else
                {
                    curr_emission_prob = log(epsilon);
                }

                double temp = dataset.dp[k][i-1] + curr_transition_prob + curr_emission_prob;

                if (temp > max)
                {
                    max = temp;
                    max_idx = k;
                }
            }

            dataset.dp[j][i] = max;
            dataset.tags[j][i] = max_idx;
        }
    }
}

void backward_pass(Dataset &dataset)
{
    std::vector <int> answer_idx;

    //last column
    int c = dataset.tags[0].size() - 1;
    double max = std::numeric_limits<double>::lowest();
    int max_idx = 0;
    // i represents rows
    for (int i = 0; i < dataset.tags.size(); i++)
    {
        if (dataset.dp[i][c] > max)
        {
            max = dataset.dp[i][c];
            max_idx = i;
        }
    }
    answer_idx.push_back(max_idx);

    while(c > 0)
    {
        max_idx = dataset.tags[max_idx][c];
        answer_idx.push_back(max_idx);
        c--;
    }

    reverse(answer_idx.begin(), answer_idx.end());

    for (int i = 0; i < answer_idx.size(); i++)
    {
        dataset.answer.push_back(dataset.idx_to_tag[answer_idx[i]]);
    }
}


void print_result(std::vector<std::string> words, Dataset dataset)
{
    std::cout << std::endl << "/////////////////" << std::endl;
    for (int i = 0; i < dataset.answer.size(); i++)
    {
        std::cout << words[i] << " -> " << dataset.tagDictionary[dataset.answer[i]] << std::endl;
    }
    std::cout << "/////////////////" << std::endl;
}