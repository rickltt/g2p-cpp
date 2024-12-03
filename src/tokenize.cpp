#include "data.h"

std::vector<std::string> tokenize(const std::string& s)
{
    std::vector<std::string> words;
    std::string temp = "";
    
    for (size_t i = 0; i < s.size(); ++i) 
    {
        char c = s[i];
        
        // Check if the character is a space or a punctuation
        if (std::isspace(c) || std::ispunct(c)) 
        {
            // If there's a non-empty word, push it to the result
            if (!temp.empty()) 
            {
                words.push_back(temp);
                temp.clear();
            }

            // If the character is a punctuation mark, treat it as a separate word
            if (std::ispunct(c)) 
            {
                words.push_back(std::string(1, c));
            }
        }
        else 
        {
            // If it's not a space or punctuation, add the character to the current word
            temp += c;
        }
    }

    // Push any remaining word after the loop
    if (!temp.empty()) 
    {
        words.push_back(temp);
    }

    return words;
}

std::vector<std::string> preprocess(std::vector<std::string> words, Dataset dataset)
{
    std::vector<std::string> processed_words;
    for (std::string word : words)
    {
        for (char &c : word) 
        {
            c = tolower(c);
        }

        if (dataset.vocab.find(word) != dataset.vocab.end())
        {
            processed_words.push_back(word);
        }
        else
        {
            processed_words.push_back("UNK");
        }
    }

    return processed_words;
}