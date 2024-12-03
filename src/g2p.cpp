#include "g2p.h"

G2P::G2P(const std::string &model_path)
{
    // Initialize graphemes
    graphemes = {"<pad>", "<unk>", "</s>"};
    for (char c = 'a'; c <= 'z'; ++c)
    {
        graphemes.push_back(std::string(1, c));
    }

    // for(auto g: graphemes){
    //     LOG(INFO) << g;
    // }
    // Initialize phonemes
    phonemes = {"<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
                "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1",
                "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1",
                "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH",
                "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"};

    // Create g2idx and p2idx maps
    for (size_t i = 0; i < graphemes.size(); ++i)
    {
        g2idx[graphemes[i]] = i;
        idx2g[i] = graphemes[i];
    }

    for (size_t i = 0; i < phonemes.size(); ++i)
    {
        p2idx[phonemes[i]] = i;
        idx2p[i] = phonemes[i];
    }

    cmudict_path = model_path + "/" + "cmudict.dict";
    std::ifstream infile(cmudict_path);
    if (!infile.is_open()) {
        std::cerr << "无法打开文件!" << std::endl;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) {
            continue;
        }
        // 使用 stringstream 来解析每一行
        std::stringstream ss(line);
        std::string word;
        std::string pronunciation;

        // 读取单词
        ss >> word;

        // 读取音标，所有音标都是一个空格分隔的部分，拼接为一个完整的字符串
        std::getline(ss, pronunciation);
        cmudict[word] = pronunciation.substr(1);
    }

    // std::cout << "读取的字典内容：" << std::endl;
    // for (const auto& entry : cmudict) {
    //     std::cout << "单词:" << entry.first << ", 发音:" << entry.second << std::endl;
    // }
    LOG(INFO) << "Load cmudict Finished!";
    construct_homograph_dictionary(model_path);
    load_dataset(model_path);
    load_model(model_path);
}

G2P::~G2P()
{

}

void G2P::load_dataset(const std::string &model_path)
{

    dataset = Dataset(model_path + "/" + "dataset" + ".pos");

    dataset.load_dataset();
    dataset.create_vocabulary();
    dataset.count_frequencies();
    dataset.calculate_probs();

    LOG(INFO) << "Load Dataset Finished!";

}
void G2P::load_model(const std::string &model_path)
{

    enc_emb = nc::load<float>(model_path + "/" + "enc_emb" + ".bin");
    enc_emb.reshape(29, 256);
    enc_w_ih = nc::load<float>(model_path + "/" + "enc_w_ih" + ".bin");
    enc_w_ih.reshape(768, 256);
    enc_w_hh = nc::load<float>(model_path + "/" + "enc_w_hh" + ".bin");
    enc_w_hh.reshape(768, 256);
    enc_b_ih = nc::load<float>(model_path + "/" + "enc_b_ih" + ".bin");
    enc_b_ih.reshape(768);
    enc_b_hh = nc::load<float>(model_path + "/" + "enc_b_hh" + ".bin");
    enc_b_hh.reshape(768);

    dec_emb = nc::load<float>(model_path + "/" + "dec_emb" + ".bin");
    dec_emb.reshape(74, 256);
    dec_w_ih = nc::load<float>(model_path + "/" + "dec_w_ih" + ".bin");
    dec_w_ih.reshape(768, 256);
    dec_w_hh = nc::load<float>(model_path + "/" + "dec_w_hh" + ".bin");
    dec_w_hh.reshape(768, 256);
    dec_b_ih = nc::load<float>(model_path + "/" + "dec_b_ih" + ".bin");
    dec_b_ih.reshape(768);
    dec_b_hh = nc::load<float>(model_path + "/" + "dec_b_hh" + ".bin");
    dec_b_hh.reshape(768);

    fc_w = nc::load<float>(model_path + "/" + "fc_w" + ".bin");
    fc_w.reshape(74, 256);
    fc_b = nc::load<float>(model_path + "/" + "fc_b" + ".bin");
    fc_b.reshape(74);

    LOG(INFO) << "Load Model Finished!";
}

void G2P::construct_homograph_dictionary(const std::string& model_path) {
    
    std::string filename = model_path  + "/" + "homographs" + ".en";
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
    
    std::string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip empty lines or comments
        
        std::stringstream ss(line);
        std::string headword, pron1, pron2, pos1;

        // Read parts of the line separated by "|"
        getline(ss, headword, '|');
        getline(ss, pron1, '|');
        getline(ss, pron2, '|');
        getline(ss, pos1, '|');
        
        // Lowercase the headword
        transform(headword.begin(), headword.end(), headword.begin(), ::tolower);
        
        // Split pronunciations into vectors
        // std::stringstream pron1_stream(pron1), pron2_stream(pron2);
        // std::string token;
        // std::vector<std::string> pron1_vec, pron2_vec;
        
        // while (getline(pron1_stream, token, ' ')) {
        //     pron1_vec.push_back(token);
        // }
        // while (getline(pron2_stream, token, ' ')) {
        //     pron2_vec.push_back(token);
        // }
        
        // Store the features in the map
        homograph2features[headword] = HomographFeatures{pron1, pron2, pos1};
    }

    file.close();
    LOG(INFO) << "Load homograph2features Finished!";
}

std::unordered_map<std::string, std::string> G2P::pos_tag(const std::string &input_text, Dataset dataset){
    std::vector<std::string> tokens = tokenize(input_text);

    tokens = preprocess(tokens, dataset);

    initialization(tokens, dataset);
    forward_pass(tokens, dataset);
    backward_pass(dataset);

    std::unordered_map<std::string, std::string> result;

    for (int i = 0; i < dataset.answer.size(); i++)
    {
        std::cout << tokens[i] << " -> " << dataset.answer[i] << std::endl;
        result[tokens[i]] = dataset.answer[i];
    }

    return result;
}
std::string G2P::processText(std::string text)
{
    // 转换为小写
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);

    // 使用正则表达式移除非字母、空格和特定标点字符
    std::string _text = std::regex_replace(text, std::regex("[^ a-z'.,?!\\-]"), "");

    // 替换 "i.e." 为 "that is"
    size_t pos = 0;
    while ((pos = _text.find("i.e.", pos)) != std::string::npos)
    {
        _text.replace(pos, 4, "that is");
        pos += 7; 
    }

    // 替换 "e.g." 为 "for example"
    pos = 0;
    while ((pos = _text.find("e.g.", pos)) != std::string::npos)
    {
        _text.replace(pos, 4, "for example");
        pos += 12; 
    }

    return _text;
}

std::string G2P::call(const std::string &text)
{
    std::string clean_text = processText(text);
    std::vector<std::string> prons;

    // boost::tokenizer<> tokens(clean_text);
    auto tokens = tokenize(clean_text);
    auto pos_result = pos_tag(clean_text, dataset);

    for (auto token: tokens){
        std::cout << token << " ";
    }
    std::cout << "\n";

    for (auto pos: pos_result){
        std::cout << pos.first << " ";
    }
    std::cout << "\n";

    // assert(tokens.size() == pos_result.size());

    for (auto token: tokens)
    {
        auto pos = pos_result[token];

        LOG(INFO) << token;
        std::string pron;
        if (!std::regex_search(token, std::regex("[a-z]")))
        {
            // pron = token;
            continue;
        }

        if (homograph2features.find(token) != homograph2features.end()){
            auto feature = homograph2features[token];
            transform(pos.begin(), pos.end(), pos.begin(), ::toupper);
            if (pos == feature.part_of_speech){
                pron = feature.pronunciation1;
            }else{
                pron = feature.pronunciation2;
            }
        }
        else if (cmudict.find(token) != cmudict.end())
        {
            pron = cmudict[token];
            LOG(INFO) << token << ": " << pron;
        }
        else{
            pron = predict_oov(token);
        }
        prons.push_back(pron);
    }

    std::string result;
    for (auto p : prons)
    {
        if (result.empty()){
            result += p;
        }else{
            result += " " + p;
        }
    }
    return result;
}

nc::NdArray<float> G2P::encode(std::string str){

    LOG(INFO) << "predict oov: " << str;
    std::vector<std::string> chars;
    for (char c : str) {
        chars.push_back(std::string(1, c));
        LOG(INFO) << std::string(1, c);
    }
    chars.push_back("</s>");

    // 将字符转换为索引（使用 g2idx 获取字符的索引，未找到则使用 <unk>）
    std::vector<int> x;
    for (const std::string& char_str : chars) {
        auto it = g2idx.find(char_str);
        if (it != g2idx.end()) {
            x.push_back(it->second);
        } else {
            x.push_back(g2idx["<unk>"]);
        }
        std::cout << it->second << " ";
    }
    std::cout << std::endl;
    // LOG(INFO) << enc_emb.shape().rows << " " << enc_emb.shape().cols;
    // LOG(INFO) << enc_emb.size();
    std::vector<nc::NdArray<float>> embs;
    for(auto id: x){
        // LOG(INFO) << id;
        nc::NdArray<float> id_vec = enc_emb(id, enc_emb.cSlice());
        // LOG(INFO) << id_vec.size();
        embs.push_back(id_vec);
        // id_vec.print();
    }
    nc::NdArray<float> all_embs = nc::vstack(embs);
    LOG(INFO) << all_embs.shape().rows << " " << all_embs.shape().cols;
    return all_embs;
}


nc::NdArray<float> G2P::gru(nc::NdArray<float> x, size_t steps, nc::NdArray<float> w_ih, nc::NdArray<float> w_hh, nc::NdArray<float>b_ih, nc::NdArray<float> b_hh, nc::NdArray<float> h0){
    auto h = h0;
    nc::NdArray<float> h_step;
    std::vector<nc::NdArray<float>> outputs;
    // auto outputs = nc::zeros<float>(h0.shape().rows, h0.shape().cols);
    for(size_t i=0; i<steps; ++i){
        h_step = x(i, x.cSlice());
        // LOG(INFO) << "h_step shape: " << h_step.shape().rows << " " << h_step.shape().cols;
        h = gru_cell(h_step, h, w_ih, w_hh, b_ih, b_hh);
        outputs.push_back(h);
        // break;
        // h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
    }
    auto gru_outputs = nc::vstack(outputs);
    // gru_outputs.print();
    LOG(INFO) << "gru_outputs shape: " << gru_outputs.shape().rows << " " << gru_outputs.shape().cols;
    return gru_outputs;
}


nc::NdArray<float> G2P::gru_cell(nc::NdArray<float> x, nc::NdArray<float> h, nc::NdArray<float> w_ih, nc::NdArray<float> w_hh, nc::NdArray<float> b_ih, nc::NdArray<float> b_hh){
    // rzn_ih = np.matmul(x, w_ih.T) + b_ih
    // rzn_hh = np.matmul(h, w_hh.T) + b_hh
    auto rzn_ih = nc::dot<float>(x, w_ih.transpose()) + b_ih; // (1, 768)
    auto rzn_hh = nc::dot<float>(h, w_hh.transpose()) + b_hh; // (1, 768)

    // LOG(INFO) << "rzn_ih shape: " << rzn_ih.shape().rows << " " << rzn_ih.shape().cols;
    // LOG(INFO) << "rzn_hh shape: " << rzn_hh.shape().rows << " " << rzn_hh.shape().cols;

    // rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
    auto rz_ih = rzn_ih(rzn_ih.rSlice(), nc::Slice(0, 512));
    auto n_ih = rzn_ih(rzn_ih.rSlice(), nc::Slice(512, 768));

    //  rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]
    auto rz_hh = rzn_hh(rzn_hh.rSlice(), nc::Slice(0, 512));
    auto n_hh = rzn_hh(rzn_hh.rSlice(), nc::Slice(512, 768));

    // LOG(INFO) << "rz_hh shape: " << rz_hh.shape().rows << " " << rz_hh.shape().cols;
    // LOG(INFO) << "n_hh shape: " << n_hh.shape().rows << " " << rz_hh.shape().cols;

    // rz = self.sigmoid(rz_ih + rz_hh)
    auto rz = 1.0f /( 1.0f + nc::exp(-1.0f * (rz_ih + rz_hh)));
    // r, z = np.split(rz, 2, -1)

    auto r = rz(rz.rSlice(), nc::Slice(0, 256));
    auto z = rz(rz.rSlice(), nc::Slice(256, 512));
    // LOG(INFO) << "r shape: " << r.shape().rows << " " << r.shape().cols;
    // LOG(INFO) << "z shape: " << z.shape().rows << " " << z.shape().cols;

    // n = np.tanh(n_ih + r * n_hh)
    auto n = nc::tanh(n_ih + r * n_hh);
    // LOG(INFO) << "n shape: " << n.shape().rows << " " << n.shape().cols;
    // h = (1 - z) * n + z * h
    auto gru_cell_outputs = (1.0f - z) * n + z * h;
    // LOG(INFO) << "gru_cell_outputs shape: " << gru_cell_outputs.shape().rows << " " << gru_cell_outputs.shape().cols;
    return gru_cell_outputs;
}
std::string G2P::predict_oov(std::string str){

    auto enc_embs = encode(str);

    auto enc_gru = gru(enc_embs, str.length()+1, enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh, nc::zeros<float>(1, enc_w_hh.shape().cols));

    auto enc_last_hidden = enc_gru(enc_gru.shape().rows-1, enc_gru.cSlice());
    auto h = enc_last_hidden;
    LOG(INFO) << "last_hidden shape: " << enc_last_hidden.shape().rows << " " << enc_last_hidden.shape().cols;

    auto dec_embs = dec_emb(g2idx["</s>"], dec_emb.cSlice());

    std::vector<int> preds;
    for(size_t i=0; i<20; ++i){
        h = gru_cell(dec_embs, h, dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh);
        // LOG(INFO) << "h shape: " << h.shape().rows << " " << h.shape().cols;

        auto logits = nc::dot<float>(h, fc_w.transpose()) + fc_b; 
        // LOG(INFO) << "logits shape: " << logits.shape().rows << " " << logits.shape().cols;

        auto pred = nc::argmax(logits)(0,0);
        LOG(INFO) << "pred: " << pred;
        if (pred == p2idx["</s>"]){
            break;
        }
        preds.push_back(pred);
        dec_embs = dec_emb(pred, dec_emb.cSlice());
    }

    std::string result;
    for(auto pred: preds){
        std::cout << pred << ":" << idx2p[pred] << " ";
        if (result.empty()){
            result += idx2p[pred];
        }else{
            result += " " + idx2p[pred];
        }
    }
    std::cout << "\n";

    return result;
}