
#include "g2p.h"

extern "C" void Load(const char *model_dir);
extern "C" void Call(const char *input, void (*callback)(const char*));

static std::shared_ptr<G2P> model;
typedef void Callback(const char *);

void Load(const char *model_dir){
    std::string model_dir_str(model_dir);
    model = std::make_shared<G2P>(model_dir_str);
    // std::cout << "load model" << std::endl;
}

void Call(const char *input, void (*callback)(const char*)){
    std::string input_str(input);
    // void (*callback_func)(const char*) = reinterpret_cast<void (*)(const char*)>(callback);
    auto func = reinterpret_cast<Callback *>(callback);
    std::string result = model->call(input_str);
    func(result.c_str());
}
