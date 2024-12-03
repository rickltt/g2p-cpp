#include <iostream>
#include <vector>
#include <string>
#include <dlfcn.h>  // 包含dlopen、dlsym、dlclose等函数

void Callback(const char* res){
    std::cout << "callback: " << res << std::endl;
}

int main() {
    // 加载共享库
    // 使用 dlopen 函数加载共享库文件。这个函数返回一个句柄，后续的函数调用会使用这个句柄来访问库中的符号。
    void *handle = dlopen("./libg2p.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    // 清除之前的错误
    dlerror(); 

    // 查找函数符号
    // 一旦共享库加载成功，可以使用 dlsym 来查找库中某个符号（例如函数或变量）。它返回该符号的指针。
    typedef void (*LoadFunc)(const char*);
    LoadFunc load = (LoadFunc) dlsym(handle, "Load");

    const char* model_dir = "../model";
    // 加载模型
    load(model_dir);


    typedef void (*CallFunc)(const char*, void (*)(const char*));
    CallFunc call = (CallFunc) dlsym(handle, "Call");

    std::vector<std::string> texts={
        "I have two hundred fifty dollars in my pocket.", // number -> spell-out
        "popular pets, e.g. cats and dogs", // e.g. -> for example
        "I refuse to collect the refuse around here.", // homograph
        "I'm an activationist." // newly coined word
    };
    for(const auto &text: texts){
        call(text.c_str(), &Callback);

    }
    // 关闭
    dlclose(handle);

    return 0;
}
