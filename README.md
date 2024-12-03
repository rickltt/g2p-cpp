# G2P (Grapheme-to-Phoneme) 

G2P C++实现，使用[NumCpp](https://github.com/dpilger26/NumCpp)修改[g2pE: A Simple Python Module for English Grapheme To Phoneme Conversion](https://github.com/Kyubyong/g2p).

## 转换模型权重文件为.bin

    import numpy as np
    import os

    model = np.load("./model/checkpoint20.npz")
    print(model.files)

    for name in model.files:
        temp = model[name]
        temp.tofile(os.path.join("model",name+".bin"))
        print(temp.shape)

    >>> ['enc_emb', 'enc_w_ih', 'enc_w_hh', 'enc_b_ih', 'enc_b_hh', 'dec_emb', 'dec_w_ih', 'dec_w_hh', 'dec_b_ih', 'dec_b_hh', 'fc_w', 'fc_b']
    >>> (29, 256)
    >>> (768, 256)
    >>> (768, 256)
    >>> (768,)
    >>> (768,)
    >>> (74, 256)
    >>> (768, 256)
    >>> (768, 256)
    >>> (768,)
    >>> (768,)
    >>> (74, 256)
    >>> (74,)

## Install

```
mkdir build
cd build
cmake ..
make
./debug
```

## Usage

详见[debug.cpp](debug.cpp)。