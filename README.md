# 复现论文
## 一些说明
- 原文献 https://arxiv.org/abs/2212.10496
- 原代码仓库 https://github.com/texttron/hyde
- 由于源代码使用jupyter notebook，个人不甚习惯，遂整理为`.py`文件。
    - 整理完发现好像jupyter notebook也挺好用，`.py`不便调试且会生成临时文件。
- 使用DeepSeek LLM API（https://platform.deepseek.com ）替换GPT-3 API
    - 该模型API可以免费注册获取500w token额度，但有时效
    - 可以本地部署LLM以避免API的相关问题
- 测试集为TREC DL19
- 实验结果如下
    |test|map|nDCG@10|Recall@1000|
    |:---:|:---:|:---:|:---:|
    |BM25|30.1|50.6|75.0|
    |Contriever|24.0|44.5|74.6|
    |HyDE|41.8|61.3|88.0|
    |1st result|39.6|58.7|85.6|
    |2nd result|39.7|59.4|87.5|
    |3rd result|39.1|59.7|85.7|
    |4th result|40.7|60.8|87.6|
    |5th result|39.5|59.1|86.7|
    |6th result|40.6|61.1|86.0|
    > 1st result: deepseek-chat, chat mode, 1 gen, system prompt: 'You are a helpful assistant'
    > 
    > 2nd result: deepseek-chat, chat mode, 1 gen, system prompt: ''
    > 
    > 3rd result: deepseek-chat, chat mode, 1 gen, system prompt: null
    > 
    > 4th result: deepseek-chat, fim mode, 1 gen, suffix: null
    > 
    > 5th result: deepseek-chat, prefix mode, 1 gen, system prompt: null, user prompt: null
    > 
    > 6th result: deepseek-chat, prefix mode, 1 gen, system prompt: 'You are a helpful assistant', user prompt: null

## 环境配置
- 下载目录并进入`my-hyde`目录
- 使用conda安装依赖
    ```shell
    conda env create -f environment.yml
    ```
    - 运行指令会生成名为`hyde-env`的环境，注意重名问题。
    - 由于cuda版本问题，需要手动安装pytorch（conda安装会有问题，这里使用pip）
        ```shell
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
        也可以前往官网手动指定cuda版本(https://pytorch.org/get-started)
    - 可能遇到numpy版本问题，需要手动回退版本
        ```shell
        pip uninstall numpy
        conda install numpy=1.26
        ```
    - 如果自己配置环境，注意需要指定`pyserini==0.39.0`，新版本似乎无法导入模块。
    - 如果自己配置环境，可能需要安装`faiss`，参考以下教程按需安装https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
- 模型、数据集、检索索引本地使用

    由于网络问题，代码中需要使用的HuggingFace模型、索引等文件，可能无法在线下载，可能需要手动下载并在`config.py`中进行本地地址配置。
    - facebook-contriever
        - https://huggingface.co/facebook/contriever/tree/main
        - 似乎只能下载文件后手动放到同一目录
    - contriever_msmarco_index
        - https://www.dropbox.com/s/dytqaqngaupp884/contriever_msmarco_index.tar.gz
        - `tar -xvf contriever_msmarco_index.tar.gz`
        - 该文件在原仓库使用`wget`下载，但似乎有验证，直接网页下载。
    - lucene-inverted.msmarco-v1-passage.20221004.252b5e
        - https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene/lucene-inverted.msmarco-v1-passage.20221004.252b5e.tar.gz
        - 使用`LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")`下载均速50kB/s，网页下载10MB/s。


    最终的本地文件目录如下：
    ```shell
    my-hyde
    ├── README.md
    ├── config.py
    ├── contriever_msmarco_index
    │   └── ...
    ├── environment.yml
    ├── facebook-contriever
    │   └── ...
    ├── gen_file
    │   └── ...
    ├── generator.py
    ├── hyde.py
    ├── lucene-inverted.msmarco-v1-passage.20221004.252b5e
    │   └── ...
    ├── promptor.py
    ├── redisplay-demo.py
    └── redisplay-dl19.py
    ```
## 运行
- 在`config.py`中配置模型、数据集、检索索引地址
- 在`config.py`中配置API key
- 运行`redisplay-dl19.py`生成测试集结果
- 运行`redisplay-demo.py`生成示例结果
