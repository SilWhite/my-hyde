from tqdm import tqdm
import json
import os

from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search import get_topics, get_qrels

from generator import DeepSeekGenerator
from promptor import Promptor
from hyde import HyDE

import config

# Initialize Contriever Index and Query Encoder
# 可能存在网络问题
# query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
query_encoder = AutoQueryEncoder(encoder_dir=config.encoder_dir, pooling="mean")
searcher = FaissSearcher(config.search_index_dir, query_encoder)
# 可能存在网络问题
# corpus = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
corpus = LuceneSearcher(index_dir=config.corpus_dir)

# Load query and judegments for dl19-passage dataset
topics = get_topics("dl19-passage")
qrels = get_qrels("dl19-passage")
# check the first 5 topics and qrels
# print("TREC DL19 Passage Topics:")
# print({k: v for k, v in list(topics.items())[:2]})
# print("TREC DL19 Passage Qrels:")
# print({k: v for k, v in list(qrels.items())[:2]})

# Run Contriever
with open("./gen_file/dl19-contriever-top1000-trec", "w") as f:
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]["title"]
            hits = searcher.search(query, k=1000)
            rank = 0
            for hit in hits:
                rank += 1
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score} rank\n")

# Evaluate Contriever
os.system(
    "python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage ./gen_file/dl19-contriever-top1000-trec"
)
os.system(
    "python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage ./gen_file/dl19-contriever-top1000-trec"
)
os.system(
    "python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage ./gen_file/dl19-contriever-top1000-trec"
)
"""
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
    - Avoid using `tokenizers` before the fork if possible
    - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
828.59s - pydevd: Sending message related to process being replaced timed-out after 5 seconds
map                   	all	0.2399
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
    - Avoid using `tokenizers` before the fork if possible
    - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
840.94s - pydevd: Sending message related to process being replaced timed-out after 5 seconds
ndcg_cut_10           	all	0.4454
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
    - Avoid using `tokenizers` before the fork if possible
    - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
853.27s - pydevd: Sending message related to process being replaced timed-out after 5 seconds
recall_1000           	all	0.7459
"""


# Run HyDE
generator = DeepSeekGenerator(config.model, config.key, api_mode=config.api_mode)
promptor = Promptor("web search")
hyde = HyDE(promptor, generator, query_encoder, searcher)

rst_file = (
    f"./gen_file/hyde-dl19-contriever-deepseek-{config.api_mode}-top1000-1gen-trec"
)
gen_file = f"./gen_file/hyde-dl19-deepseek-{config.api_mode}-1gen.jsonl"
with open(rst_file, "w") as f, open(gen_file, "w") as fgen:
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]["title"]
            gen_context = hyde.generate(query)
            fgen.write(
                json.dumps(
                    {"query_id": qid, "query": query, "contexts": [query] + gen_context}
                )
                + "\n"
            )
            hyde_vector = hyde.encode(query, gen_context)
            hits = hyde.search(hyde_vector, k=1000)

            rank = 0
            for hit in hits:
                rank += 1
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score} rank\n")
# Evaluate HyDE
os.system(f"python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage {rst_file}")
os.system(
    f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage {rst_file}"
)
os.system(
    f"python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage {rst_file}"
)

"""
paper results:
            mAP     nDCG@10   Recall@1000
BM25:       30.1    50.6      75.0
Contriever: 24.0    44.5      74.6
HyDE:       41.8    61.3      88.0
"""
"""
1st result(deepseek-chat, chat mode, 1 gen, system prompt: 'You are a helpful assistant'):
map                   	all	    0.3962
ndcg_cut_10           	all	    0.5868
recall_1000           	all	    0.8556

2nd result(deepseek-chat, chat mode, 1 gen, system prompt: ''):
map                   	all	    0.3967
ndcg_cut_10           	all	    0.5937
recall_1000           	all	    0.8753

3rd result(deepseek-chat, chat mode, 1 gen, system prompt: null):
map                   	all	    0.3905
ndcg_cut_10           	all	    0.5966
recall_1000           	all	    0.8571

4th result(deepseek-chat, fim mode, 1 gen, suffix: null):
map                   	all	    0.4073
ndcg_cut_10           	all	    0.6077
recall_1000           	all	    0.8760

5th result(deepseek-chat, prefix mode, 1 gen, system prompt: null, user prompt: null):
map                   	all	    0.3949
ndcg_cut_10           	all	    0.5906
recall_1000           	all	    0.8665

6th result(deepseek-chat, prefix mode, 1 gen, system prompt: 'You are a helpful assistant', user prompt: null):
map                     all     0.4062
ndcg_cut_10             all     0.6110
recall_1000             all     0.8595
"""
