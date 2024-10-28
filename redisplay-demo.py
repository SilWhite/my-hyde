import json

from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder

from generator import DeepSeekGenerator
from promptor import Promptor
from hyde import HyDE

import config


promptor = Promptor("web search")
generator = DeepSeekGenerator(config.model, config.key)
# 可能存在网络问题
# encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
encoder = AutoQueryEncoder(encoder_dir=config.encoder_dir, pooling="mean")
searcher = FaissSearcher(config.search_index_dir, encoder)
# 可能存在网络问题
# corpus = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
corpus = LuceneSearcher(index_dir=config.corpus_dir)

# buile a hyde pipeline
hyde = HyDE(promptor, generator, encoder, searcher)

# example query
query = "how long does it take to remove wisdom tooth"

# build zero-shot prompt
prompt = hyde.prompt(query)
print("Example Prompt:\n", prompt)
print()

# generate hypothesis documents
hypothesis_documents = hyde.generate(query)
for i, doc in enumerate(hypothesis_documents):
    print(f"HyDE Generated Document: {i}")
    print(doc.strip())
print()

# encode hyde vector
hyde_vector = hyde.encode(query, hypothesis_documents)
print("hyde_vector shape:", hyde_vector.shape)
print()

# search relevant documents using HyDE vector
hits = hyde.search(hyde_vector, k=10)
for i, hit in enumerate(hits):
    print(f"HyDE Retrieved Document: {i}")
    print(hit.docid)
    print(json.loads(corpus.doc(hit.docid).raw())["contents"])
print()
# end-to-end search
# which is all steps above
# hits = hyde.e2e_search(query, k=10)
# for i, hit in enumerate(hits):
#     print(f"HyDE Retrieved Document: {i}")
#     print(hit.docid)
#     print(json.loads(corpus.doc(hit.docid).raw())["contents"])
