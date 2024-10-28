import numpy as np


class HyDE:

    def __init__(self, promptor, generator, encoder, searcher):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher

        self.prompt_rst = None
        self.gen_docs = None
        self.hyde_vector = None
        self.hits = None

    def prompt(self, query):
        self.prompt_rst = self.promptor.build_prompt(query)
        return self.prompt_rst

    def generate(self, query):
        prompt = (
            self.prompt_rst if self.prompt_rst else self.promptor.build_prompt(query)
        )
        gen_docs = self.generator.generate(prompt)
        self.gen_docs = gen_docs
        return gen_docs

    def encode(self, query, gen_docs):
        all_emb_c = []
        for c in [query] + gen_docs:
            c_emb = self.encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        self.hyde_vector = hyde_vector
        return hyde_vector

    def search(self, hyde_vector, k=10):
        hits = self.searcher.search(hyde_vector, k=k)
        self.hits = hits
        return hits

    def e2e_search(self, query, k=10):
        prompt = (
            self.prompt_rst if self.prompt_rst else self.promptor.build_prompt(query)
        )
        gen_docs = self.gen_docs if self.gen_docs else self.generator.generate(prompt)
        hyde_vector = (
            self.hyde_vector if self.hyde_vector else self.encode(query, gen_docs)
        )
        hits = self.hits if self.hits else self.searcher.search(hyde_vector, k=k)
        return hits
