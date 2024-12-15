import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from kiwipiepy import Kiwi
import numpy as np
from scipy.sparse import csr_array, vstack
from abc import ABC, abstractmethod
import json
from pathlib import Path
import requests


class BaseSparseEmbedding(ABC):
    @abstractmethod
    def embed_query(self, query: str) -> Dict[int, float]:
        pass
        
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        pass
        

class KiwiBM25EmbeddingFunction:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.kiwi = Kiwi()
        self.corpus_size = 0
        self.avgdl = 0
        self.idf = {}
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.average_idf = 0.0

    def _tokenize_text(self, text: str) -> List[str]:
        return [token.form for token in self.kiwi.tokenize(text)]

    def _tokenize_corpus(self, corpus: List[str]) -> List[List[str]]:
        return [[token.form for token in tokens] for tokens in self.kiwi.tokenize(corpus)]

    def _compute_statistics(self, corpus: List[List[str]]):
        term_document_frequencies = Counter()
        doc_lengths = np.array([len(doc) for doc in corpus])
        total_word_count = np.sum(doc_lengths)
        self.corpus_size = len(corpus)
        self.avgdl = total_word_count / self.corpus_size
        
        for document in corpus:
            unique_words = set(document)
            term_document_frequencies.update(unique_words)
            
        return term_document_frequencies

    def _calc_idf(self, term_document_frequencies: Dict):
        freqs = np.array(list(term_document_frequencies.values()))
        words = list(term_document_frequencies.keys())
        
        idfs = np.log(self.corpus_size - freqs + 0.5) - np.log(freqs + 0.5)
        self.average_idf = np.mean(idfs)
        eps = self.epsilon * self.average_idf
        idfs[idfs < 0] = eps
        self.idf = {word: [idf, i] for i, (word, idf) in enumerate(zip(words, idfs))}

    def fit(self, corpus: List[str]):
        self.corpus_size = 0
        self.idf = {}
        
        tokenized_corpus = self._tokenize_corpus(corpus)
        term_document_frequencies = self._compute_statistics(tokenized_corpus)
        self._calc_idf(term_document_frequencies)

    @property
    def dim(self):
        return len(self.idf)

    def encode_queries(self, queries: List[str]) -> csr_array:
        sparse_embs = []
        for query in queries:
            terms = self._tokenize_text(query)
            values, rows, cols = [], [], []
            for term in terms:
                if term in self.idf:
                    values.append(self.idf[term][0])
                    rows.append(0)
                    cols.append(self.idf[term][1])
            sparse_embs.append(
                csr_array((values, (rows, cols)), shape=(1, len(self.idf))).astype(np.float32)
            )
        return vstack(sparse_embs).tocsr()

    def encode_documents(self, documents: List[str]) -> csr_array:
        sparse_embs = []
        tokenized_docs = self._tokenize_corpus(documents)
        
        for terms in tokenized_docs:
            frequencies = Counter(terms)
            doc_len = len(terms)
            term_set = set(terms)
            
            values, rows, cols = [], [], []
            for term in term_set:
                if term in self.idf:
                    term_freq = frequencies[term]
                    value = (
                        term_freq
                        * (self.k1 + 1)
                        / (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    )
                    rows.append(0)
                    cols.append(self.idf[term][1])
                    values.append(value)
                    
            sparse_embs.append(
                csr_array((values, (rows, cols)), shape=(1, len(self.idf))).astype(np.float32)
            )
        return vstack(sparse_embs).tocsr()

    def save(self, path: str):
        bm25_params = {
            "version": "v1",
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "average_idf": self.average_idf,
            "idf_word": [None] * len(self.idf),
            "idf_value": [None] * len(self.idf),
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon
        }

        for word, (idf_value, idx) in self.idf.items():
            bm25_params["idf_word"][idx] = word
            bm25_params["idf_value"][idx] = float(idf_value)

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("w", encoding="utf-8") as json_file:
            json.dump(bm25_params, json_file, ensure_ascii=False, indent=2)

    def load(self, path: Optional[str] = None):
        default_meta_filename = "bm25_kiwi_v1.json"
        default_meta_url = "https://your-default-url.com/bm25_kiwi_v1.json"

        if path is None:
            path = Path(default_meta_filename)
            
            if not path.exists():
                response = requests.get(default_meta_url, timeout=30)
                response.raise_for_status()
                with path.open("wb") as f:
                    f.write(response.content)

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with load_path.open("r", encoding="utf-8") as json_file:
            bm25_params = json.load(json_file)

        required_fields = ["corpus_size", "avgdl", "idf_word", "idf_value"]
        missing_fields = [field for field in required_fields if field not in bm25_params]
        if missing_fields:
            raise ValueError(f"Missing required fields in JSON file: {missing_fields}")

        self.corpus_size = bm25_params["corpus_size"]
        self.avgdl = bm25_params["avgdl"]
        self.average_idf = bm25_params.get("average_idf", 0.0)
        self.k1 = bm25_params.get("k1", 1.5)
        self.b = bm25_params.get("b", 0.75)
        self.epsilon = bm25_params.get("epsilon", 0.25)

        self.idf = {}
        for i, (word, value) in enumerate(zip(bm25_params["idf_word"], bm25_params["idf_value"])):
            if word is None or value is None:
                raise ValueError(f"Invalid IDF entry at index {i}")
            self.idf[word] = [float(value), i]


class KiwiBM25SparseEmbedding(BaseSparseEmbedding):
    def __init__(self, corpus: Optional[List[str]] = None):
        self.bm25_ef = KiwiBM25EmbeddingFunction()
        if corpus is not None:
            self.bm25_ef.fit(corpus)
    
    def save(self, path: str):
        if not self.bm25_ef.idf:
            raise RuntimeError("Model must be fitted with corpus or loaded before saving")
        self.bm25_ef.save(path)
    
    def load(self, path: Optional[str] = None):
        self.bm25_ef.load(path)
    
    def fit(self, corpus: List[str]):
        self.bm25_ef.fit(corpus)

    def embed_query(self, text: str) -> Dict[int, float]:
        if not self.bm25_ef.idf:
            raise RuntimeError("Model must be fitted with corpus or loaded before embedding")
        sparse_array = self.bm25_ef.encode_queries([text])
        return self._sparse_to_dict(sparse_array)

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        if not self.bm25_ef.idf:
            raise RuntimeError("Model must be fitted with corpus or loaded before embedding")
        sparse_arrays = self.bm25_ef.encode_documents(texts)
        return [self._sparse_to_dict(sparse_array) for sparse_array in sparse_arrays]

    def _sparse_to_dict(self, sparse_array: csr_array) -> Dict[int, float]:
        row_indices, col_indices = sparse_array.nonzero()
        non_zero_values = sparse_array.data
        result_dict = {}
        for col_index, value in zip(col_indices, non_zero_values):
            result_dict[col_index] = value
        return result_dict