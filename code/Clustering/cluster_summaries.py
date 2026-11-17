#!/usr/bin/env python3
# cluster_summaries.py
# usage:
#   pip install scikit-learn
#   python cluster_summaries.py --input ./hdbscan/clustered_posts.json --outdir ./narratives --top_k 15 --exemplars 10

import os, json, argparse, re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# --------- utils ---------
def clean(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # remove urls
    text = re.sub(r"[^\w\s@#']", " ", text)              # keep words, mentions, hashtags, apostrophes
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def load_clusters(path: str) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clusters = {}
    for k, v in data.get("clusters", {}).items():
        try: clusters[int(k)] = v
        except: pass
    return clusters, data.get("summary", {})

def build_class_docs(clusters: Dict[int, List[Dict[str, Any]]]) -> Tuple[List[str], List[int], Dict[int,int]]:
    docs, labels, sizes = [], [], {}
    for cid, posts in clusters.items():
        if cid == -1: continue
        texts = [clean(p.get("text","")) for p in posts if p.get("text")]
        if not texts: continue
        sizes[cid] = len(texts)
        docs.append(" ".join(texts))
        labels.append(cid)
    return docs, labels, sizes

def top_terms_per_class(class_docs: List[str], labels: List[int], top_k: int=15) -> Dict[int, List[Tuple[str,float]]]:
    if not class_docs: return {}
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        min_df=2,
        max_features=50000
    )
    X = vectorizer.fit_transform(class_docs)     # shape: n_classes x vocab
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    X_tfidf = tfidf.fit_transform(X)             # c-TF-IDF-like weighting
    vocab = np.array(vectorizer.get_feature_names_out())
    results = {}
    for i, cid in enumerate(labels):
        row = X_tfidf[i].toarray().ravel()
        idx = np.argsort(-row)[:top_k]
        terms = [(vocab[j], float(row[j])) for j in idx if row[j] > 0]
        results[cid] = terms
    return results

def exemplars(posts: List[Dict[str,Any]], k: int=10) -> List[Dict[str,Any]]:
    def key(p): return (p.get("probability",0.0), p.get("likes",0), p.get("reposts",0))
    ex = sorted(posts, key=key, reverse=True)[:k]
    fields = ("post_id","thread_id","user_handle","probability","likes","reposts","quotes","text")
    return [{f: p.get(f) for f in fields} for p in ex]

# --------- main ---------
def main():
    ap = argparse.ArgumentParser(description="c-TF-IDF summaries + exemplars per HDBSCAN cluster (exclude -1).")
    ap.add_argument("--input", default="./hdbscan/clustered_posts.json")
    ap.add_argument("--outdir", default="./narratives")
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--exemplars", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    clusters, summary = load_clusters(args.input)
    class_docs, labels, sizes = build_class_docs(clusters)
    terms = top_terms_per_class(class_docs, labels, args.top_k)

    out = {
        "source": os.path.abspath(args.input),
        "summary": summary,
        "noise": {"posts": len(clusters.get(-1, []))},
        "clusters": {}
    }

    for cid in sorted([c for c in clusters.keys() if c != -1]):
        posts = clusters[cid]
        out["clusters"][str(cid)] = {
            "size": sizes.get(cid, len(posts)),
            "top_terms": terms.get(cid, []),
            "exemplars": exemplars(posts, args.exemplars)
        }

    out_path = os.path.join(args.outdir, "cluster_summaries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(out_path)

if __name__ == "__main__":
    main()

#python cluster_summaries.py --input ./hdbscan/clustered_posts.json --outdir ./narratives --top_k 15 --exemplars 10
