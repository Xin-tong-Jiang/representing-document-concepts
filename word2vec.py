import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = []
with open('data.json', 'r') as f:
    for line in f:
        data.append(eval(line.strip()))  # each line is a JSON object for a Reddit post

tokenized_texts = [
    simple_preprocess(item['title'] + " " + item['body'] + " " + item['keywords'])
    for item in data
]

vector_size_list = [50, 100, 150]

for vector_size in vector_size_list:
    print(f"\n===== Word2Vec + Binned Bag-of-Words (vector_size={vector_size}) =====")
    
    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        epochs=40
    )

    # Cluster words into bins
    vocab_words = list(w2v_model.wv.index_to_key)
    word_vectors = np.array([w2v_model.wv[word] for word in vocab_words])
    kmeans_words = KMeans(n_clusters=vector_size, n_init=20, random_state=42)
    word_bins = kmeans_words.fit_predict(word_vectors)
    word2bin = {word: b for word, b in zip(vocab_words, word_bins)}

    # Build document vectors
    doc_vectors = []
    for tokens in tokenized_texts:
        vec = np.zeros(vector_size)
        for token in tokens:
            if token in word2bin:
                vec[word2bin[token]] += 1
        if len(tokens) > 0:
            vec = vec / len(tokens)
        doc_vectors.append(vec)
    doc_vectors = np.array(doc_vectors)
    Xn = normalize(doc_vectors)

    # Find best K for document clusters using silhouette
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(2, 11):
        kmeans_docs = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels_k = kmeans_docs.fit_predict(Xn)
        score = silhouette_score(Xn, labels_k, metric="cosine")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels_k

    print(f"Best k = {best_k} (silhouette = {best_score:.4f})")

    # Save vectors and labels
    np.save(f'word2vec_vectors_{vector_size}.npy', doc_vectors)
    np.save(f'word2vec_labels_{vector_size}.npy', best_labels)

    # Qualitative inspection
    df = pd.DataFrame({
        'document': [" ".join(tokens) for tokens in tokenized_texts],
        'cluster': best_labels
    })
    for c in sorted(df['cluster'].unique()):
        print(f"\nCluster {c}")
        samples = df[df['cluster'] == c].sample(
            n=min(10, len(df[df['cluster'] == c])),
            random_state=42
        )
        for i, text in enumerate(samples['document']):
            print(f"  {i+1}. {text[:150]}")

    # TSNE visualization
    X_reduced = TSNE(n_components=2, random_state=42).fit_transform(Xn)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=best_labels, cmap='jet', alpha=0.6)
    plt.title(f'TSNE Word2Vec (vector_size={vector_size}, best_k={best_k})')
    plt.savefig(f'tsne_word2vec_{vector_size}.png')
    plt.clf()
