import json
import random
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load Data
data = []
with open('data.json', 'r') as f:
    for line in f:
        data.append(eval(line.strip()))

# Prepare Tagged Documents
documents = [TaggedDocument(simple_preprocess(item['title'] + " " + item['body'] + " " + item['keywords']), [i])
             for i, item in enumerate(data)]
texts = [" ".join(doc.words) for doc in documents]

# Try different vector sizes
vector_size_list = [50, 100, 150]
for vector_size in vector_size_list:
    print(f"\n===== Training Doc2Vec (vector_size={vector_size}) =====")
    model = Doc2Vec(documents, vector_size=vector_size, window=10, min_count=1, workers=4, epochs=40)

    # Get vectors for all documents
    vectors = np.array([model.dv[i] for i in range(len(documents))])
    np.save(f'vectors_{vector_size}.npy', vectors)

    # Normalize
    Xn = normalize(vectors)

    # Find best K by silhouette
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels_k = kmeans.fit_predict(Xn)
        score = silhouette_score(Xn, labels_k, metric="cosine")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels_k

    print(f"Best k = {best_k} (silhouette = {best_score:.4f})")

    labels = best_labels
    df = pd.DataFrame({'document': texts, 'cluster': labels})
    np.save(f"labels_{vector_size}.npy", labels)

    # Qualitative evaluation
    for c in sorted(df['cluster'].unique()):
        print(f"\nCluster {c}")
        samples = df[df['cluster'] == c].sample(n=min(10, len(df[df['cluster'] == c])), random_state=42)
        for i, text in enumerate(samples['document']):
            print(f"  {i+1}. {text[:150]}")

    # Visualization
    X_reduced = TSNE(n_components=2, random_state=42).fit_transform(Xn)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='jet', alpha=0.6)
    plt.title(f'TSNE (vector_size={vector_size}, k={best_k})')
    plt.savefig(f'tsne_plot_{vector_size}.png')
    plt.clf()