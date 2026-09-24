import math
import numpy as np

def build_vocabulary(corpus):
    vocab = set()
    for doc in corpus:
        for word in doc:
            vocab.add(word)
    return sorted(list(vocab))

def compute_counts(corpus, vocabulary):
    counts = []
    for doc in corpus:
        doc_count = {word: doc.count(word) for word in vocabulary}
        counts.append(doc_count)
    return counts

def compute_tf(doc_counts):
    total_terms = sum(doc_counts.values())
    if total_terms == 0:
        return {word: 0.0 for word in doc_counts}
    return {word: count / total_terms for word, count in doc_counts.items()}

def compute_idf(corpus, vocabulary):
    N = len(corpus)
    idf = {}
    for word in vocabulary:
        df = sum(1 for doc in corpus if word in doc)
        idf[word] = math.log10(N / df) if df > 0 else 0.0
    return idf

def compute_tfidf(tf, idf):
    return {word: tf[word] * idf[word] for word in tf}

def cosine_similarity(vec1, vec2):
    if isinstance(vec1, dict):
        v1 = np.array(list(vec1.values()))
        v2 = np.array(list(vec2.values()))
    else:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

if __name__ == "__main__":
    corpus_raw = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]
    corpus = [doc.split() for doc in corpus_raw]
    
    vocab = build_vocabulary(corpus)
    print("Vocabulary: \n", vocab)
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocab == expected_vocab, f"Vocabulary sai: {vocab}"
    
    print("\nTerm Frequency")
    counts = compute_counts(corpus, vocab)
    for i, doc_count in enumerate(counts):
        tf = compute_tf(doc_count)
        print(f"D{i+1} counts: {list(doc_count.values())}")
        print(f"D{i+1} TF: {[round(v, 4) for v in tf.values()]}")
    
    tf_D1 = compute_tf(counts[0])
    assert abs(tf_D1["cat"] - (1/3)) < 1e-9
    
    print("\nInverse Document Frequency:")
    idf = compute_idf(corpus, vocab)
    print("IDF:", [round(v, 4) for v in idf.values()])
    assert abs(idf["fish"] - 0.0) < 1e-9
    assert abs(idf["cat"] - math.log10(1.5)) < 1e-9
    
    print("\nTF-IDF:")
    for i, doc_count in enumerate(counts):
        tf = compute_tf(doc_count)
        tfidf = compute_tfidf(tf, idf)
        print(f"D{i+1} TF-IDF: {[round(v, 4) for v in tfidf.values()]}")

    print("\nCosine Similarity:")
    x = [1, 1, 1]
    y = [1, 1, 0]
    sim = cosine_similarity(x, y)
    print(f"Vector x: {x}")
    print(f"Vector y: {y}")
    print(f"Cosine Similarity(x, y): {sim:.4f}")
    
    expected_sim = 2 / (math.sqrt(3) * math.sqrt(2))
    assert abs(sim - expected_sim) < 1e-9


    # ==========================================
    # 8.5. SO SÁNH VỚI THƯ VIỆN (REFERENCE IMPLEMENTATION)
    # ==========================================
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as ref_cosine_similarity

    print("\n=== 8.5. SO SÁNH VỚI REFERENCE IMPLEMENTATION (scikit-learn) ===")

    # 1. Khởi tạo TfidfVectorizer mặc định của scikit-learn
    vectorizer = TfidfVectorizer()
    ref_tfidf_matrix = vectorizer.fit_transform(corpus_raw).toarray()
    ref_vocab = vectorizer.get_feature_names_out().tolist()

    print(f"Student Vocabulary  : {vocab}")
    print(f"Reference Vocabulary: {ref_vocab}")
    print(f"Vocabulary khớp nhau: {vocab == ref_vocab}")

    # 2. So sánh Cosine Similarity với thư viện
    student_sim = cosine_similarity(x, y)
    ref_sim = float(ref_cosine_similarity([x], [y])[0][0])

    print(f"\nStudent Cosine Similarity  : {student_sim:.6f}")
    print(f"Reference Cosine Similarity: {ref_sim:.6f}")
    assert abs(student_sim - ref_sim) < 1e-9, "Cosine similarity không khớp với thư viện!"
    print("-> Cosine Similarity khớp hoàn toàn với thư viện!")

    # 3. So sánh TF-IDF giữa hai bên cho Document 1
    student_d1_tfidf = [round(v, 4) for v in compute_tfidf(compute_tf(counts[0]), idf).values()]
    ref_d1_tfidf = [round(v, 4) for v in ref_tfidf_matrix[0]]

    print(f"\nStudent D1 TF-IDF  : {student_d1_tfidf}")
    print(f"Reference D1 TF-IDF: {ref_d1_tfidf}")