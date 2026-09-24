# Reflection

### 1. 
Tỷ lệ zero entries là trên 99.9% lệch 9.9% so với dự đoán

### 2. Kết quả nào bất ngờ nhất?
Tỷ lệ zero entries lên trên 99%

### 3. Experiment nào cung cấp evidence mạnh nhất?
Experiment 2
### 4. Failure case quan trọng nhất là gì?
Hiện tượng **Đa nghĩa (Polysemy)** ở truy vấn `"transformer language model"`: Rank 1 là Document 27936 (nói về biến áp điện và mô hình xe lửa đồ chơi Hobby 720). TF-IDF chỉ đếm từ rời rạc nên bị đánh lừa bởi `transformer` và `model`, không hiểu ngữ cảnh NLP. Kế đến là lỗi **Lệch từ vựng (Synonym Mismatch)** (`bbq` vs `barbecue`, `heart attack` vs `myocardial infarction`) khiến tương đồng bằng 0.

### 5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?
Hệ thống tìm kiếm 3 tầng lai ghép (**Tri-hybrid Retrieval**):
1. **Tầng Lexical (FTS5 / BM25):** Bắt chính xác tên riêng, từ khóa canon và từ viết tắt.
2. **Tầng Semantic (Dense Vector Embeddings):** Vector hóa ngữ nghĩa toàn câu (Gemini/BERT), giải quyết từ đồng nghĩa/đa nghĩa; kết hợp với FTS qua thuật toán **Reciprocal Rank Fusion (RRF)**.
3. **Tầng Knowledge Graph (Graph RAG):** Biểu diễn thực thể thành bộ ba `<Subject, Predicate, Object>`, cho phép **suy luận đa bước (Multi-hop Reasoning)** theo quan hệ logic, loại bỏ ảo giác.

---

### 6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?

- **AI Contribution:**
  - Hỗ trợ tối ưu code tính Cosine Similarity ma trận thưa trong notebook (`experiments.ipynb`).
  - Hỗ trợ giải thích convention của `scikit-learn` (smoothing, L2-norm).
  - Hỗ trợ định dạng và trình bày lại các file theo chuẩn Markdown/LaTeX.
- **Sinh viên tự thực hiện:** Tự giải và đối chiếu kết quả tính tay, tự đưa ra dự đoán trước thực nghiệm, trực tiếp phân tích failure cases trên 30K documents và tự đúc kết bài thu hoạch này.
