# NLP_TH

## LAB 01 — From Text Processing to Search

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)  
**Học kỳ:** Học kỳ I - 2026  
**Chủ đề:** Text Processing, TF-IDF Representation và Document Search

---

## 1. Cấu trúc thư mục nộp bài (Deliverables)

Các thành phần nộp bài theo mục 17 của tài liệu `W1.pdf` được đặt trực tiếp trong thư mục gốc của dự án:

```
./
├── README.md              # Báo cáo tổng hợp, hướng dẫn chạy và cấu trúc lab
├── calculations.md        # Lời giải chi tiết các bài tính toán lý thuyết (Part B)
├── prediction.md          # Các dự đoán trước khi tiến hành thực nghiệm (Part C)
├── implementation.py      # Tự cài đặt thuật toán TF-IDF cốt lõi và kiểm thử unit test (Part E)
├── experiments.ipynb      # Notebook chứa toàn bộ chu trình thực nghiệm trên 30K documents (Parts D -> J)
├── results.csv            # Bảng kết quả đánh giá định lượng retrieval metrics (Part H)
├── reflection.md          # Bài thu hoạch và phân tích nhận thức cá nhân (< 500 từ) (Section 16)
└── datasets/              # Chứa tập ngữ liệu mẫu C4 (30.000 documents)
```

---

## 2. Tóm tắt kết quả thực nghiệm chính

### 2.1. Biểu diễn thưa trên 30.000 documents (Part D)
- **Số lượng văn bản ($N$):** 30.000 documents.
- **Kích thước từ vựng ($V$):** 177.230 unique terms.
- **Độ thưa của ma trận (Sparsity):** **$99.9068\%$** (chỉ có $0.0932\%$ phần tử khác 0).

### 2.2. So sánh Preprocessing Ablation (Part F)

| Pipeline | Chi tiết Preprocessing | Vocabulary Size | Avg Tokens/Doc | Matrix Sparsity | OOV Rate | Recall@5 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A** | Lowercase + Whitespace | 473.388 | 361.09 | 99.9611% | 0.00% | 6.67% | 0.2099 |
| **Pipeline B** | Lowercase + Punctuation + Stopwords | 167.107 | 188.36 | 99.9286% | 0.00% | **14.17%** | **0.2868** |
| **Pipeline C** | Normalization + Subword BPE | 7.859 | 548.55 | 97.0985% | 0.00% | 4.17% | 0.1833 |

**Kết luận thực nghiệm:**
- **Pipeline B** cho chất lượng tìm kiếm vượt trội nhất.
- Giảm kích thước từ vựng không đồng nghĩa với tìm kiếm tốt hơn (Pipeline C từ vựng nhỏ nhất nhưng kết quả kém nhất do phân mảnh từ thành các mẩu unigram gây nhiễu).

### 2.3. Đánh giá hệ thống tìm kiếm (Part G & H)
Hệ thống Document Search Engine xây dựng trên ma trận thưa TF-IDF với chuẩn hóa L2 và Cosine Similarity đạt hiệu năng trung bình trên tập kiểm thử:
- **Mean Precision@5:** 6.67%
- **Mean Recall@5:** 7.50%
- **Mean Reciprocal Rank (MRR):** 0.1250

*(Chi tiết từng truy vấn được lưu tại `results.csv`).*

### 2.4. Phân tích lỗi (Error Analysis - Part I & J)
- **Hiện tượng Đa nghĩa (Polysemy):** Truy vấn `"transformer language model"` trả về bài viết về máy biến áp mạch điện mô hình đồ chơi ở Rank 1 vì cùng chia sẻ các từ khóa unigram `transformer` và `model`.
- **Hiện tượng Lệch từ vựng (Synonym Mismatch):** `"heart attack treatment"` vs `"myocardial infarction therapy"` hoàn toàn không tìm thấy nhau (Cosine similarity = 0).
- **Động lực tiến hóa:** Sự thất bại của biểu diễn từ vựng rời rạc (Lexical Representation) dẫn tới nhu cầu về **Biểu diễn ngữ nghĩa (Distributional Semantics, Word2Vec, BERT, Dense Retrieval)**.

---

## 3. Hướng dẫn chạy và tái lập thực nghiệm

### Cài đặt môi trường:
```bash
pip install numpy scipy scikit-learn pandas tokenizers ipykernel nbclient
```

### Chạy kiểm thử mã nguồn cốt lõi (Unit Tests):
```bash
python implementation.py
```

### Chạy và xem Notebook:
Mở và thực thi toàn bộ các cells trong `experiments.ipynb` bằng Jupyter Notebook hoặc VS Code / Antigravity IDE. Toàn bộ các cells đã được thực thi và lưu lại đầy đủ đầu ra (outputs, charts, tables).
