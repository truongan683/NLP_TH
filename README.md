# LAB 03 - Word Representations and Embeddings

## 1. Nội dung bài làm

Trong bài này, em tìm hiểu cách biểu diễn từ dựa trên ngữ cảnh. Em bắt
đầu với ma trận co-occurrence để thấy các giá trị được đếm từ đâu, sau
đó dùng Word2Vec để khảo sát similarity, analogy và tìm kiếm văn bản.

| File | Nội dung |
|---|---|
| `README.md` | Cách thực hiện, kết quả và nhận xét chung |
| `calculations.md` | Co-occurrence, cosine, sparse/dense, CBOW, Skip-gram và tính analogy |
| `prediction.md` | Bốn dự đoán ban đầu, lý do và mức độ tin tưởng |
| `cooccurrence.py` | Xây vocabulary, đếm context, tính cosine, tìm từ gần nhất |
| `word_embedding.ipynb` | Thực nghiệm và kết quả chạy |
| `results.csv` | Số liệu chi tiết của từng cấu hình |
| `error_analysis.md` | Ba kết quả phù hợp và ba kết quả bất ngờ |
| `reflection.md` | So sánh biểu diễn và phân tích từ nhiều nghĩa |

Em dùng môi trường `NLP` ở root, Python 3.14.6, Gensim 4.4.0,
NumPy 2.5.3, SciPy 1.18.1 và scikit-learn 1.9.1. Kernel của notebook
là **Python (NLP)**. Có thể kích hoạt môi trường từ root bằng:

```powershell
.\NLP\Scripts\Activate.ps1
```

## 2. Dữ liệu và cách xử lý

Em lấy 6.000 document đầu tiên trong file
`datasets/c4-train.00000-of-01024-30K.json`, theo đúng thứ tự file.
Em không chọn riêng các document có chứa từ cần đánh giá.

Cách xử lý gồm chuyển chữ thường, tách câu tại `.`, `!`, `?` và xuống
dòng, sau đó lấy token bằng biểu thức `[a-z]+`. Em giữ stopwords và
chỉ dùng các câu có ít nhất 3 token. Dữ liệu sau xử lý có 128.104 câu,
2.224.629 token và 67.927 từ khác nhau. Với min_count=3, vocabulary
của Word2Vec còn 29.367 từ.

Cách tách này đơn giản, nhưng có thể xử lý chưa tốt tên viết tắt, số
và từ ghép. Corpus web cũng có những đoạn lặp hoặc nội dung nhiễu.
Vì vậy, khi phân tích kết quả, em xem thêm tần suất và câu nguồn.

## 3. Thực nghiệm co-occurrence

Em dùng 12 câu ngắn cho phần thử nghiệm ban đầu. Window là số vị trí
xét ở mỗi phía của target. Mỗi context trong cửa sổ được cộng 1 và
không đếm sang câu khác. Nếu một từ nằm ngoài vocabulary, em vẫn giữ
vị trí của nó khi xác định khoảng cách.

| Window | Vocabulary | Ma trận | Nonzero | cat-dog | doctor-physician | doctor-hospital |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 26 | 26 × 26 | 68 | 1.0000 | 1.0000 | 0.7746 |
| 2 | 26 | 26 × 26 | 114 | 0.8750 | 1.0000 | 0.8199 |
| 5 | 26 | 26 × 26 | 149 | 0.8750 | 1.0000 | 0.8496 |

Vocabulary và kích thước ma trận không đổi khi tăng window. Số ô khác
0 tăng vì mỗi target được xét với nhiều context hơn. Cosine lại không
tăng theo cùng một quy luật. Doctor và physician có context giống nhau
trong corpus nhỏ này nên similarity bằng 1 ở cả ba window. Cat và dog
chỉ bằng 1 ở window=1; khi lấy thêm fish, milk, meat ở xa hơn, similarity
giảm còn 0,875.

Em cũng thử PPMI kết hợp SVD 3 chiều để minh họa cách nén ma trận đếm
thành vector dense. Phần này giúp so sánh hai dạng biểu diễn; em chưa
đánh giá nó trên một benchmark riêng.

## 4. Word2Vec và các từ gần nhất

Em dùng Skip-gram với negative sampling và các thông số sau:

| Thông số | Giá trị |
|---|---|
| Objective | Skip-gram (`sg=1`) |
| Negative samples | 5, `hs=0` |
| min_count | 3 |
| epochs | 8 |
| sample | 0,001 |
| workers | 1 |
| seed | 42 |
| shrink_windows | False |
| sorted_vocab | 1 |
| Learning rate | 0,025 giảm đến 0,0001 |

Model baseline có 100 chiều và window=5. Em so sánh thêm window 2, 10
và dimension 50, 300, giữ các tham số còn lại như nhau. Baseline dùng
chung trong hai phép so sánh nên có tổng cộng 5 cấu hình. Em không dùng
pretrained vectors.

| Từ | Tần suất | Top-5 (cosine) |
| --- | --- | --- |
| doctor | 204 | physician (0.7590); anesthesia (0.7328); dentist (0.7133); surgeon (0.6918); ophthalmologist (0.6898) |
| hospital | 234 | cardiologist (0.6620); chongqing (0.6414); rehabilitation (0.6349); navicent (0.6332); pediatrics (0.6322) |
| patient | 232 | anesthesia (0.6640); childbirth (0.6281); doctor (0.6230); hospice (0.6160); checkup (0.6072) |
| disease | 233 | chronic (0.8041); diseases (0.7817); degenerative (0.7749); kidney (0.7515); rheumatoid (0.7501) |
| computer | 341 | device (0.6632); encrypt (0.6348); uploading (0.6314); settings (0.6288); desktop (0.6259) |
| football | 162 | basketball (0.7583); willamette (0.7137); playoffs (0.7043); championships (0.7034); cardinals (0.7033) |
| banana | 31 | tuscan (0.7041); fennel (0.7015); confectioners (0.6985); tomatoes (0.6930); roux (0.6909) |

Physician đứng đầu danh sách của doctor, phù hợp với quan hệ gần nghĩa.
Dentist, surgeon và ophthalmologist cũng thuộc các nghề khám chữa bệnh.
Anesthesia lại chỉ việc gây mê. Em thấy danh sách này chứa cả các từ
gần nghĩa và các từ liên quan về chủ đề.

Trong corpus, doctor xuất hiện 204 lần và physician xuất hiện 65 lần.
Các context chung có care, medical, consult và medicine. Doc 129 có
family physician, clinical practice và attending physician; doc 282
có doctor trong một câu về skincare clinic và treatment. Những ví dụ
này hỗ trợ cách giải thích dựa trên context y tế.

## 5. Ảnh hưởng của window

Em giữ dimension=100 và thay window:

| Cặp từ | Window 2 | Window 5 | Window 10 |
| --- | --- | --- | --- |
| doctor-physician | 0.7143 | 0.7590 | 0.6365 |
| doctor-hospital | 0.5244 | 0.5629 | 0.5202 |
| cat-dog | 0.6668 | 0.6130 | 0.5694 |

Doctor-physician tăng từ 0,7143 lên 0,7590 khi window tăng từ 2 lên 5,
nhưng giảm còn 0,6365 ở window=10. Cat-dog giảm ở cả hai lần tăng window.
Kết quả cho thấy cửa sổ lớn hơn không phải lúc nào cũng tốt hơn.

Em nghĩ window rộng có thể bổ sung thông tin về chủ đề, đồng thời đưa
vào context không đặc trưng cho target. Trong Gensim, việc loại các từ
hiếm và subsampling cũng làm context dùng để train khác với cửa sổ đếm
trên chuỗi token gốc.

## 6. Ảnh hưởng của số chiều

Em giữ window=5 và so sánh ba dimension:

| Dimension | Train (s) | Model (MiB) | Vectors (MiB) | Spearman | Analogy@1 | Analogy@5 | Search MRR | Recall@3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 53.39 | 12.05 | 5.60 | 0.6156 | 12.5% | 25.0% | 1.0000 | 1.0000 |
| 100 | 56.93 | 23.26 | 11.20 | 0.6156 | 25.0% | 37.5% | 1.0000 | 1.0000 |
| 300 | 100.69 | 68.07 | 33.61 | 0.6156 | 12.5% | 50.0% | 1.0000 | 1.0000 |

Train time gồm xây vocabulary và huấn luyện, không gồm đánh giá hoặc
lưu model. MiB bằng 2²⁰ byte. Cột Model đo file đã lưu, còn Vectors chỉ
đo mảng input embeddings float32. Thời gian trong bảng lấy từ lượt
chạy mới nhất được ghi trong `results.csv`.

Model 300 chiều lớn hơn model 100 chiều, nhưng accuracy Top-1 của
analogy giảm từ 25% xuống 12,5%. Top-5 lại tăng từ 37,5% lên 50%. Em
thấy một cấu hình có thể tốt hơn ở metric này nhưng kém hơn ở metric
khác, nên cần chọn theo mục tiêu sử dụng.

Cả ba model đều đạt MRR và Recall@3 bằng 1 trên bộ tìm kiếm nhỏ. Tập này
chưa đủ khó để phân biệt ba cấu hình. Trong phạm vi bài lab, em giữ
100 chiều làm baseline vì chi phí thấp hơn 300 chiều và Top-1 analogy
cao hơn trong lượt thử này.

## 7. Đánh giá word similarity

| Hạng | Cặp từ | Cosine | Mức trực giác (0-10) |
| --- | --- | --- | --- |
| 1 | doctor-physician | 0.7590 | 9.5 |
| 2 | cat-dog | 0.6130 | 7.0 |
| 3 | car-automobile | 0.5851 | 9.5 |
| 4 | king-queen | 0.4753 | 7.5 |
| 5 | computer-banana | 0.0157 | 0.0 |

Em so sánh thứ hạng với các mức trực giác đã đặt ban đầu: doctor-physician
và car-automobile là 9,5; king-queen là 7,5; cat-dog là 7; computer-banana
là 0. Spearman thu được là 0,6156, với đủ 5/5 cặp trong vocabulary.

Doctor-physician đứng cao và computer-banana đứng thấp là phù hợp.
Điểm em thấy bất ngờ là cat-dog cao hơn car-automobile dù cặp sau gần
đồng nghĩa hơn. Trong corpus, car xuất hiện 671 lần còn automobile chỉ
12 lần. Sự chênh lệch này có thể khiến automobile thiếu context để học,
nhưng em chưa thử tăng dữ liệu để xác nhận.

Các mức trực giác là nhận định chủ quan trên 5 cặp từ, chưa phải nhãn
chuẩn được nhiều người đánh giá. Vì vậy, hệ số trên chỉ giúp kiểm tra
sơ bộ, chưa đủ để kết luận chất lượng tổng quát của model.

## 8. Đánh giá word analogy

| Phép tính | Đáp án kỳ vọng | Top-1 thực tế | Top-1 đúng? | Trong Top-5? |
| --- | --- | --- | --- | --- |
| king-man+woman | queen | cal | Không | Không |
| paris-france+italy | rome | vilnius | Không | Không |
| berlin-germany+france | paris | madame | Không | Không |
| madrid-spain+italy | rome | multipliers | Không | Không |
| father-man+woman | mother | mother | Có | Có |
| brother-man+woman | sister | husband | Không | Có |
| walking-walk+swim | swimming | dom | Không | Không |
| bigger-big+small | smaller | smaller | Có | Có |

Cả 8 câu đều có đủ từ trong vocabulary. Model đúng Top-1 ở 2/8 câu
(25%) và có đáp án trong Top-5 ở 3/8 câu (37,5%). Câu king-man+woman
trả về cal, không phải queen. Kết quả không phải lúc nào cũng giống
ví dụ thường dùng để giới thiệu Word2Vec.

Em hiểu phép cộng trừ có thể phản ánh quan hệ nếu các hướng tương ứng
được học khá nhất quán. Tuy nhiên, đây vẫn là kết quả trong không gian
vector, chưa đủ để nói model hiểu quan hệ như người. Lệnh `most_similar`
của Gensim dùng các vector đầu vào đã chuẩn hóa; bài tính trên giấy dùng
trực tiếp các vector giả định được cho trong đề.

## 9. Ứng dụng semantic search

Em dùng 12 document và 6 query ngắn. Với mỗi query, em chỉ định trước
những document liên quan để tính metric. Model học từ corpus C4, không
train Word2Vec trên 12 document tìm kiếm này.

Vector của query và document là trung bình word vectors có trọng số
IDF. IDF được tính từ 12 document, không dùng nhãn relevance. Token
ngoài vocabulary được bỏ qua; nếu không còn token hợp lệ thì vector
bằng 0. Em xếp hạng bằng cosine và so sánh với TF-IDF trên cùng dữ liệu.
Chỉ những kết quả có điểm lớn hơn 0 mới được đưa vào danh sách truy hồi.

| Phương pháp | MRR | Mean Recall@3 |
| --- | --- | --- |
| tfidf | 0.8333 | 0.6389 |
| embedding | 1.0000 | 1.0000 |

MRR là trung bình nghịch đảo hạng của document liên quan đầu tiên.
Recall@3 là tỷ lệ document liên quan có trong Top-3, tính cho từng
query rồi lấy trung bình.

| Query | Relevant | TF-IDF Top-3 | Embedding Top-3 |
| --- | --- | --- | --- |
| medical treatment | med1, med2, med3 | med3 | med3, med2, med1 |
| vehicle repair | car1, car2 | (không có) | car2, car1, med3 |
| computer program | tech1, tech2 | tech1, tech2 | tech1, tech2, med3 |
| soccer match | sport1, sport2 | sport2 | sport2, sport1, tech2 |
| monetary savings | money1 | money1 | money1, med3, car2 |
| river shore | river1 | river1 | river1, car1, car2 |

Với query medical treatment, TF-IDF chỉ trả về med3 vì có từ treatment.
Embedding tìm được cả med1 và med2, chứa những từ như physician, therapy,
clinical, care và patient. Đây là một ví dụ tìm được văn bản liên quan
dù không khớp trực tiếp từ query.

Bộ document và query còn nhỏ và được soạn theo các chủ đề khá rõ.
Em chỉ xem đây là minh họa, chưa kết luận embedding luôn tốt hơn TF-IDF.
Muốn đánh giá kỹ hơn cần nhiều query và các document dễ nhầm lẫn hơn.

## 10. Đối chiếu dự đoán và kết luận

- Dự đoán doctor gần physician phù hợp với kết quả baseline: cosine
  0,7590 và physician là Top-1 của doctor.
- Tăng window làm similarity thay đổi theo từng cặp; không phải cặp
  nào cũng tăng. Kết quả phù hợp với dự đoán thứ hai.
- Tăng dimension làm model lớn hơn, còn chất lượng không tăng đều ở
  mọi metric. Kết quả phù hợp với dự đoán thứ ba.
- Dự đoán về corpus 100 câu chưa được kiểm chứng vì bài chưa chạy riêng
  trường hợp này.

Qua bài lab, em thấy context quyết định nhiều đến biểu diễn của từ.
Cosine cao có thể phản ánh gần nghĩa, cùng chủ đề hoặc context đặc thù
trong một số document. Khi chọn model, em cần xem cả kết quả tìm kiếm,
analogy, các trường hợp lỗi và chi phí chạy. Giới hạn với từ nhiều
nghĩa như bank được trình bày trong `reflection.md`.

Em mới thử một seed và một lần đo cho mỗi cấu hình. Năm cặp similarity,
tám câu analogy và sáu query chỉ đủ cho phạm vi bài lab, chưa phải một
đánh giá tổng quát.

## AI assistance statement

- **Tool:** OpenAI Codex / ChatGPT.
- **Purpose:** hỗ trợ xây dựng và chỉnh sửa bài, cài thư viện, chạy và kiểm tra thực nghiệm.
- **Generated content:** code, notebook, lời giải tính toán, dự đoán, diễn giải, phân tích lỗi và reflection.
- **Modified content:** phần trình bày được biên tập theo giọng sinh viên; bản này có hỗ trợ AI.
- **Verification:** số liệu được đối chiếu với kết quả chạy, tần suất và context của corpus; code đã được kiểm tra.
