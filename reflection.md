# Reflection - Từ static đến contextual embeddings

## 1. So sánh các cách biểu diễn

Trong bảng này, context-dependent nghĩa là biểu diễn của một token có
thay đổi theo câu đang xét hay không. Điều đó khác với việc model được
học từ context trong toàn bộ corpus.

| Representation | Context-dependent? | Sparse / Dense | Một từ có nhiều vector theo lần xuất hiện? |
|---|---|---|---|
| TF-IDF thông thường | Không tạo token embedding theo câu; trọng số TF có thể đổi theo document | Sparse | Không theo nghĩa contextual; thường mỗi document có một vector |
| Co-occurrence | Không; context được tổng hợp trên corpus | Thường sparse | Một vector cho mỗi word type trong cùng corpus và window |
| Word2Vec | Không | Dense | Một vector cố định cho mỗi word type trong một model |
| Contextual embedding | Có | Thường dense | Có, tùy câu và lần xuất hiện |

TF-IDF có thể thay đổi trọng số của cùng một từ giữa các document,
nhưng điều đó chưa làm nó trở thành contextual word embedding. Trong
cách dùng thông thường, các chiều của vector document tương ứng với
vocabulary, còn giá trị là trọng số TF-IDF của từng từ.

## 2. Tại sao bank cần contextual representation?

Trong câu “I deposited money in the bank”, bank có nghĩa là ngân hàng.
Trong câu “We sat on the river bank”, bank có nghĩa là bờ sông. Em thấy
các từ deposited, money và river giúp xác định nghĩa phù hợp.

Word2Vec chỉ học một vector cho token bank từ tất cả những lần xuất
hiện của nó. Khi gặp một câu mới, model vẫn lấy vector đó, nên chưa
thể chọn riêng biểu diễn cho nghĩa ngân hàng hoặc bờ sông. Nếu một
nghĩa xuất hiện nhiều hơn, vector có thể phản ánh nghĩa đó rõ hơn.
Tăng số chiều không tự thay đổi cách tra một vector cố định này.

Trong corpus đã dùng, bank xuất hiện 250 lần. Cosine(bank, money) là
0,4683, cao hơn cosine(bank, river) là 0,1671. Có 12 câu đồng xuất hiện
bank/money và 1 câu bank/river. Doc 315 nói về gửi tiền khi không có
ngân hàng; doc 718 mô tả con đường đi theo bờ sông. Kết quả gợi ý ngữ
cảnh tài chính nổi bật hơn trong tập dữ liệu này. Tuy nhiên, số câu
đồng xuất hiện chưa phải nhãn nghĩa, nên em chưa dùng nó để tính tỷ lệ
chính xác của hai nghĩa.

Contextual embedding tính biểu diễn của token từ câu hiện tại. Với
Transformer, attention cho phép thông tin từ các từ khác tham gia vào
biểu diễn bank. Nhờ đó, bank trong câu có money có thể nhận vector khác
bank trong câu có river. Model vẫn có thể sai nếu dữ liệu hoặc ngữ cảnh
không đủ rõ, nhưng cách biểu diễn này phù hợp hơn với từ nhiều nghĩa.

Lấy trung bình các Word2Vec trong query cũng có thể làm “bank money”
khác “bank river”. Trong trường hợp đó, vector của query thay đổi vì
có thêm money hoặc river; vector của riêng bank vẫn giữ nguyên.

## 3. Điều em rút ra sau bài lab

Ở phần co-occurrence, cat và dog có vector giống nhau với window=1,
nhưng cosine giảm còn 0,875 khi window=2. Em hiểu rằng biểu diễn thay
đổi theo các context được đưa vào, chứ không chỉ theo nghĩa em biết sẵn
về hai từ.

Ở phần Word2Vec, physician gần doctor nhất, nhưng anesthesia cũng đứng
rất cao. Điều này giúp em phân biệt giữa gần nghĩa và cùng chủ đề.
Trường hợp banana gần tuscan còn cho thấy cần xem lại câu nguồn trước
khi giải thích một kết quả bất ngờ.

Tăng dimension làm model lớn hơn, còn accuracy Top-1 của analogy không
tăng đều. Vì vậy, khi chọn cấu hình, em cần xem kết quả của bài toán
đích và chi phí chạy. Bài mới có một seed, 5 cặp similarity, 8 câu
analogy và 6 query tự soạn, nên kết luận chỉ có phạm vi nhỏ.

## 4. Ôn tập individual learning check

**Distributional hypothesis là gì?** Những từ xuất hiện trong các
context tương tự thường có nghĩa hoặc cách sử dụng gần nhau. Đây là
một xu hướng; các từ cùng chủ đề hoặc trái nghĩa cũng có thể chia sẻ context.

**Tại sao doctor và physician có thể gần nhau?** Hai từ có thể xuất
hiện với patient, treatment, care và medical. Khi corpus cung cấp đủ
context tương tự, model có cơ sở để học các vector gần nhau.

**CBOW khác Skip-gram ở đâu?** CBOW dùng context để dự đoán từ ở giữa.
Skip-gram dùng từ ở giữa để dự đoán từng từ context.

**Tại sao tăng window vừa tốt vừa xấu?** Window rộng thu thêm thông tin
về chủ đề, nhưng cũng có thể lấy vào context xa và ít đặc trưng. Cần
kiểm tra trên nhiệm vụ cụ thể để biết thay đổi có ích hay không.

**Tại sao Word2Vec không phân biệt hai nghĩa của bank theo câu?** Nó
tra một vector cố định cho bank, không tính lại biểu diễn của bank từ
các từ trong câu hiện tại.

**Tại sao TF-IDF không phải word embedding theo cách dùng trong lab?**
TF-IDF biểu diễn document bằng các trọng số trên vocabulary. Nó không
trực tiếp học các dense word vectors dựa trên context để những từ liên
quan có thể nằm gần nhau.
