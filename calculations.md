# LAB 03 - Lý thuyết và bài tính

Trong phần này, em trình bày cách đếm context và các bước tính cosine,
sau đó trả lời các câu hỏi về sparse, dense, CBOW, Skip-gram và analogy.

## 1. Distributional hypothesis

Em hiểu giả thuyết phân bố là: những từ xuất hiện trong các ngữ cảnh tương
tự thường có nghĩa hoặc cách sử dụng gần nhau. Ví dụ, trong hai câu
“the doctor treated the patient” và “the physician treated the patient”,
doctor và physician có chung các context treated, patient. Vì vậy, biểu
diễn dựa trên context có cơ sở để đặt hai từ gần nhau.

Đây là một xu hướng, không phải quy tắc tuyệt đối. Hai từ trái nghĩa cũng
có thể dùng trong cùng cấu trúc câu; cùng context không đồng nghĩa với
hoàn toàn cùng nghĩa.

## 2. Bài 1 - Word-context vectors, window = 1

Em sử dụng bốn câu trong Bài 1:

```text
the cat eats fish
the dog eats fish
the cat likes milk
the dog likes meat
```

Em dùng thứ tự vocabulary như phần ví dụ của đề:

```text
[cat, dog, eats, likes, fish, milk, meat]
```

Window = 1 nghĩa là xét một từ bên trái và một từ bên phải, không đi qua
biên câu. Từ the nằm ngoài vocabulary này nên không có cột tương ứng.
Em vẫn giữ vị trí của the khi xác định context.

- cat: gặp eats một lần và likes một lần. Hai lần gặp the không được đưa
  vào vector 7 chiều.
- dog: gặp eats một lần và likes một lần.
- eats: trong câu 1 gặp cat và fish; trong câu 2 gặp dog và fish. Do đó
  fish được đếm **hai lần**.
- likes: trong câu 3 gặp cat và milk; trong câu 4 gặp dog và meat.

| Target | cat | dog | eats | likes | fish | milk | meat |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| dog | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| eats | 1 | 1 | 0 | 0 | 2 | 0 | 0 |
| likes | 1 | 1 | 0 | 0 | 0 | 1 | 1 |

Vậy:

```text
v_cat   = [0, 0, 1, 1, 0, 0, 0]
v_dog   = [0, 0, 1, 1, 0, 0, 0]
v_eats  = [1, 1, 0, 0, 2, 0, 0]
v_likes = [1, 1, 0, 0, 0, 1, 1]
```

Nếu dùng toàn bộ vocabulary và thêm the ở đầu, cat và dog có thêm thành
phần 2; eats và likes có thêm thành phần 0. Hai cách đều cần nêu rõ
vocabulary. Trong cấu hình 7 chiều trên, cat và dog có cùng vector nên
cosine bằng 1. Kết quả cho thấy corpus này chưa có context đủ khác biệt
để phân biệt hai con vật, không có nghĩa chúng là cùng một loài.

## 3. Bài 2 - Cosine similarity

Với x = [1, 2, 1] và y = [2, 4, 2]:

```text
x · y = 1×2 + 2×4 + 1×2 = 12
||x|| = sqrt(1² + 2² + 1²) = sqrt(6)
||y|| = sqrt(2² + 4² + 2²) = sqrt(24) = 2sqrt(6)
cos(x,y) = 12 / (sqrt(6) × 2sqrt(6)) = 12/12 = 1
```

Em thấy y = 2x: hai vector cùng hướng nhưng khác độ dài. Cosine chỉ đo
hướng tương đối, nên vẫn bằng 1. Trong biểu diễn count, điều này có thể
xảy ra khi phân bố context giữ nguyên tỷ lệ nhưng một từ xuất hiện nhiều
hơn. Không thể từ đó khẳng định hai từ luôn có cùng nghĩa.

## 4. Bài 3 - Semantic similarity

**Dự đoán trước phép tính:** physician gần doctor hơn banana, vì physician
là từ gần nghĩa với doctor còn banana thuộc chủ đề trái cây.

Đặt d = [0.8, 0.1, 0.7], p = [0.7, 0.2, 0.8], b = [-0.2, 0.9, -0.1].

```text
d · p = 0.8×0.7 + 0.1×0.2 + 0.7×0.8
      = 0.56 + 0.02 + 0.56 = 1.14
||d||² = 0.64 + 0.01 + 0.49 = 1.14
||p||² = 0.49 + 0.04 + 0.64 = 1.17
cos(d,p) = 1.14 / sqrt(1.14×1.17)
         = 1.14 / sqrt(1.3338) ≈ 0.9871

d · b = 0.8×(-0.2) + 0.1×0.9 + 0.7×(-0.1)
      = -0.16 + 0.09 - 0.07 = -0.14
||b||² = 0.04 + 0.81 + 0.01 = 0.86
cos(d,b) = -0.14 / sqrt(1.14×0.86)
         = -0.14 / sqrt(0.9804) ≈ -0.1414
```

Vì 0.9871 > -0.1414, các vector cho trước phù hợp với dự đoán physician
gần doctor hơn. Cosine âm biểu thị hướng tạo góc lớn hơn 90 độ trong
không gian này; không có nghĩa doctor và banana là hai từ trái nghĩa.

## 5. Bài 4 - Sparse và dense

1. Vector word-context 10.000 chiều, chỉ 30 thành phần khác 0 là **sparse**.
   Tỷ lệ khác 0 là 30/10.000 = 0,003 = 0,3%; tỷ lệ bằng 0 là 99,7%.
2. Embedding 300 chiều, hầu hết thành phần khác 0 là **dense**.
3. Dense embedding có thể gom những pattern context tương tự vào một
   không gian nhỏ hơn. Hai từ không cần có chung một cột khác 0 trực tiếp
   vẫn có thể có hướng gần nhau nhờ các pattern đã học. Phép cosine trên
   vector ngắn cũng ít phép toán hơn so với duyệt một vector dài dạng dense.
4. Dense không chắc tốt hơn mọi bài toán. Tìm mã sản phẩm, tên riêng hoặc
   từ khóa chính xác có thể phù hợp với TF-IDF. Embedding có thể bỏ sót
   từ hiếm, phụ thuộc corpus và làm mất chi tiết. Vector sparse lưu bằng
   chỉ số và giá trị khác 0 có thể rất tiết kiệm; không được so bộ nhớ như
   thể luôn phải lưu đủ 10.000 giá trị.

Nếu V = 100.000, ma trận đầy đủ có 100.000² = 10¹⁰ phần tử. Float32 cần
khoảng 40 GB thập phân, float64 khoảng 80 GB, chưa tính overhead. Dạng
sparse giảm bộ nhớ nhưng vẫn phụ thuộc số cặp thực sự xuất hiện.
100.000 embedding 300 chiều float32 cần khoảng 120 MB cho riêng một
mảng vector; Word2Vec khi train còn có output vectors và cấu trúc khác.

## 6. CBOW và Skip-gram khác nhau thế nào?

- **CBOW:** input là các từ context, target là từ ở giữa. Model tổng hợp
  context để dự đoán target.
- **Skip-gram:** input là từ ở giữa, target là từng từ context. Một vị trí
  có thể sinh nhiều cặp huấn luyện.

Với “the cat eats fish”, window = 1, các ví dụ CBOW là:

| Input context | Target |
|---|---|
| [cat] | the |
| [the, eats] | cat |
| [cat, fish] | eats |
| [eats] | fish |

Các cặp Skip-gram (input, target context) là:

```text
(the, cat)
(cat, the)
(cat, eats)
(eats, cat)
(eats, fish)
(fish, eats)
```

Như vậy có 4 ví dụ CBOW và 6 cặp Skip-gram nếu dùng tất cả vị trí, không
padding, không loại stopwords và không subsampling. Ví dụ CBOW “the ___
eats fish” trong phần lý thuyết là minh họa chung; với window = 1 quanh
cat thì fish không thuộc context trực tiếp.

## 7. Bài tính analogy

```text
king - man + woman
= [8, 2, 7] - [5, 1, 5] + [5, 3, 5]
= [8-5+5, 2-1+3, 7-5+5]
= [8, 4, 7]
```

Hiệu woman - man = [0, 2, 0] là một hướng quan hệ trong ví dụ giả định.
Áp dụng hướng đó vào king cho [8, 4, 7], có thể tương ứng với quan hệ
king -> queen. Đề không cho vector queen nên em chỉ kết luận đây là
vector ứng viên, chưa thể xác nhận nearest neighbor thật sự là queen.
Trong embedding học được, em chưa thể gán mỗi chiều cho một thuộc tính
nghĩa riêng. Phép cộng trừ phản ánh một quan hệ giữa các vector, chưa đủ
để kết luận model hiểu quan hệ đó như con người.
