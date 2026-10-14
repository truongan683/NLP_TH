# Error analysis - Phân tích kết quả similarity

Em chọn ba kết quả phù hợp và ba kết quả bất ngờ của model Skip-gram
100 chiều, window=5, 8 epochs. Khi đọc kết quả, em phân biệt giữa gần
nghĩa và cùng chủ đề. Cosine cao chưa đủ để kết luận hai từ đồng nghĩa.

## 1. Kết quả phù hợp: doctor - physician

- **Observed:** cosine = 0,7590; physician đứng Top-1 của doctor.
- **Expected:** em kỳ vọng hai từ gần đồng nghĩa này có similarity cao.
- **Possible explanation:** cả hai đều chỉ người khám chữa bệnh và
  xuất hiện trong các câu về y học, chăm sóc sức khỏe.
- **Evidence from corpus:** doctor xuất hiện 204 lần trong 89 document;
  physician xuất hiện 65 lần trong 40 document. Doc 129 có family
  physician, clinical practice và attending physician. Doc 282 có
  doctor trong một câu về skincare clinic và treatment. Các context
  chung có care (7), medical (2), consult (2) và medicine (2).

Em thấy kết quả phù hợp với cả nghĩa từ và các ví dụ trong corpus.
Tuy nhiên, số lần gặp context chung là thống kê của corpus, không phải
bằng chứng về nguyên nhân duy nhất tạo ra điểm cosine.

## 2. Kết quả phù hợp: disease - chronic

- **Observed:** cosine = 0,8041; chronic đứng Top-1 của disease.
- **Expected:** em kỳ vọng hai từ có liên quan về bệnh mạn tính nằm gần
  nhau, dù chúng không đồng nghĩa.
- **Possible explanation:** chronic thường mô tả bệnh hoặc tình trạng
  sức khỏe kéo dài, nên hai từ có thể chia sẻ context về bệnh lý.
- **Evidence from corpus:** disease xuất hiện 233 lần trong 97 document;
  chronic xuất hiện 60 lần trong 37 document. Doc 163 có chronic kidney
  illness và các thuật ngữ về kidney. Context chung có patients (10),
  pulmonary (5), diseases (5), kidney (4) và heart (4).

Chronic là tính từ, disease là danh từ. Em xem đây là kết quả phù hợp
về chủ đề, không phải hai từ có thể thay thế cho nhau trong câu.

## 3. Kết quả phù hợp: football - basketball

- **Observed:** cosine = 0,7583; basketball đứng Top-1 của football.
- **Expected:** em nghĩ hai môn thể thao có thể gần nhau về chủ đề.
- **Possible explanation:** các tin thể thao thường nói về đội, cầu thủ,
  trận đấu, trường học hoặc huấn luyện viên.
- **Evidence from corpus:** football xuất hiện 162 lần trong 70 document;
  basketball xuất hiện 62 lần trong 38 document. Doc 154 là tin về một
  trận basketball; doc 173 nói về chấn thương của cầu thủ. Context chung
  có player (8), college (6), team (5), game (5) và coach (4).

Hai môn có luật và cách chơi khác nhau, nhưng dùng nhiều từ cùng nhóm
chủ đề. Em thấy cosine cao trong trường hợp này là hợp lý.

## 4. Kết quả bất ngờ: doctor - anesthesia

- **Observed:** cosine = 0,7328; anesthesia đứng thứ 2 trong Top-5 doctor,
  cao hơn dentist và surgeon.
- **Expected:** nếu tìm từ gần nghĩa hoặc chức danh, em kỳ vọng tên
  nghề đứng trước một phương pháp y tế.
- **Possible explanation:** các context về thủ thuật, bệnh nhân và
  phòng khám có thể làm hai vector gần nhau. Anesthesia xuất hiện ít,
  nên một số bài viết y tế có thể ảnh hưởng nhiều đến biểu diễn.
- **Evidence from corpus:** doctor xuất hiện 204 lần trong 89 document;
  anesthesia chỉ có 20 lần trong 8 document. Doc 1229 nói về anesthesia,
  patients, medical procedures, operations và health care professional.
  Các context chung có local (2), clinic (1), patients (1) và surgery (1).

Anesthesia có liên quan đến công việc của bác sĩ, nên kết quả không
phải hoàn toàn vô nghĩa. Điều bất ngờ là model xếp một từ chỉ phương
pháp gần hơn một số từ chỉ nghề.

## 5. Kết quả bất ngờ: hospital - chongqing

- **Observed:** cosine = 0,6414; chongqing đứng thứ 2 trong Top-5 hospital.
- **Expected:** em kỳ vọng hospital gần clinic hoặc một cơ sở y tế hơn
  là tên một thành phố.
- **Possible explanation:** với từ hiếm, context trong tên tổ chức có
  thể chi phối mạnh biểu diễn.
- **Evidence from corpus:** hospital có 234 lần trong 106 document;
  chongqing chỉ có 7 lần trong 2 document. Doc 5052 chứa cụm first
  affiliated hospital of chongqing medical university hai lần trong
  một đoạn cơ quan. Context chung có medical (4), university (4) và
  affiliated (2). Doc 738 lại dùng Chongqing trong chủ đề phát triển đô thị.

Em nghĩ tên bệnh viện và đại học y là một lý do có thể giải thích kết
quả. Em chưa bỏ đoạn này rồi train lại, nên chưa khẳng định được ảnh
hưởng thực sự của nó.

## 6. Kết quả bất ngờ: banana - tuscan

- **Observed:** cosine = 0,7041; tuscan đứng Top-1 của banana.
- **Expected:** em kỳ vọng banana gần các loại quả hoặc từ về trái cây
  hơn một tính từ chỉ vùng hoặc phong cách.
- **Possible explanation:** các câu về ẩm thực có thể góp phần tạo
  quan hệ giữa hai từ, nhưng dữ liệu cho Tuscan rất ít.
- **Evidence from corpus:** banana có 31 lần trong 17 document; tuscan
  chỉ có 5 lần trong 3 document. Doc 15 và 762 có banana bread, còn
  doc 3418 có tuscan garlic, kitchen và olive oil. Doc 400 nói về một
  địa điểm chụp ảnh cưới theo phong cách Tuscan; doc 5283 nói về địa
  điểm thuê nhà.

Sau khi bỏ stopwords ở bước hiển thị, phép đếm radius=5 không tìm thấy
context trực tiếp chung cho cặp này. Vì vậy, giải thích về chủ đề ẩm
thực chỉ là một giả thuyết từ các câu nguồn. Em cần thêm dữ liệu hoặc
thử nhiều seed để kiểm tra, chưa thể kết luận chắc chắn.

## 7. Nhận xét chung

Qua sáu trường hợp, em thấy cosine cao có thể phản ánh gần nghĩa, cùng
chủ đề hoặc context riêng của một tên tổ chức. Chỉ đọc Top-5 chưa đủ;
em cần xem thêm tần suất và câu nguồn.

Các context chung ở trên được đếm trong corpus gốc với radius=5, dùng
min(count_a, count_b). Stopwords và các mảnh s/t/m/re/ve/ll/d chỉ bị bỏ
ở bước hiển thị, không bị bỏ khỏi dữ liệu train. Đây không phải các
negative samples hoặc context sau subsampling của Gensim.

Nếu làm tiếp, em sẽ thử corpus lớn hơn, nhiều seed và kiểm tra ảnh
hưởng của các đoạn tên cơ quan bị lặp. Những cách này chưa được chạy
trong bài, nên em chưa kết luận chúng sẽ cải thiện kết quả.
