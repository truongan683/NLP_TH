# Dự đoán trước thực nghiệm

Em giữ các dự đoán ban đầu để đối chiếu với kết quả chạy. Confidence là
mức độ tin tưởng vào nhận định, không phải một xác suất tính từ dữ liệu.

## Prediction 1 - Những từ nào gần nhau nhất?

Các từ được xét: doctor, physician, hospital, banana và car.

- **Prediction:** em dự đoán doctor và physician gần nhau nhất. Hospital
  có thể gần doctor, còn banana và car có similarity thấp hơn với doctor.
- **Reason:** doctor và physician đều chỉ người khám chữa bệnh, nên có
  thể gặp trong các câu chứa patient, treatment hoặc hospital. Hospital
  cùng chủ đề y tế nhưng chỉ địa điểm. Banana và car thuộc các chủ đề khác.
- **Confidence:** 90% với nhận định về nghĩa từ; 65% với thứ hạng mà
  model sẽ trả về vì kết quả còn phụ thuộc corpus.

## Prediction 2 - Tăng window từ 2 lên 5

- **Prediction:** similarity sẽ thay đổi, nhưng không phải cặp nào cũng tăng.
- **Reason:** cửa sổ lớn giúp model thấy thêm context ở xa. Các context
  này có thể bổ sung thông tin về chủ đề, nhưng cũng có thể ít liên quan
  đến target và làm biểu diễn bớt đặc trưng.
- **Confidence:** 85%.

## Prediction 3 - Tăng dimension từ 50 lên 100 rồi 300

- **Prediction:** vector và model sẽ lớn hơn; thời gian train có thể
  tăng. Em không nghĩ chất lượng chắc chắn tăng theo cùng thứ tự.
- **Reason:** nhiều chiều giúp biểu diễn thêm thông tin, nhưng cần đủ
  dữ liệu để học. Với corpus nhỏ, các chiều bổ sung có thể không giúp
  ích cho similarity, analogy hoặc tìm kiếm.
- **Confidence:** 85% với nhận định về chi phí; 60% với dự đoán thứ hạng
  chất lượng của ba model.

## Prediction 4 - Corpus chỉ có 100 câu

- **Prediction:** doctor và physician không chắc có vector gần nhau.
- **Reason:** physician có thể chưa xuất hiện hoặc bị loại bởi min_count.
  Ngay cả khi cả hai từ có trong vocabulary, ít context vẫn có thể
  khiến vector chưa ổn định. Nếu cả hai xuất hiện nhiều trong các câu
  tương tự thì kết quả có thể tốt hơn.
- **Confidence:** 90%.

Em đối chiếu ba dự đoán đầu với thực nghiệm trong README. Bài này chưa
train riêng trên corpus 100 câu, nên dự đoán thứ tư chưa được kiểm chứng.
