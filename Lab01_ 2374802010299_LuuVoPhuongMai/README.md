# README – Lab01

## Công nghệ sử dụng
- **Python 3**
- **NumPy**: tạo mảng dữ liệu và dữ liệu giả lập
- **PyTorch**:
  - Tensor và các hàm khởi tạo tensor
  - Autograd (`requires_grad`, `backward`)
  - Cài đặt Gradient Descent thủ công

---

## Cách hoạt động
- Tạo và thao tác tensor trong PyTorch (`empty`, `zeros`, `ones`, `rand`)
- Chuyển đổi dữ liệu giữa NumPy và PyTorch, so sánh dùng chung bộ nhớ và sao chép dữ liệu
- Sử dụng cơ chế **autograd** để tính đạo hàm của các hàm số
- Áp dụng **Gradient Descent** để cập nhật tham số theo từng vòng lặp
- Xây dựng mô hình **Linear Regression** đơn giản trên dữ liệu giả lập và huấn luyện mô hình bằng Gradient Descent

---

## Kết quả
- Hiểu rõ cách PyTorch quản lý tensor và gradient
- Tính được đạo hàm và độ dốc của hàm số bằng autograd
- Quan sát được quá trình hội tụ của Gradient Descent
- Mô hình Linear Regression cho thấy loss giảm dần theo số vòng lặp, các tham số tiến gần giá trị mong muốn
