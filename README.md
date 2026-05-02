# AI Safe Post - Nội dung Kiểm duyệt Đa nhiệm (PhoBERT)

Hệ thống AI kiểm duyệt nội dung tự động dựa trên mô hình ngôn ngữ PhoBERT, được thiết kế đặc biệt cho tiếng Việt. Dự án này cung cấp API để phát hiện các vi phạm tiêu chuẩn cộng đồng (bắt nạt, quấy rối, xúc phạm) và tự động trích xuất các từ khóa vi phạm.

## 🚀 Tính năng chính

- **Phân loại nội dung**: Xác định văn bản có vi phạm tiêu chuẩn cộng đồng hay không với độ chính xác cao.
- **Trích xuất từ khóa**: Tự động nhận diện các từ ngữ nhạy cảm, xúc phạm hoặc gây hận thù trong câu.
- **Độ tự tin AI**: Cung cấp xác suất vi phạm và mức độ tin cậy của mô hình cho mỗi dự đoán.
- **Tự động tải Model**: Tự động kết nối và tải weights từ Google Drive khi khởi động nếu chưa có sẵn.

## 🛠️ Công nghệ sử dụng

- **Mô hình**: [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2)
- **Framework**: Flask, PyTorch, Transformers
- **Triển khai**: Gunicorn, Render, Docker/Blueprint hỗ trợ.

## 📡 API Endpoints

### 1. Kiểm duyệt nội dung
- **Endpoint**: `POST /api/moderation`
- **Body**:
  ```json
  {
    "text": "nội dung cần kiểm tra"
  }
  ```
- **Response**:
  ```json
  {
    "is_violation": true,
    "label": "Vi phạm",
    "violation_score": 98.5,
    "confidence": 99.2,
    "keywords": ["từ_khóa_1", "từ_khóa_2"]
  }
  ```

### 2. Kiểm tra trạng thái
- **Endpoint**: `GET /`
- **Response**: JSON chứa trạng thái server và thông tin dịch vụ.

## 💻 Cài đặt Local

1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy ứng dụng:
   ```bash
   python app.py
   ```
   *Lưu ý: Lần đầu chạy sẽ tự động tải model (~600MB) về thư mục `model/`.*

## 🌐 Triển khai lên Render

Dự án đã được cấu hình sẵn cho **Render Blueprint**:
1. Đẩy code lên GitHub.
2. Tại Render Dashboard, chọn **New → Blueprint**.
3. Kết nối với Repo này và nhấn **Apply**.

Hoặc cài đặt thủ công **Web Service**:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300`
- **Environment Variable**: Thêm `PYTHON_VERSION: 3.11.8`.