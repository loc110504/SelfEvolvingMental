# SelfEvolvingMental: PsyVEC Implementation

Offline-research implementation of **PsyVEC** (Self-Evolving Mental Health Assessment Agent) built from the characterized AgentMental baseline. 

> **Safety Notice:** This project is an offline research benchmark and is not an autonomous diagnostic system for live clinical use.

---

## 📑 Mục lục
1. [Cài đặt & Môi trường](#1-cài-đặt--môi-trường)
2. [Cơ chế Tự tiến hóa (Self-Evolution) & Tùy chọn Turn ON / OFF](#2-cơ-chế-tự-tiến-hóa-self-evolution--tùy-chọn-turn-on--off)
3. [Hướng dẫn chạy Đơn mẫu (Single Sample Assessment)](#3-hướng-dẫn-chạy-đơn-mẫu-single-sample-assessment)
4. [Hướng dẫn chạy Hàng loạt (Full Batch Evaluation)](#4-hướng-dẫn-chạy-hàng-loạt-full-batch-evaluation)
5. [Cấu hình Model: HuggingFace, vLLM, Ollama, API](#5-cấu-hình-model-huggingface-vllm-ollama-api)
6. [Evidence-Grounded Retrieval (Grounded Interview Loop)](#6-evidence-grounded-retrieval-grounded-interview-loop)
7. [Offline Verification & Testing](#7-offline-verification--testing)

---

## 1. Cài đặt & Môi trường

Yêu cầu: **Python 3.10+** (hỗ trợ macOS Apple Silicon `mps`, Linux `cuda`, CPU).

```bash
# Clone repository
git clone git@github.com:loc110504/SelfEvolvingMental.git
cd SelfEvolvingMental

# Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt toàn bộ dependencies cần thiết
pip install -r requirements.txt

# (Tùy chọn) Cài đặt package dưới dạng editable
pip install -e ".[all]"
```

---

## 2. Cơ chế Tự tiến hóa (Self-Evolution) & Tùy chọn Turn ON / OFF

PsyVEC thiết kế quá trình tự tiến hóa theo 2 trục hoàn toàn độc lập (Decoupled):

1. **Fast-Path (Lesson Memory / In-Context Update)**:
   - Module: `src/psyvec/lessons/core.py` (`LessonStore`).
   - Đúc kết các bài học hành vi quy trình (`do`, `avoid`, `trigger`, `criterion`) từ các ca phỏng vấn trước và tiêm vào prompt của agent ở các ca sau. **Không thay đổi trọng số mô hình**.
2. **Slow-Path (Parametric Model Update / LoRA)**:
   - Module: `src/psyvec/training/` và `src/psyvec/policy/`.
   - Huấn luyện lại trọng số mô hình (LoRA adapters) cho từng vai trò (`interviewer`, `scorer`, `updater`, `reporter`) qua các cặp dữ liệu tương phản (Preference pairs / DPO).

### Ma trận 4 chế độ thí nghiệm (Ablation / ON-OFF Study)

Trong `src/psyvec/evaluation/onoff.py`, hệ thống định nghĩa 4 cấu hình để đo lường hiệu quả độc lập:

| Cấu hình (Arm) | Lesson Update (Fast-Path) | Model Update (Slow-Path) | Mô tả |
|---|:---:|:---:|---|
| **`Initial-OFF`** (Zero-shot Baseline) | ❌ **OFF** | ❌ **OFF** | Mô hình gốc thuần túy, không dùng bài học, không adapter. |
| **`Initial-ON`** (Memory-only) |  **ON** | ❌ **OFF** | Mô hình gốc + tiêm bài học kinh nghiệm ngữ cảnh vào prompt. |
| **`Evolved-OFF`** (Model-only) | ❌ **OFF** |  **ON** | Mô hình đã tinh chỉnh trọng số (LoRA Adapter), tắt kho bài học. |
| **`Evolved-ON`** (Full PsyVEC) |  **ON** |  **ON** | Chạy đầy đủ cả 2: Mô hình đã cập nhật LoRA + Truy xuất bài học. |

---

## 3. Hướng dẫn chạy Đơn mẫu (Single Sample Assessment)

Script [`scripts/run_qwen_sample.py`](scripts/run_qwen_sample.py) chạy trọn vẹn quy trình đánh giá 1 bệnh nhân qua 8 chủ đề của thang đo **PHQ-8**:
1. Thu thập thông tin nhân khẩu học (tuổi) bằng regex trực tiếp trên transcript thật — không còn gọi LLM cho bước này (tránh bịa thông tin không có căn cứ).
2. **Truy xuất bằng chứng (evidence retrieval)** cho từng chủ đề từ **toàn bộ** `real_interview` (không giới hạn 50 lượt đầu như trước) — xem [mục 6](#6-evidence-grounded-retrieval-grounded-interview-loop).
3. Phỏng vấn qua đủ 8 chủ đề PHQ-8 (vẫn giữ nguyên cơ chế hỏi-đáp nhiều vòng để khai thác sắc thái cảm xúc/mức độ), nhưng "bệnh nhân ảo" giờ trả lời dựa trên bằng chứng thật đã truy xuất ở bước 2 thay vì tự do tưởng tượng; tự động hỏi thêm hoặc dừng theo tiêu chí `necessity` (mặc định khi không parse được giờ phụ thuộc bằng chứng có sẵn hay không, không còn là hằng số cố định).
4. Chấm điểm từng mục (0–3), viết cơ sở tóm tắt, kèm trích dẫn `evidence_turn_ids` về đúng lượt thoại thật đã dùng để chấm.
5. Tổng hợp điểm PHQ-8 (0–24), xếp loại mức độ trầm cảm, đối chiếu trực tiếp với Ground Truth và lưu báo cáo JSON (kèm `evidence_diagnostics` cho từng mẫu).

### Lệnh chạy:
```bash
# Chạy mẫu mặc định (Participant 302 trong tập dev)
python3 scripts/run_qwen_sample.py

# Chỉ định mẫu bệnh nhân cụ thể:
python3 scripts/run_qwen_sample.py --sample-path data/processed_daic_woz/dev/307.json
python3 scripts/run_qwen_sample.py --sample-path data/processed_daic_woz/train/300.json

# Đổi model HuggingFace khác:
python3 scripts/run_qwen_sample.py --model-name Qwen/Qwen2.5-7B-Instruct
```

### Danh sách tham số CLI của `run_qwen_sample.py`:
- `--sample-path` (Path): Đường dẫn tới file JSON mẫu DAIC-WOZ (mặc định: `data/processed_daic_woz/dev/302.json`).
- `--model-name` (str): Tên mô hình HuggingFace, checkpoint local, hoặc tên model trên server (mặc định: `Qwen/Qwen2.5-0.5B-Instruct`).
- `--api-base-url` (str): URL server tương thích OpenAI (vLLM, Ollama, v.v.).
- `--api-key` (str): API key (nếu server yêu cầu).
- `--output-dir` (Path): Thư mục lưu file JSON đánh giá (mặc định: `results/evaluations/`).

---

## 4. Hướng dẫn chạy Hàng loạt (Full Batch Evaluation)

Script [`scripts/run_batch_eval.py`](scripts/run_batch_eval.py) dùng để đánh giá tự động trên toàn bộ tập dữ liệu mẫu:
- **Tối ưu**: Nạp mô hình một lần duy nhất vào bộ nhớ để đánh giá liên tục nhiều mẫu.
- **Tính toán chỉ số tự động**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Pearson Correlation ($r$)
  - Binary Classification (ngưỡng PHQ-8 $\ge 10$): Accuracy, Precision, Recall, Macro/Binary F1-Score, Confusion Matrix.
  - **Evidence diagnostics** (mới): Evidence coverage rate (% chủ đề có bằng chứng thật tìm được), Citation-mismatch rate (% lần model trích dẫn `turn_id` không có thật), Low-faithfulness rate (% hội thoại có dấu hiệu lệch khỏi bằng chứng đã cấp) — xem [mục 6](#6-evidence-grounded-retrieval-grounded-interview-loop).
- **Lưu trữ**: Xuất bảng tổng kết ra màn hình, đồng thời lưu file chi tiết từng mẫu + file tổng hợp `batch_summary_<timestamp>.json` và `batch_summary_<timestamp>.csv`.

### Lệnh chạy mẫu:

```bash
# 1. Chạy thử 3 mẫu đầu tiên tập dev:
python3 scripts/run_batch_eval.py --num-samples 3 --data-dir data/processed_daic_woz/dev

# 2. Chạy full cả tập Development (35 mẫu), tự động bỏ qua mẫu đã có kết quả:
python3 scripts/run_batch_eval.py --data-dir data/processed_daic_woz/dev --skip-existing

# 3. Chạy full cả tập Training (107 mẫu):
python3 scripts/run_batch_eval.py --data-dir data/processed_daic_woz/train --skip-existing

# 4. Bật hiển thị chi tiết từng câu hỏi/đáp của các lượt phỏng vấn:
python3 scripts/run_batch_eval.py --data-dir data/processed_daic_woz/dev --num-samples 2 --verbose
```

### Danh sách tham số CLI của `run_batch_eval.py`:
- `--data-dir` (Path): Thư mục chứa các file JSON processed (mặc định: `data/processed_daic_woz/dev`).
- `--num-samples` (int): Số lượng mẫu tối đa muốn chạy (mặc định: chạy toàn bộ file trong thư mục).
- `--skip-existing` (flag): Bỏ qua mẫu nếu file kết quả `<id>_evaluation.json` đã tồn tại trong thư mục output (rất hữu ích khi tiếp tục chạy sau khi bị ngắt quãng).
- `--verbose` (flag): In toàn bộ hội thoại từng lượt ra màn hình terminal.
- `--model-name` (str): Model name (mặc định: `Qwen/Qwen2.5-0.5B-Instruct`).
- `--api-base-url` (str): URL endpoint OpenAI-compatible (vLLM/Ollama).
- `--api-key` (str): API key (nếu cần).
- `--output-dir` (Path): Thư mục lưu kết quả (mặc định: `results/evaluations/batch/`).

---

## 5. Cấu hình Model: HuggingFace, vLLM, Ollama, API

Hệ thống hỗ trợ 4 phương thức cấp phát mô hình LLM linh hoạt:

### A. Tải trực tiếp qua HuggingFace / PyTorch (Local)
Mặc định hệ thống tự động nhận diện thiết bị (`mps` cho Apple Silicon Mac, `cuda` cho GPU NVIDIA, hoặc `cpu`):

```bash
# Chạy với model siêu nhẹ (mặc định, 0.5B parameters)
python3 scripts/run_batch_eval.py --model-name Qwen/Qwen2.5-0.5B-Instruct

# Chạy với model lớn hơn (cần GPU có VRAM phù hợp)
python3 scripts/run_batch_eval.py --model-name Qwen/Qwen2.5-7B-Instruct
python3 scripts/run_batch_eval.py --model-name meta-llama/Llama-3.2-3B-Instruct

# Chạy với thư mục checkpoint local trên máy
python3 scripts/run_batch_eval.py --model-name /path/to/local/qwen_weights
```

---

### B. Kết nối với vLLM (Tối ưu tốc độ cao & Multi-GPU)

Nếu bạn có server GPU và muốn inference với throughput tối đa, hãy khởi chạy vLLM trước:

```bash
# 1. Khởi động vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16

# 2. Chạy evaluation kết nối qua vLLM endpoint
python3 scripts/run_batch_eval.py \
    --api-base-url http://localhost:8000/v1 \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --data-dir data/processed_daic_woz/dev
```

---

### C. Kết nối với Ollama (Chạy local trên máy cá nhân)

Ollama tích hợp sẵn OpenAI-compatible endpoint tại port `11434`:

```bash
# 1. Tải và chạy model trong Ollama
ollama run qwen2.5:7b

# 2. Chạy evaluation kết nối trực tiếp đến Ollama
python3 scripts/run_batch_eval.py \
    --api-base-url http://localhost:11434/v1 \
    --model-name qwen2.5:7b \
    --data-dir data/processed_daic_woz/dev
```

---

### D. Kết nối qua API Cloud bên ngoài (OpenAI, DeepSeek, OpenRouter)

```bash
# Ví dụ chạy với DeepSeek API:
python3 scripts/run_batch_eval.py \
    --api-base-url https://api.deepseek.com/v1 \
    --api-key sk-your-deepseek-api-key \
    --model-name deepseek-chat \
    --data-dir data/processed_daic_woz/dev --num-samples 5
```

---

## 6. Evidence-Grounded Retrieval (Grounded Interview Loop)

Trước đây, "bệnh nhân ảo" chỉ được cấp 50 lượt thoại đầu tiên của `real_interview` làm bối cảnh (thường toàn small-talk mở đầu), nên khi được hỏi về từng chủ đề PHQ-8, model phải **tự bịa** câu trả lời — không neo vào bất kỳ điều gì participant thật sự đã nói. Điều này đã được sửa (xem [`Plan_Improve.md`](Plan_Improve.md)):

- **Retrieval layer** (`src/psyvec/evaluation/evidence_retrieval.py`) quét **toàn bộ** `real_interview` của từng participant, tìm bằng chứng thật cho mỗi chủ đề bằng 2 cơ chế:
  1. **Tag-anchor**: Ellie (interviewer ảo trong DAIC-WOZ) đôi khi để lộ tag giao thức nội bộ dạng `easy_sleep (how easy is it...)` — các tag này được map thủ công sang 8 chủ đề PHQ-8 trong [`configs/scales/topic_tag_map.json`](configs/scales/topic_tag_map.json).
  2. **Keyword fallback**: với 5/8 chủ đề không có tag trực tiếp (Loss of Interest, Fatigue, Appetite, Concentration, Psychomotor), quét từ khóa trong [`configs/scales/topic_keywords.json`](configs/scales/topic_keywords.json) trên toàn bộ lượt của Participant.
  3. Các tag hỏi về tâm trạng/tiền sử chung (chẩn đoán trầm cảm, đi trị liệu...) được gom vào bucket `CROSS_CUTTING`, dùng làm "nền cảm xúc" khi một chủ đề không có bằng chứng riêng.
- **Grounded Interview Loop** (`src/psyvec/evaluation/interview_policy.py`): system prompt của "bệnh nhân ảo", tiêu chí `necessity` mặc định, câu hỏi follow-up, và prompt của scorer/Memory-Update giờ đều được build từ bằng chứng đã truy xuất — vẫn giữ nguyên số vòng hỏi-đáp như trước, chỉ khác là mọi câu trả lời phải nhất quán với bằng chứng thật (hoặc, nếu không có bằng chứng riêng, phải trả lời thận trọng theo tâm trạng chung, không được bịa triệu chứng cụ thể mới).

### Tinh chỉnh lexicon/tag-map (chỉ làm trên tập `train`, không tune trên `dev`/`test`)

```bash
# Liệt kê toàn bộ tag Ellie tìm được trong 1 split (mặc định: train)
python3 scripts/build_tag_inventory.py --data-dir data/processed_daic_woz/train

# Xem từng dòng thoại thật mà 1 từ khóa trong topic_keywords.json khớp phải,
# dùng để phát hiện từ khóa gây nhiễu (vd "down" khớp nhầm "downtown")
python3 scripts/audit_keyword_lexicon.py --data-dir data/processed_daic_woz/train
```

Sau khi sửa `configs/scales/topic_tag_map.json` hoặc `topic_keywords.json`, chạy lại `scripts/run_batch_eval.py` trên `dev` và so 3 chỉ số evidence diagnostics (mục 4) với lần chạy trước để biết thay đổi có cải thiện độ bao phủ bằng chứng hay không.

---

## 7. Offline Verification & Testing

Đảm bảo mã nguồn PsyVEC vượt qua toàn bộ 232 bài kiểm thử và kiểm tra tĩnh:

```bash
# Chạy bộ unit và integration test (232 tests, 6 bị skip do thiếu thư mục
# baseline AgentMental — không liên quan lỗi code)
python3 -m unittest discover -s tests -p 'test_*.py'

# Hoặc dùng pytest
pytest

# Kiểm tra linting và style code
ruff check src tests/unit

# Kiểm tra type hinting nghiêm ngặt
mypy --strict src/psyvec
```
