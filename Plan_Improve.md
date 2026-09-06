# Plan Improve: Evidence-Grounded Interactive PHQ-8 Screening

Trạng thái: **Phase 1 steps 1-5 implemented (code + unit/integration tests), chờ review.** Steps 6-7 (chạy thật trên `train`/`dev` để tune lexicon và so metric với baseline) cần môi trường có model/API thật (torch+transformers hoặc `--api-base-url`), chưa chạy được trong môi trường review này — xem "Tình trạng triển khai" cuối mục 6.
Phạm vi: sửa lại pipeline eval hiện tại (`scripts/run_qwen_sample.py` + `scripts/run_batch_eval.py`) để chấm điểm PHQ-8 bám sát transcript thật, **đồng thời giữ nguyên cơ chế hỏi-đáp nhiều vòng (interviewer ↔ client) đang có**, vì cơ chế này cho phép khai thác sắc thái cảm xúc/mood/mức độ trầm cảm giống một buổi phỏng vấn lâm sàng thật hơn là chỉ trích xuất câu chữ thô. Đồng thời nối vào kiến trúc typed đã có sẵn trong `src/psyvec/` (hiện chưa được dùng).

> **Thay đổi so với v1**: v1 đề xuất bỏ hẳn client-roleplay, chuyển sang chấm điểm trực tiếp từ evidence trích xuất (Evidence-Grounded Topic Scoring — không còn hội thoại). Theo phản hồi, hướng đó bị loại vì làm mất khả năng khai thác sâu (follow-up hỏi mức độ/tần suất, sắc thái cảm xúc) mà một buổi phỏng vấn thật có. **v2 giữ nguyên vòng lặp hỏi-đáp (client + necessity-check + follow-up) nhưng bơm grounding vào từng bước** để nó không còn bịa nữa, thay vì loại bỏ nó.

---

## 0. Recap nguyên nhân gốc (bối cảnh cho plan này)

Đã xác nhận bằng dữ liệu thật (participant 302, 346, 404, 440...):

1. `run_qwen_sample.py:262` cắt `real_interview[:50]` — trung bình interview dài 266 lượt (min 119, max 466), nội dung liên quan triệu chứng PHQ-8 thường nằm sau lượt 50. → "client ảo" bịa câu trả lời không liên quan transcript thật (vd 404: GT=0 nhưng model tự bịa "trouble staying asleep" trong khi transcript thật nói "handling my sleep pretty well" ở lượt 145 — ngoài phạm vi 50 lượt).
2. Không có bước retrieval nối câu hỏi PHQ-8 ↔ đoạn hội thoại thật liên quan — client-persona không có gì để "nhớ" khi trả lời theo từng topic.
3. `max_depth=2`, follow-up sinh ra dựa trên câu trả lời **đã bịa** ở vòng trước → khuếch đại lỗi thay vì đào sâu đúng hướng.
4. Necessity-parse-fail mặc định "dừng hỏi" (hằng số cứng, không phân biệt topic có bằng chứng hay không) → giảm bằng chứng đúng lúc cần nhất.
5. Bước Memory-Update chỉ thấy `dialogue_transcript` giả lập, không thấy `real_interview` → không sửa được sai số ban đầu, đôi khi khuếch đại thêm.
6. Kiến trúc typed (`src/psyvec/state`, `roles`, `memory`, `research/splits.py`) đã có sẵn đúng các ràng buộc cần thiết (evidence bắt buộc phải có `source_turn_ids`, split protocol chống leak) nhưng **chưa được nối vào** eval pipeline thực tế.

Mục tiêu của plan này: **giữ trải nghiệm phỏng vấn nhiều vòng** (để đào sâu emotion/mood/severity như thật) nhưng **loại bỏ nguồn hallucination**: mọi phát ngôn của "client" phải được neo (grounded) vào bằng chứng trích xuất trực tiếp từ `real_interview` của chính participant đó, chứ không phải tự do tưởng tượng từ một đoạn small-talk chung chung.

---

## 1. Nguyên tắc thiết kế

- **Grounding trước, roleplay sau, không phải grounding thay cho roleplay.** Client-persona vẫn được hỏi-đáp nhiều vòng, nhưng mỗi câu trả lời phải bắt nguồn từ evidence thật của đúng topic đang hỏi (hoặc, khi topic không có evidence trực tiếp, từ tín hiệu cảm xúc/mood tổng quát — cross-cutting evidence — chứ không phải bịa từ hư không).
- **"Không có bằng chứng cụ thể" ≠ "không hỏi".** Vẫn hỏi đủ 8 topic như một buổi phỏng vấn thật (giữ tính tự nhiên), nhưng khi topic không có evidence riêng, system prompt của client phải nói rõ ràng: trả lời **thận trọng, nhất quán với tâm trạng tổng thể đã biết**, **không được bịa thêm triệu chứng cụ thể mới** không có căn cứ.
- **Follow-up phải nhắm đúng cái còn thiếu để chấm điểm 0-3**, không phải hỏi mở vô định hướng. PHQ-8 rubric cần "tần suất/số ngày trong 2 tuần" — nếu câu trả lời gốc (dù đã grounded) chưa có thông tin tần suất, follow-up phải hỏi cụ thể điều đó, dựa trên evidence đã có chứ không tạo evidence mới.
- **Scorer phải nhìn thấy cả 2 nguồn**: (a) hội thoại grounded (client + follow-up) để lấy sắc thái/mức độ, và (b) evidence gốc (trích dẫn `turn_id` thật) để đối chiếu — chấm điểm phải cite được evidence, không chỉ dựa vào lời client đã qua "diễn dịch" của roleplay.
- **Ít lệnh gọi lãng phí hơn, không nhất thiết ít lệnh gọi hơn tổng thể.** Vì vẫn giữ hội thoại nhiều vòng, số lệnh gọi LLM sẽ không giảm mạnh như v1 — hiệu quả đến từ: prompt ngắn/đúng trọng tâm hơn (evidence snippet thay vì 50 lượt small-talk), giảm follow-up thừa khi evidence đã đủ rõ ràng, và song song hoá (Phase 3), chứ không phải cắt bỏ vòng hỏi-đáp.
- **Tôn trọng split protocol**: mọi việc "tinh chỉnh" (từ khóa, ngưỡng, prompt) chỉ được làm trên `train` (107 mẫu); `dev` (35 mẫu) dùng để đánh giá/so sánh phương án; `test` (47 mẫu, đáp án ở `full_test_split.csv`) chỉ chạy đúng 1 lần ở cuối — đúng tinh thần `src/psyvec/research/splits.py`.
- **Không phá vỡ phần đang hoạt động tốt**: giữ nguyên `src/psyvec/evaluation/response_parsing.py` (strict JSON/score parser) và `configs/scales/*.json` (thang đo + rubric 0-3 đã chuẩn, không cần sửa).

---

## 2. Kiến trúc mới: Evidence-Grounded Interactive Assessment (EGIA)

### 2.1 So sánh Old vs New

| | Old (hiện tại) | New (EGIA) |
|---|---|---|
| Ngữ cảnh nền cho client-persona | 50 lượt đầu (thường toàn small-talk, không liên quan) | Evidence bundle theo từng topic, trích từ **toàn bộ** `real_interview` |
| Cơ chế hỏi-đáp nhiều vòng (necessity + follow-up) | **Giữ** nhưng dựa trên câu trả lời đã bịa → khuếch đại lỗi | **Giữ nguyên cấu trúc**, nhưng mỗi vòng đều bám evidence + rubric-gap (thiếu tần suất/mức độ thì hỏi đúng cái đó) |
| Khi topic không có evidence cụ thể | Vẫn hỏi, client tự bịa tự do | Vẫn hỏi (giữ trải nghiệm phỏng vấn), nhưng client bị ràng buộc "trả lời thận trọng theo tâm trạng tổng thể, không bịa triệu chứng cụ thể mới" |
| Cơ sở chấm điểm | Chỉ `topic_history` (hội thoại giả lập) | `topic_history` (hội thoại grounded) **+** evidence gốc có `turn_id`, bắt buộc cite khi chấm |
| Bước "update" | Nhìn dialogue giả lập → khuếch đại lỗi | Nhìn cả dialogue grounded + evidence gốc toàn cục → review chéo hợp lệ |
| Số lệnh gọi LLM / participant | ~26–50 | ~24–46 (tương đương, không cắt hội thoại — xem §4) |

### 2.2 Pipeline mới (per participant, per topic)

```
real_interview (full, không cắt [:50])
        │
        ▼
[1] Retrieval Layer (rule-based, không tốn LLM call) — GIỐNG v1, vẫn cần làm trước tiên
      1a. Tag-anchor match  (Ellie "tag (câu hỏi)" → topic)
      1b. Keyword lexicon match (fallback, quét toàn bộ lượt Participant)
      1c. Cross-cutting evidence (depression_diagnosed, feel_down, therapy_*, symptoms_*...)
          — dùng làm "tâm trạng nền" khi 1a/1b không ra evidence riêng cho topic
        │
        ▼  → EvidenceBundle theo 8 topic: {status: known|missing, snippets: [...], cross_cutting: [...]}
        │
        ▼
[2] Grounded Interview Loop (giữ cấu trúc vòng lặp hiện tại, PER TOPIC)
      2a. Client-persona system prompt = EvidenceBundle[topic] (+ cross_cutting nếu missing)
          + rule cứng: "chỉ trả lời dựa trên trích dẫn bên dưới; nếu topic không có
          trích dẫn riêng, trả lời nhất quán với tâm trạng tổng thể, KHÔNG bịa
          triệu chứng cụ thể mới"
      2b. Necessity-check: giữ, nhưng default khi parse-fail phụ thuộc evidence status
          (status=known & chưa có thông tin tần suất → default tiếp tục hỏi;
           status=missing → default dừng)
      2c. Follow-up question: giữ, nhưng target rubric-gap cụ thể (tần suất/mức độ/
          thời lượng) thay vì hỏi mở tự do
        │
        ▼  → topic_history (grounded dialogue) + EvidenceBundle[topic]
        │
        ▼
[3] Scorer (per topic, giữ 8 lệnh như hiện tại)
      Input: topic_history (grounded) + EvidenceBundle[topic] (trích dẫn turn_id gốc)
             + rubric 0-3 (configs/scales/scoring_standards.json, không đổi)
      Output: {score, evidence_turn_ids, summary} — evidence_turn_ids phải là tập con
              của turn_id đã gửi (reject nếu bịa)
        │
        ▼
[4] Cross-topic Consistency Review (Memory-Update, giữ 1 lệnh)
      Input: toàn bộ topic_history + EvidenceBundle (mọi topic) + điểm sơ bộ
      Output: revised scores — chỉ được sửa nếu trích được evidence_slot_ids hỗ trợ
        │
        ▼
[5] Report (giữ cấu trúc output JSON hiện tại, bổ sung provenance: evidence_turn_ids
    dùng ở từng bước, evidence_status từng topic)
```

### 2.3 Retrieval Layer — chi tiết (không đổi so với v1, vẫn là nền tảng bắt buộc)

**(a) Tag-anchor mapping.** Đã quét toàn bộ `dev` set và tìm được 146 tag Ellie dạng `tag_name (câu hỏi/câu nói)` (vd `easy_sleep -> how easy is it for you to get a good night's sleep`). Đây là "protocol tag" cố định của DAIC-WOZ — **phải build lại inventory này trên `train` set** (không chỉ dev, để tránh bias) rồi map thủ công sang 8 topic. Sơ bộ rút ra được từ dev:

| PHQ-8 Topic | Ellie tag ứng viên (tìm thấy) |
|---|---|
| Depressed Mood | `feel_down`, `feel_lately`, `how_doingV` |
| Sleep Problems | `easy_sleep`, `sleep_affects` |
| Low Self-Worth | `feelbadly`, `feelguilty`, `self_change`, `too_hard` |
| Loss of Interest | *(không có tag trực tiếp — dựa vào keyword fallback)* |
| Fatigue or Low Energy | *(không có tag trực tiếp — dựa vào keyword fallback)* |
| Appetite or Weight Changes | *(không có tag trực tiếp — dựa vào keyword fallback)* |
| Concentration Difficulties | *(không có tag trực tiếp — dựa vào keyword fallback)* |
| Psychomotor Changes | *(không có tag trực tiếp — dựa vào keyword fallback)* |
| Cross-cutting (tâm trạng nền, dùng khi topic missing evidence riêng) | `depression_diagnosed`, `disturbing_thoughts`, `symptoms_what`, `symptoms_cope`, `suspect_problem`, `when_diagnosed`, `why_seek_help`, `bouts_symptoms`, `behavior_changes`, `trigger`, `therapist_*`, `therapy_*` |

→ **Action item**: `scripts/build_tag_inventory.py` quét `data/processed_daic_woz/train/*.json`, liệt kê tag + tần suất, review thủ công để chốt `configs/scales/topic_tag_map.json`. 5/8 topic không có tag trực tiếp → bắt buộc phải có keyword fallback + cross-cutting fallback tốt.

**(b) Keyword lexicon fallback.** Với mỗi topic, quét *toàn bộ* lượt `Participant` (không giới hạn) tìm câu chứa từ khóa liên quan, lấy kèm 1 lượt liền trước/sau làm ngữ cảnh:

| Topic | Từ khóa gốc (mở rộng dần bằng lỗi thực tế trên train) |
|---|---|
| Loss of Interest | interest, enjoy, hobby, motivat, bored, boring, care about |
| Depressed Mood | down, sad, hopeless, depress, cry, blue, low mood |
| Sleep Problems | sleep, insomnia, wake up, rest, tired at night, nap |
| Fatigue or Low Energy | tired, exhaust, energy, fatigue, drained, worn out |
| Appetite or Weight Changes | eat, appetite, weight, meal, hungry, food |
| Low Self-Worth | failure, worthless, guilt, blame myself, disappoint, inadequate |
| Concentration Difficulties | concentrat, focus, distract, forget, foggy, can't think |
| Psychomotor Changes | slow, restless, fidget, pace, sluggish, agitat |

**(c) Cross-cutting evidence bucket (MỚI so với v1, quan trọng cho hybrid).** Vì vẫn phải hỏi đủ 8 topic kể cả khi không có evidence riêng, cần một bucket riêng gom các tín hiệu tâm trạng/tổng quát (tag `depression_diagnosed`, `feel_down`, `therapy_*`, `symptoms_*`...) để làm "nền cảm xúc" cho client-persona trả lời các topic thiếu evidence cụ thể — thay vì client hoàn toàn không có gì để dựa vào (như hiện tại).

**(d) Semantic fallback (Phase 2, optional):** nếu sau khi đánh giá trên `train`/`dev` mà recall của (a)+(b) vẫn thấp cho 5 topic không có tag, bổ sung embedding similarity (vd `sentence-transformers/all-MiniLM-L6-v2`, CPU) — chỉ làm sau khi đo được lexicon thiếu ở đâu.

**(e) Evidence cap.** Mỗi topic tối đa N snippet cụ thể (đề xuất N=6) + tối đa M snippet cross-cutting (đề xuất M=3) để prompt gọn.

### 2.4 Grounded Interview Loop — chi tiết (đây là phần thay đổi cốt lõi so với v1)

Giữ nguyên vòng lặp `while depth < max_depth` hiện có trong `run_qwen_sample.py:336-409`, nhưng sửa từng phần:

- **Client system prompt** (`run_qwen_sample.py:267-273`): bỏ `interview_history` (50 lượt small-talk), thay bằng `EvidenceBundle[topic]` đã retrieve (snippet cụ thể + turn_id) và, nếu `status=missing`, thêm cross-cutting bucket. Kèm rule cứng bằng tiếng Anh trong system prompt, ví dụ ý tưởng (sẽ viết chính xác lúc code):
  - Nếu `status=known`: "Answer strictly consistent with the quotes below from your own words earlier in this interview. Do not contradict them."
  - Nếu `status=missing`: "You have not explicitly discussed this topic yet. Answer briefly and conservatively, consistent with your overall mood shown below. Do not invent specific new symptoms."
- **Necessity-check default** (`run_qwen_sample.py:384`): thay hằng số `0` bằng default phụ thuộc `EvidenceBundle[topic].status` và nội dung `topic_history` đã có (có nhắc tần suất/mức độ chưa — check bằng regex nhẹ trước khi hỏi LLM, không chỉ dựa vào LLM necessity call). Cụ thể: nếu status=known nhưng chưa có từ chỉ tần suất ("every day", "sometimes", "a few days"...) trong `topic_history` → default tiếp tục hỏi (1) thay vì dừng (0) khi parse fail; nếu status=missing → default dừng (0) như cũ (không có gì để đào sâu thêm).
- **Follow-up question generation** (`run_qwen_sample.py:391-408`): thêm bước kiểm tra rubric-gap (regex/heuristic: có thiếu thông tin tần suất/mức độ/thời lượng theo đúng 4 mốc của `scoring_standards.json` không) trước khi gọi LLM sinh follow-up; nếu thiếu, ép prompt follow-up hỏi đúng khía cạnh đó (vd "Hỏi cụ thể số ngày trong 2 tuần qua"), thay vì để LLM tự chọn hướng hỏi mở.
- **max_depth**: giữ nguyên cấu trúc tham số, nhưng thử nghiệm trên `train` xem tăng lên 3 (khớp `MAX_QUESTIONS_PER_TOPIC=3` đã định nghĩa sẵn ở `src/psyvec/assessment/stopping.py:15` — hiện chưa được dùng ở `run_qwen_sample.py`) có cải thiện độ chính xác tần suất/mức độ không, đổi lại chi phí thêm 1 lượt/topic khi cần.

### 2.5 Scorer — chi tiết

- Giữ cấu trúc hiện tại (1 lệnh/topic, 8 lệnh tổng — `run_qwen_sample.py:413-436`), nhưng prompt bổ sung **evidence gốc kèm turn_id** bên cạnh `topic_history` grounded, và yêu cầu output thêm field `evidence_turn_ids` (mở rộng `ScoreParse`/`parse_score_and_summary` trong `response_parsing.py`, giữ backward-compatible — thêm field optional, không đổi hành vi cũ khi field vắng mặt).
- Validate `evidence_turn_ids` trả về phải là tập con turn_id đã gửi trong prompt cho topic đó — reject/flag "citation mismatch" nếu bịa, coi như một dạng parse-failure mới để theo dõi riêng.
- Khi `status=missing` toàn bộ (không có cả specific lẫn cross-cutting evidence — hiếm) và `topic_history` cũng không có thông tin gì mới: cho phép scorer chấm 0 với `evidence_turn_ids=[]` + lý do "no disclosure found", đây là kết quả hợp lệ, không phải lỗi.

### 2.6 Cross-topic Consistency Review (Memory-Update) — chi tiết

- Input đổi từ `dialogue_transcript` giả lập (`run_qwen_sample.py:462-465`) sang: `dialogue_transcript` grounded (giữ, vì giờ đã đáng tin hơn) **+** toàn bộ `EvidenceBundle` gốc của 8 topic (để review chéo có căn cứ thật, không chỉ dựa vào lời client đã qua roleplay).
- Ràng buộc cứng (map theo `src/psyvec/roles/updater.py:26-45`): mọi điểm bị sửa **phải** kèm `evidence_slot_ids`/`evidence_turn_ids` khác rỗng — nếu Updater trả về sửa điểm mà không trích evidence, reject thay đổi đó (giữ điểm bước 2.5).

### 2.7 Demographics (phụ, không ảnh hưởng PHQ-8 score)

Giữ đề xuất ở v1: quét `real_interview` (regex/keyword đơn giản ở phần đầu interview) thay vì roleplay hỏi tuổi/giới/nghề nghiệp — ưu tiên thấp, làm sau cùng.

---

## 3. Nối vào kiến trúc typed sẵn có (`src/psyvec/`)

Không đổi so với v1 — vẫn để ở **Phase 2**, sau khi Grounded Interview Loop đã chạy đúng và đo được cải thiện MAE:

1. **`AssessmentState` + `TopicState` + `EvidenceSlot`** (`state/contracts.py`) làm kiểu dữ liệu chính thức cho EvidenceBundle + điểm sơ bộ/điểm cuối. Vì giờ có 2 loại "evidence" (trích dẫn thật `verbatim` vs hội thoại grounded `elaborated`), đề xuất dùng `EvidenceSlot.kind` để phân biệt — cho phép audit sau này biết điểm số dựa trên câu nói thật hay dựa trên phần "đào sâu" qua hội thoại.
2. **`DecisionEventEngine`** (`state/engine.py`) ghi lại mỗi vòng hỏi-đáp, mỗi lần chấm điểm, mỗi lần update thành `DecisionEvent` có `pre_state_hash`/`post_state_hash`.
3. **`assessment/stopping.py`** (`should_continue`, `MAX_QUESTIONS_PER_TOPIC=3`) — dùng thay cho logic necessity tự viết trong `run_qwen_sample.py`, đã có sẵn đúng ngữ nghĩa AgentMental baseline.
4. **`roles/updater.py`** dùng nguyên cho bước 2.6.
5. **`roles/reporter.py`** dùng để render `AssessmentReport` cuối (label-free) thay vì tự in bảng.
6. **`memory/case_memory.py`** dùng làm nơi lưu EvidenceBundle/điểm giữa các bước, enforce đúng vai nào được đọc gì.
7. **`research/splits.py`** dùng để "khóa" quyền truy cập `test` split.

---

## 4. Hiệu quả (số lệnh gọi LLM & tốc độ) — kỳ vọng thực tế hơn so với v1

Vì **giữ nguyên vòng hỏi-đáp nhiều lượt**, số lệnh gọi LLM sẽ **không giảm mạnh** như v1 (v1 loại bỏ hoàn toàn roleplay). Hiệu quả trong v2 đến từ những chỗ khác:

| Nguồn hiệu quả | Giải thích |
|---|---|
| Prompt ngắn & đúng trọng tâm hơn | Evidence snippet (vài câu, đúng topic) thay cho 50 lượt small-talk toàn bộ interview — giảm token/lệnh gọi dù *số lệnh gọi* không đổi nhiều |
| Necessity chính xác hơn → ít follow-up thừa | Khi evidence đã rõ ràng + đã có thông tin tần suất, necessity-check (giờ có default thông minh hơn) sẽ dừng đúng lúc thay vì luôn chạy đủ `max_depth` |
| Ít phải làm lại do parse-failure/hallucination | Grounding giảm khả năng client trả lời mâu thuẫn khiến scorer/updater phải fallback hoặc parse fail |
| Song song hoá (Phase 3) | Không đổi thuật toán, chỉ đổi cách gọi — chạy 8 topic hoặc nhiều participant song song khi dùng `--api-base-url` (vLLM/Ollama) |

Ước tính số lệnh gọi/participant: **~24–46** (so với baseline ~26–50) — cải thiện khiêm tốn về *số lượng*, nhưng cải thiện đáng kể về *chất lượng mỗi lệnh gọi* (không lãng phí vào ngữ cảnh sai/hallucination). Đây là đánh đổi có chủ đích theo yêu cầu giữ trải nghiệm phỏng vấn thật.

---

## 5. Đánh giá & Validation Protocol

### 5.1 Metrics bổ sung (giữ nguyên MAE/RMSE/Pearson/Accuracy/F1 hiện có ở `evaluation/metrics.py`, thêm mới):

- **Per-item MAE/accuracy** (8 mục PHQ-8 riêng biệt) — biết chính xác topic nào (dự đoán: Concentration/Appetite/Psychomotor — không có tag anchor) còn yếu.
- **Evidence coverage rate**: % topic/participant có `status=known` (evidence riêng) vs chỉ có cross-cutting vs hoàn toàn `missing`.
- **Faithfulness / hallucination rate (MỚI, quan trọng cho hybrid)**: heuristic kiểm tra client_reply có nhắc tới nội dung/từ khóa **không xuất hiện** trong EvidenceBundle lẫn cross-cutting bucket của topic đó không (lexical-overlap thấp bất thường so với evidence được cấp) → gắn cờ "possible fabrication" để review thủ công. Đây là chỉ số theo dõi trực tiếp việc "roleplay vẫn còn hỏi-đáp tự do" có bị lạm dụng hay không.
- **Parse-failure / excluded-sample rate** — giữ nguyên (`run_batch_eval.py:58-64`), theo dõi có giảm không.
- **Citation-mismatch rate (MỚI)**: % lần scorer/updater trả `evidence_turn_ids` không thuộc tập đã gửi (bịa trích dẫn) — nếu >0 đáng kể, cần siết lại prompt/parser.
- **Golden regression set**: khóa 4 participant đã phân tích kỹ — 302 (GT 4, dễ), 346 (GT 23, ca thiếu evidence nặng nhất trong 50 lượt đầu), 404 (GT 0, ca false-positive điển hình do bịa "trouble sleeping"), 440 (GT 19, ca thiếu nghiêm trọng). Mọi thay đổi retrieval/prompt sau này phải chạy lại đúng 4 ca này trước khi merge, kiểm tra thủ công cả điểm số lẫn nội dung hội thoại (đọc `dialogue_transcript` xem client có còn bịa không, dù điểm đúng).

### 5.2 Kỷ luật split

1. Xây tag inventory + tune keyword lexicon **chỉ trên `train`** (107 mẫu).
2. Đánh giá A/B (EGIA vs baseline hiện tại) **trên `dev`** (35 mẫu) — baseline hiện tại: MAE=5.0/5.37, Pearson r=0.50/0.40 (2 model đã chạy).
3. `test` (47 mẫu, đáp án ở `full_test_split.csv`) **chỉ chạy 1 lần duy nhất**, sau khi đã chốt toàn bộ thiết kế từ (1)+(2).

### 5.3 Ablation cần chạy để có kết luận chắc chắn

| Arm | Mục đích |
|---|---|
| Baseline hiện tại (đã có kết quả) | Mốc so sánh |
| EGIA: chỉ sửa retrieval + client system prompt (giữ necessity/follow-up như cũ) | Đo riêng tác động của grounding lên hallucination |
| EGIA + necessity/follow-up nhắm rubric-gap | Đo thêm tác động của việc hỏi đúng trọng tâm (tần suất/mức độ) |
| EGIA đầy đủ + Cross-topic Review dùng evidence gốc | Đo thêm tác động bước review |
| EGIA đầy đủ + semantic fallback (nếu Phase 2 cần) | Đo thêm tác động embedding cho 5 topic không có tag |

Dùng `src/psyvec/evaluation/onoff.py` (đã có sẵn `OnOffArm`/`OnOffResult`) làm khung so sánh cặp paired.

---

## 6. Work Breakdown (thứ tự implement đề xuất)

### Phase 1 — Grounded Interview Loop (accuracy fix, giữ nguyên cấu trúc hội thoại)
1. `scripts/build_tag_inventory.py` — quét `train`, xuất tần suất tag → review thủ công → `configs/scales/topic_tag_map.json`.
2. `configs/scales/topic_keywords.json` — lexicon khởi tạo theo §2.3(b).
3. `src/psyvec/evaluation/evidence_retrieval.py` (module mới) — hàm `retrieve_evidence(real_interview, topics_dict, tag_map, keyword_lexicon) -> dict[topic, EvidenceBundle]`, EvidenceBundle gồm `status`, `snippets` (specific), `cross_cutting` (fallback tâm trạng nền). Unit test (`tests/unit/test_evidence_retrieval.py`) trên fixture 302/346/404/440 — assert 404 tìm thấy lượt 145 "handling my sleep pretty well" vào bucket Sleep Problems; 346 phần lớn topic chỉ có cross-cutting evidence (đúng thực tế của 50 lượt đầu).
4. Sửa `run_qwen_sample.py`:
   - Bỏ `interview_history = real_interview[:50]` làm system prompt chung; thay bằng: gọi retrieval 1 lần đầu sample, rồi build system prompt **per-topic** từ `EvidenceBundle[topic]` (§2.4).
   - Sửa necessity default theo evidence status + regex rubric-gap check (§2.4).
   - Sửa follow-up prompt để nhắm rubric-gap cụ thể (§2.4).
   - Sửa scorer prompt để nhận cả `topic_history` lẫn evidence gốc kèm turn_id; thêm field `evidence_turn_ids` vào output.
   - Sửa Memory-Update prompt để nhận cả evidence gốc toàn cục, không chỉ `dialogue_transcript`.
5. Mở rộng `response_parsing.py`: thêm field optional `evidence_turn_ids` vào `ScoreParse`/`UpdaterParse` (không đổi hành vi khi field vắng mặt, giữ backward-compat với test hiện có), thêm validate "citation phải thuộc tập đã gửi" ở tầng gọi (không nhất thiết trong parser thuần).
6. Chạy trên `train` để tune lexicon/prompt (lặp bước 1-5 tới khi ổn định) — theo dõi cả metric số lẫn đọc thủ công `dialogue_transcript` của vài ca để chắc chắn hội thoại không còn bịa lộ liễu.
7. Chạy trên `dev`, so trực tiếp với `results/evaluations/qwen*_memory_only` hiện tại bằng metrics ở §5.1 (bao gồm Faithfulness rate, Citation-mismatch rate — 2 chỉ số mới). Lưu vào `results/evaluations/<model>_egia_v1/`.
8. Review kết quả với user (bạn) trước khi đi Phase 2 — đặc biệt xem lại 4 golden case + vài ca ngẫu nhiên đọc `dialogue_transcript` để xác nhận cảm giác "giống phỏng vấn thật" vẫn được giữ.

### Phase 2 — Nối kiến trúc typed (auditability, không đổi thuật toán)
9. Map `EvidenceBundle` → `EvidenceSlot`/`TopicState` (`state/contracts.py`), phân biệt `kind` verbatim vs elaborated; chạy qua `DecisionEventEngine`.
10. Thay necessity logic tự viết bằng `assessment/stopping.should_continue` (`MAX_QUESTIONS_PER_TOPIC=3`).
11. Cross-topic Review dùng `roles/updater.Updater.apply` thật.
12. Output cuối dùng `roles/reporter.AssessmentReport`.
13. `research/splits.py` gate quyền đọc `data/processed_daic_woz/test`.

### Phase 3 — Hiệu năng (song song hoá, không đổi kết quả)
14. Song song hoá theo participant và/hoặc theo topic khi dùng `--api-base-url` (vLLM/Ollama hỗ trợ concurrent) — vòng `for` tuần tự hiện tại ở `run_batch_eval.py:257`.
15. Cache retrieval layer theo participant (độc lập model) để tái dùng khi so sánh nhiều model.

### Phase 4 — Test-split (chỉ 1 lần, sau khi đã chốt ở Phase 1-3)
16. Chạy EGIA (đã tune xong) trên `test` split với đáp án `full_test_split.csv`, báo cáo số liệu cuối cùng.

### Tình trạng triển khai Phase 1 (cập nhật sau khi implement)

- ✅ Bước 1: `scripts/build_tag_inventory.py` — quét `train`, tìm 156 tag. Đã tự tay review và chốt `configs/scales/topic_tag_map.json` (chỉ giữ tag có ý nghĩa lâm sàng rõ ràng; loại các tag backchannel/small-talk).
- ✅ Bước 2: `configs/scales/topic_keywords.json` — lexicon khởi tạo, đã tune 1 vòng bằng `scripts/audit_keyword_lexicon.py` chạy trên `train` (phát hiện và sửa các collision rõ ràng: "numb" khớp nhầm vào "number", "blue" khớp nhầm màu sắc, "weight" khớp nhầm "lifting weights", "down" quá nhiễu, "can't think"/"guilty" bắt nhầm idiom/ngữ cảnh pháp lý).
- ✅ Bước 3: `src/psyvec/evaluation/evidence_retrieval.py` + `tests/unit/test_evidence_retrieval.py` (20 tests, gồm golden case participant 404 xác nhận bắt được lượt 145 "handling my sleep pretty well" — đúng như phân tích ở mục 0).
- ✅ Bước 4: `scripts/run_qwen_sample.py` đã sửa theo đúng thiết kế §2.4-2.6: bỏ `real_interview[:50]`, client system prompt giờ build per-topic từ `EvidenceBundle`, necessity-default và follow-up nhắm rubric-gap qua `psyvec.evaluation.interview_policy`, scorer + Memory-Update đều nhận evidence gốc kèm `turn_id` và bị validate citation (`invalid_citations`). Demographics chuyển sang regex (`extract_demographics`), bỏ hẳn 1 lệnh gọi LLM hallucination-prone.
- ✅ Bổ sung ngoài kế hoạch gốc nhưng cần cho Definition of Done ở mục 8: `evidence_turn_ids` (field mới trong `ScoreParse`/`UpdaterParse`, `response_parsing.py`), Faithfulness/citation-mismatch tracking (`low_faithfulness_flag`, `interview_policy.py`), và `run_batch_eval.py` giờ tính + in + lưu 3 chỉ số ở §5.1 (evidence coverage rate, citation-mismatch rate, low-faithfulness rate) qua `compute_evidence_diagnostics_summary`.
- ✅ Kiểm chứng wiring: `tests/integration/test_grounded_interview_pipeline.py` chạy toàn bộ `run_full_sample_assessment` với fake engine (không cần torch/transformers thật) trên chính participant 404 — xác nhận pipeline mới không crash và trả đúng schema, bao gồm nhánh Memory-Update reject revision không có evidence.
- ✅ Toàn bộ test suite: 232 passed (thêm 62 test mới so với baseline), `ruff check` sạch cho mọi file mới/sửa, `mypy --strict` sạch cho toàn bộ `src/psyvec/evaluation/`.
- ⏳ Bước 6-7 (Phase 1 done-definition ở mục 8): **chưa chạy được** trong môi trường này vì không có model thật (không có torch/transformers/GPU, không có endpoint `--api-base-url` khả dụng). Cần bạn tự chạy trên `train` rồi `dev` với model thật (vd `python scripts/run_batch_eval.py --data-dir data/processed_daic_woz/dev --enable-memory-update --api-base-url ...`) để: (a) tinh chỉnh thêm lexicon nếu cần, (b) lấy MAE/Pearson r/evidence-coverage/faithfulness-rate thật so với baseline (MAE 5.0/5.37) trước khi coi Phase 1 là "done" và đi tiếp Phase 2.

---

## 7. Rủi ro & Giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Roleplay vẫn có thể hallucinate dù có evidence (chỉ giảm chứ không triệt tiêu — bản chất giữ hội thoại tự do là có rủi ro này) | Faithfulness-rate metric (§5.1) + golden regression set đọc thủ công; rule cứng trong system prompt; nếu rate cao sau Phase 1, cân nhắc thêm 1 bước "faithfulness re-check" nhẹ (LLM tự so sánh reply với evidence, gắn cờ nếu lệch — optional, chỉ làm nếu cần) |
| Lexicon/tag-map overfit vào `train` | Bắt buộc build trên `train`, chỉ *xem* kết quả `dev` để so sánh, không sửa lexicon dựa trên lỗi cụ thể ở `dev` |
| 5/8 topic không có tag-anchor → dễ chỉ dựa vào cross-cutting bucket, có thể làm các topic này "đồng nhất hóa" theo tâm trạng chung thay vì phản ánh đúng triệu chứng riêng | Đo evidence-coverage rate riêng cho các topic này; nếu thấy điểm 5 topic này quá tương quan với nhau (gần như luôn giống điểm Depressed Mood), ưu tiên làm semantic fallback (§2.3d) sớm hơn dự kiến |
| Follow-up "nhắm rubric-gap" bằng regex có thể miss các cách diễn đạt tần suất không chuẩn | Regex chỉ là gate để quyết định có hỏi thêm không; nếu miss, hệ quả là 1 lượt hỏi thừa (an toàn hơn là hỏi thiếu) — chấp nhận được |
| Model nhỏ (0.5B/2.5B) không theo được thêm field `evidence_turn_ids` trong JSON | Giữ field optional ở parser, nếu model không trả field đó thì fallback: chấp nhận score nhưng đánh dấu "unverified citation" thay vì reject cả kết quả |
| Cross-topic Review "sửa điểm" nhưng không có evidence mới | Enforce cứng theo `roles/updater.py:38-41`: reject revision không kèm evidence |
| Không giảm được nhiều lệnh gọi LLM như kỳ vọng ban đầu (đánh đổi đã biết trước, xem §4) | Bù lại bằng Phase 3 (song song hoá) thay vì cắt giảm cấu trúc hội thoại |

---

## 8. Định nghĩa "Done" cho từng phase

- **Phase 1 done khi**: (a) unit test retrieval + parser xanh; (b) 4 golden case (302/346/404/440) — đọc thủ công `dialogue_transcript` xác nhận client không còn phát biểu mâu thuẫn trực tiếp với evidence thật (vd 404 không còn tự bịa "trouble sleeping"); (c) so trên `dev`: MAE tổng giảm rõ rệt so với baseline (5.0/5.37), Pearson r tăng; (d) evidence-coverage rate + faithfulness rate + citation-mismatch rate được báo cáo cho cả 8 topic.
- **Phase 2 done khi**: pipeline chạy hoàn toàn qua `DecisionEventEngine`/`CaseMemory`/`assessment.stopping`, có thể trace lại từng điểm số về đúng `DecisionEvent` + evidence turn_id, không đổi metric so với cuối Phase 1 (refactor thuần).
- **Phase 3 done khi**: thời gian chạy batch trên `dev` giảm đáng kể (so `elapsed_seconds` trước/sau), metric không đổi.
- **Phase 4 done khi**: có đúng 1 bộ số liệu trên `test` split, kèm toàn bộ log/config đã dùng để tái lập (`ProvenanceRef`).

---

## 9. Việc KHÔNG làm ở đợt này (out of scope, tránh scope creep)

- Không train/fine-tune LoRA hay bất kỳ adapter nào (`src/psyvec/model/adapters.py`, `training/*`).
- Không đổi rubric/thang điểm trong `configs/scales/scoring_standards.json`.
- Không làm semantic-embedding fallback ngay (chỉ làm nếu Phase 1 cho thấy cần).
- Không loại bỏ cơ chế hỏi-đáp nhiều vòng (đã chốt theo yêu cầu — đây chính là điểm khác biệt lớn nhất so với bản v1).
- Không đụng vào `src/psyvec/evolution/*`, `lessons/*` (thuộc phạm vi self-evolving/lesson-mining, khác mục tiêu "fix accuracy of PHQ-8 screening" của plan này).
