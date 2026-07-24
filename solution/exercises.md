# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Ở temperature 0.0, mô hình tạo ra câu trả lời ổn định, ít sáng tạo và tập trung vào các thông tin phổ biến. Ở mức 0.7, câu trả lời tự nhiên hơn, đa dạng cách diễn đạt nhưng vẫn giữ được tính mạch lạc. Khi tăng lên 1.2 và đặc biệt 1.8, mô hình có xu hướng sáng tạo quá mức, xuất hiện một số thông tin chưa chính xác hoặc cách diễn đạt kém tự nhiên; mức 1.8 bắt đầu có dấu hiệu giảm độ mạch lạc.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Đối với trợ lý soạn thảo hợp đồng pháp lý, tôi sẽ đặt temperature thấp, khoảng 0.0–0.2, vì lĩnh vực pháp lý yêu cầu tính chính xác, nhất quán và hạn chế tạo ra những nội dung không có căn cứ. Mô hình nên ưu tiên các mẫu câu chuẩn, thuật ngữ pháp lý chính xác và ít thay đổi trong mỗi lần sinh. Đối với trợ lý viết slogan quảng cáo, tôi sẽ đặt temperature cao hơn, khoảng 0.8–1.2, vì công việc này cần sự sáng tạo, nhiều ý tưởng mới và cách diễn đạt độc đáo. Tuy nhiên, vẫn cần giới hạn temperature để tránh tạo ra những câu khẩu hiệu thiếu ý nghĩa hoặc không phù hợp với thương hiệu.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Tổng số lượt gọi/ngày = 20.000 × 2 = 40.000 lượt; tổng token đầu ra/ngày = 40.000 × 500 = 20.000.000 token = 20.000 (nghìn token). Với giá output trong bảng: GPT-4o ($0.010/1K) → 20.000 × 0.010 = **$200/ngày**; GPT-4o-mini ($0.0006/1K) → 20.000 × 0.0006 = **$12/ngày** — chênh lệch gần **17 lần**. Model lớn đáng chi phí khi output cần độ chính xác/suy luận cao và sai sót tốn kém hơn nhiều so với tiền API — ví dụ trợ lý soạn thảo hợp đồng pháp lý hoặc debug code phức tạp, nơi một câu trả lời sai có thể gây thiệt hại lớn. Model nhỏ là lựa chọn đúng khi workload có khối lượng lớn nhưng tác vụ đơn giản, ít rủi ro — ví dụ phân loại intent trong chatbot hỗ trợ khách hàng, autocomplete, hay trả lời FAQ — nơi tốc độ và chi phí quan trọng hơn từng chút chất lượng biên.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona "nhà thơ", phản hồi mở đầu bằng hình ảnh ví von — ví dụ: "Hãy hình dung thế này nhé: Nó giống như một người làm vườn..." — hoàn toàn tránh thuật ngữ kỹ thuật. Với persona "kỹ sư senior", phản hồi mở đầu trực tiếp, xưng danh vai trò rõ ràng — "Chào bạn, với tư cách là một kỹ sư phần mềm senior, tôi sẽ giải thích..." — và có xu hướng đi thẳng vào định nghĩa kỹ thuật. Cùng một câu hỏi nhưng giọng văn, cách mở đầu và mức độ dùng thuật ngữ khác hẳn nhau, trong khi độ dài ở mức tương đương do cùng giới hạn `max_tokens`. Điều này cho thấy system prompt điều khiển tốt giọng văn, góc nhìn/vai trò và mức độ kỹ thuật của câu trả lời, nhưng không tự động kiểm soát độ dài — muốn giới hạn độ dài vẫn cần `max_tokens` hoặc yêu cầu tường minh trong prompt.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với đoạn văn ~150 từ về Hà Nội (138 từ thực tế): `count_tokens` (tiktoken)
> đếm được **195 token**, trong khi ước lượng `số từ / 0.75` cho ra **184
> token** — chênh lệch khoảng **5,6%**, ước lượng thô thấp hơn số thật. Nếu
> dùng công thức thô để dự toán ngân sách, bạn sẽ **dự toán thiếu** (under-
> estimate) chi phí thật. Lý do: công thức "0,75 từ ≈ 1 token" được hiệu
> chỉnh theo tiếng Anh, còn tiếng Việt có dấu thường bị tokenizer tách thành
> nhiều sub-token hơn (do dấu thanh, từ ghép, ký tự Unicode tổ hợp), nên số
> token thực tế trên mỗi từ tiếng Việt thường cao hơn tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản (a) hưởng lợi nhiều nhất từ streaming: người dùng nhìn
> thấy chữ xuất hiện ngay lập tức thay vì nhìn màn hình trống chờ vài giây,
> cảm giác "đang được phản hồi" giúp giảm cảm giác chờ đợi rõ rệt, đặc biệt
> với câu trả lời dài. Trợ lý giọng nói (b) hưởng lợi ít hơn vì hệ thống
> text-to-speech thường cần một đoạn văn bản (hoặc ít nhất một câu hoàn
> chỉnh) mới có thể tổng hợp giọng nói tự nhiên — stream từng token nhỏ có
> thể khiến giọng đọc bị ngắt quãng hoặc phải xử lý theo cụm câu thay vì
> từng chunk thô. Pipeline dịch tài liệu chạy ngầm ban đêm (c) hoàn toàn
> không cần streaming vì không có người dùng nào đang chờ xem kết quả theo
> thời gian thực — hệ thống chỉ cần chờ toàn bộ output rồi lưu file, streaming
> ở đây chỉ thêm độ phức tạp code mà không có lợi ích thực tế.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Với delay cố định, tất cả client bị lỗi cùng lúc sẽ retry sau đúng cùng
> một khoảng thời gian — server vừa mới hồi phục lại nhận nguyên một đợt
> tấn công retry đồng loạt lần nữa, có thể sập tiếp (hiện tượng gọi là
> "thundering herd"). Exponential backoff giãn dần khoảng chờ (0.1s → 0.2s →
> 0.4s...) nên tải retry giảm dần theo thời gian, cho server đủ thời gian
> phục hồi thay vì bị dồn dập ngay lập tức. Tuy nhiên nếu hàng nghìn client
> đều bắt đầu lỗi cùng một thời điểm (ví dụ server down đồng loạt), chúng vẫn
> retry đồng bộ theo đúng cùng một dãy delay — vẫn tạo ra các đợt sóng retry
> tập trung tại 0.1s, 0.2s, 0.4s... của mọi client. "Jitter" giải quyết vấn đề
> này bằng cách thêm một khoảng ngẫu nhiên vào mỗi lần chờ, làm các lần retry
> của các client rải đều ra theo thời gian thay vì dồn vào cùng một thời
> điểm, giảm nguy cơ tạo ra đợt tải đột biến mới.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: *"Bạn là trợ giảng thân thiện của khóa AI Practical
> Competency Program. Luôn trả lời ngắn gọn trong tối đa 3 câu, dùng tiếng
> Việt đơn giản, dễ hiểu cho người mới bắt đầu. Nếu câu hỏi nằm ngoài phạm vi
> lập trình/AI của khóa học, hãy lịch sự từ chối và hướng người dùng quay lại
> chủ đề khóa học."*
>
> Hai chỗ nếu xóa sẽ đổi hành vi rõ rệt:
> 1. **"Luôn trả lời ngắn gọn trong tối đa 3 câu"** — nếu xóa, trợ lý sẽ có
>    xu hướng trả lời dài dòng, lan man hơn, mất đi tính "trợ giảng súc tích"
>    và tốn nhiều token/chi phí hơn mỗi lượt.
> 2. **"Nếu câu hỏi nằm ngoài phạm vi... hãy lịch sự từ chối và hướng người
>    dùng quay lại chủ đề"** — nếu xóa, trợ lý sẽ trả lời mọi câu hỏi kể cả
>    ngoài chủ đề khóa học (ví dụ hỏi về nấu ăn, thời tiết), mất đi vai trò
>    tập trung vào một domain cụ thể mà persona này được thiết kế để phục vụ.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Tình huống: Lượt 1, người dùng nói "Tôi tên là Đạt, tôi đang học Python và
> muốn ưu tiên ví dụ code ngắn gọn." Từ lượt 2 đến lượt 5, cuộc trò chuyện
> chuyển sang các câu hỏi kỹ thuật khác không nhắc lại tên hay yêu cầu ban
> đầu. Đến lượt 6, người dùng hỏi "Bạn có nhớ tôi tên gì và tôi thích code
> kiểu nào không?" — vì `history` chỉ giữ 4 lượt gần nhất (8 message), thông
> tin ở lượt 1 đã bị cắt khỏi history, nên trợ lý trả lời sai hoặc nói không
> biết, dù đây là thông tin quan trọng người dùng đã cung cấp ngay từ đầu.
> Cách khắc phục: khi history sắp bị cắt, tóm tắt các lượt cũ nhất thành 1–2
> câu ngắn (ví dụ dùng chính LLM để tóm tắt) và giữ bản tóm tắt đó như một
> message hệ thống bổ sung ("Ghi nhớ: người dùng tên Đạt, thích code ngắn
> gọn") thay vì xóa hẳn — vừa giữ được ngữ cảnh quan trọng, vừa không để
> history phình to vô hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
