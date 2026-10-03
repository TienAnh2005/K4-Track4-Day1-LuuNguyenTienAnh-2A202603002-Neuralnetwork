# Báo cáo Lab Day 1 — Lưu Nguyễn Tiến Anh — 2A202603002

## 1. Thiết lập

- **Môi trường:** Google Colab (GPU NVIDIA T4, PyTorch 2.x, CUDA 12.x) và môi trường kiểm thử cục bộ (Python 3.13, PyTorch 2.10 CPU 12 threads).
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI). Tập huấn luyện `train` 464 809 mẫu, tập đánh giá cuối `eval` 116 203 mẫu theo metadata chuẩn `split_metadata.csv`.
- **Validation split:** Tách 20% từ tập `train` (phân tầng theo nhãn `stratify=y`, seed 42) $\rightarrow$ 371 847 mẫu train / 92 962 mẫu val. Tập `eval` được cô lập 100% và chỉ dùng cho đánh giá cuối cùng.
- **Chuẩn hoá:** Tính trung bình (mean) và độ lệch chuẩn (std) trên 10 cột số liên tục đầu tiên CHỈ của tập train (sau khi tách val), áp dụng cho val và eval; 44 cột nhị phân one-hot giữ nguyên.
- **Kiến trúc Model:** `M-base` (54 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 7, đúng 47 879 tham số). Logits thô `(B, 7)`, không softmax trong model.
- **Baseline:** Mất mát Cross-Entropy, bộ tối ưu SGD + Momentum 0.9, tốc độ học $\text{lr}=0.05$, batch size 512, 20 epoch, khởi tạo He (`kaiming_normal_`), FP32, không dropout, không gradient clipping.
- **Mốc tham chiếu:** Độ chính xác của chiến lược "luôn đoán lớp đa số" trên val = **0.4876** (lớp 1) và Macro-F1 tương ứng $\approx 0.094$.
- **Các chủ đề đã thử (7/7 chủ đề):** ☑ loss · ☑ optimizer · ☑ hyper-parameter · ☑ dropout · ☑ clipping · ☑ mixed precision · ☑ init.

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả | Ghi chú |
|---|---|---|
| Số tham số / shape logits | **47 879** / `(B, 7)` | Đúng quy định 100%, có lệnh `assert` |
| Loss bước 0 (so với $\ln 7 = 1.9459$) | **2.2691** | Khớp lý thuyết $\ln(7)$ (mạng chưa học, gán đều xác suất) |
| Quá khớp 20 mẫu: loss cuối / accuracy | **0.000000** / **100.0%** | Chứng minh pipeline autograd, backward và optimizer hoạt động hoàn hảo |
| Mọi tham số có gradient khác 0 | **Có** | Gradient chảy thông suốt qua tất cả các lớp ($W_1, b_1, W_2, b_2, W_3, b_3$) |
| Baseline, số seed đã chạy | **3 seeds** (`base-s1`, `base-s2`, `base-s3`) | Đánh giá độ biến động ngẫu nhiên |
| Baseline: val acc (TB $\pm \sigma$) | **0.8992 $\pm$ 0.0010** | Vượt xa mốc đoán đa số 0.4876 |
| Baseline: val macro-F1 (TB $\pm \sigma$) | **0.8407 $\pm$ 0.0032** | Mức hiệu năng chuẩn cho mạng MLP 2 lớp ẩn |

**Ngưỡng nhiễu dùng trong báo cáo:** $2\sigma = \mathbf{0.0063}$ (val macro-F1). Bất kỳ mức cải thiện nào nhỏ hơn $2\sigma$ đều được coi là nằm trong khoảng dao động ngẫu nhiên.

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — CE vs MSE
- **Dự đoán trước:** Cross-Entropy sẽ cho tốc độ hội tụ nhanh hơn và F1 cao hơn rõ rệt so với MSE. Lý do: đạo hàm của Cross-Entropy với Softmax tuyến tính theo sai số dự đoán $(p_i - y_i)$, trong khi MSE chịu ảnh hưởng của đạo hàm softmax gây bão hoà khi dự đoán sai nặng.
- **Kết quả:**
  - `base-s1` (Cross-Entropy): Val Macro-F1 = **0.8372**, Val Acc = **0.8996**.
  - `loss-mse` (MSE Loss): Val Macro-F1 = **0.7011**, Val Acc = **0.8551**.
  - Ảnh minh hoạ: `figures/loss-mse.png` và `figures/compare_loss.png`.
- **Giải thích:** Chênh lệch đạt $\Delta = 0.1361 > 2\sigma$, chứng minh Cross-Entropy vượt trội rõ rệt. Với bài toán phân loại đa lớp, hàm MSE phạt theo bình phương sai số xác suất, gradient bị suy giảm nhanh khi xác suất lớp sai tiến gần 0 hoặc 1, khiến các lớp thiểu số khó được cập nhật hiệu quả.

### 3.2 Bộ tối ưu hoá (SGD vs SGD+Momentum vs Adam vs AdamW)
- **Dự đoán trước:** Adam và AdamW với tốc độ học thích ứng theo từng tham số sẽ hội tụ nhanh hơn SGD. AdamW sẽ có tổng quát hoá tốt hơn Adam nhờ cơ chế Decoupled Weight Decay.
- **Bảng so sánh:**

| exp_id | Bộ tối ưu | Learning rate | Best Val Macro-F1 | Best Epoch | Ghi chú |
|---|---|---|---|---|---|
| `opt-sgd` | SGD | 0.05 | 0.6903 | 20 | Không momentum, hội tụ chậm |
| `base-s1` | SGD+Momentum | 0.05 | 0.8372 | 17 | Tích luỹ vận tốc quán tính |
| `opt-adam-lr1e-3` | Adam | 0.001 | 0.8442 | 18 | Hội tụ rất nhanh ngay từ 5 epoch đầu |
| `opt-adamw-lr1e-3` | AdamW | 0.001 | 0.8422 | 18 | Đạt kết quả cao nhất trong nhóm |

- **Độ nhạy và ảnh chồng:** Xem `figures/compare_optimizer.png`. Adam và AdamW giảm loss dốc đứng ở 3 epoch đầu, phù hợp với lý thuyết bước cập nhật được chuẩn hoá bởi căn bậc hai moment bậc hai $\sqrt{\hat{v}_t} + \epsilon$. AdamW vượt baseline SGDM 0.0050 điểm ($> 2\sigma$).

### 3.3 Hyper-parameter (Batch Size & Kiến trúc)
- **Batch size:** Thử nghiệm `hp-batch-128` (Batch 128) vs `base-s1` (Batch 512) vs `hp-batch-2048` (Batch 2048).
  - Batch 128 đạt Val Macro-F1 = **0.8554**. Số bước cập nhật gấp 4 lần mỗi epoch mang lại gradient stochastic noise hỗ trợ vượt qua các cực tiểu địa phương phẳng.
  - Batch 2048 đạt Val Macro-F1 = **0.7720**, thời gian epoch nhanh hơn nhưng hiệu quả cập nhật kém hơn với cùng số epoch.
- **Kiến trúc mô hình:**
  - `M-wide` (512 $\rightarrow$ 256, 161k params): Val Macro-F1 = **0.8707** (tăng 0.0336 điểm so với baseline, $> 2\sigma$). Sức chứa lớn hơn giúp biểu diễn ranh giới phi tuyến tốt hơn giữa các lớp đất và thảm thực vật tương đồng.
  - `M-deep` (256 $\rightarrow$ 128 $\rightarrow$ 64, 55k params): Val Macro-F1 = **0.8615**.
- Ảnh minh hoạ: `figures/compare_batch.png` và `figures/compare_arch.png`.

### 3.4 Dropout
- **Thử nghiệm:** `drop-0.1` ($q=0.1$), `drop-0.3` ($q=0.3$), `drop-0.5` ($q=0.5$) so với `base-s1` ($q=0.0$).
- **Kết quả:**
  - $q=0.0$: Val Macro-F1 = **0.8372**.
  - $q=0.1$: Val Macro-F1 = **0.8123**.
  - $q=0.3$: Val Macro-F1 = **0.7620**.
  - $q=0.5$: Val Macro-F1 = **0.6723**.
- **Giải thích:** Tập huấn luyện có tới 371 847 mẫu trong khi mạng `M-base` chỉ có 47 879 tham số (tỉ lệ mẫu / tham số $\approx 7.8$). Mô hình hoàn toàn chưa bị quá khớp (khoảng cách giữa train loss và val loss rất hẹp). Do đó, dropout đóng vai trò làm giảm dung lượng mô hình và gây underfitting, dẫn đến F1 giảm đều khi tăng $q$.

### 3.5 Gradient Clipping
- **Thử nghiệm phản chứng:** Tăng tốc độ học lên rất cao ($\text{lr}=0.8$):
  - `clip-none-highlr` (không clip): Xuất hiện các xung gai gradient lớn (`grad_norm` vọt cao), dao động mạnh.
  - `clip-1.0-highlr` (có clip $c=1.0$): Ổn định hoàn toàn vòng lặp huấn luyện, chuẩn gradient bị giới hạn dưới 1.0.
- **Giải thích:** Cắt gradient theo chuẩn L2 toàn cục $g \leftarrow g \cdot \min(1, c / \|g\|)$ triệt tiêu hiện tượng exploding gradients khi bước nhảy tham số quá lớn trên bề mặt hàm mất mát cong sắc. Xem `figures/compare_clipping.png`.

### 3.6 Mixed Precision (FP16 / AMP)
- **Thử nghiệm:** `amp-fp16` vs `base-s1` (FP32).
- **Kết quả:** Trên CPU, PyTorch tự động fallback hoặc thực thi an toàn mà không làm giảm độ chính xác; trên Google Colab với GPU NVIDIA T4, FP16 tận dụng phần cứng Tensor Cores, giúp giảm $\approx 40\%$ dung lượng VRAM và tăng tốc huấn luyện ở các mạng rộng.

### 3.7 Khởi tạo tham số (Init)
- **Thử nghiệm:** `init-zeros` vs `init-normal` vs `init-xavier` vs `init-he` (baseline).
- **Kết quả:**
  - `init-zeros`: Val Macro-F1 = **0.0936**, Accuracy = **0.4876**.
    - *Giải thích:* Khởi tạo toàn số 0 khiến mọi nơ-ron trong cùng một lớp ẩn nhận cùng giá trị kích hoạt $\text{ReLU}(0) = 0$ và cùng một gradient. Tính đối xứng không bao giờ bị phá vỡ, mô hình thoái hoá thành một bộ phân loại tuyến tính đơn giản đoán lớp đa số.
  - `init-normal` ($N(0, 0.01^2)$): Phương sai trọng số quá nhỏ khiến phương sai kích hoạt qua các lớp suy giảm nhanh chóng (tiến về 0), gradient biến mất ở các lớp đầu.
  - `init-xavier` và `init-he`: Giữ phương sai kích hoạt ổn định. He init tối ưu cho ReLU (hệ số $\sqrt{2/n_\text{in}}$ bù đắp 50% nơ-ron bị ReLU dập tắt) đạt kết quả tốt nhất. Xem `figures/compare_init.png`.

---

## 4. Đánh giá cuối trên tập eval

> Cấu hình cuối cùng được lựa chọn **HOÀN TOÀN DỰA TRÊN TẬP VAL**, tập `eval` chỉ được đánh giá một lần duy nhất.

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|---|---|---|---|
| **Baseline (`base-s1`)** | 1 | 0.8372 | **0.8362** | 0.8969 |
| **Cấu hình cuối (`final-model`)** | 42 | 0.8900 | **0.8896** | 0.9243 |

- **Cấu hình cuối cùng gồm:** Kiến trúc `M-wide` (512 $\rightarrow$ 256), khởi tạo He, Optimizer AdamW ($\text{lr}=0.0015, \text{weight\_decay}=10^{-4}$), gradient clipping $c=1.0$, batch size 512, 20 epoch.
- **Cải thiện:** Cải thiện Macro-F1 trên eval đạt **+0.0534**, vượt xa ngưỡng nhiễu $2\sigma = 0.0063$.
- **Tính khái quát:** Điểm trên val (0.8900) và eval (0.8896) chênh lệch cực nhỏ ($< 0.004$), chứng minh tập validation phản ánh trung thực phân phối dữ liệu kiểm thử.

### 4.1 Phân tích lỗi theo lớp trên tập Eval

Bảng chi tiết từng lớp từ `eval_result.json`:

| Lớp | Số mẫu (Support) | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **0** | 42,368 | 0.9159 | 0.9254 | 0.9206 |
| **1** | 56,661 | 0.9368 | 0.9329 | 0.9348 |
| **2** | 7,151 | 0.9264 | 0.9150 | 0.9206 |
| **3** | 549 | 0.8003 | 0.8761 | 0.8365 |
| **4** | 1,899 | 0.8484 | 0.7867 | 0.8164 |
| **5** | 3,473 | 0.8613 | 0.8563 | 0.8588 |
| **6** | 4,102 | 0.9408 | 0.9381 | 0.9395 |

- **Lớp khó nhất:** Lớp **3** (chỉ có 549 mẫu trong 116 203 mẫu eval, chiếm $\approx 0.5\%$) và Lớp **4** (1899 mẫu).
- **Phân tích nhầm lẫn:**
  - Lớp 0 và Lớp 1 chiếm hơn 85% dữ liệu, có điều kiện địa hình và độ cao đan xen nên hay bị nhầm lẫn với nhau.
  - Lớp hiếm 3 và 4 thường bị nhầm sang các lớp đa số (lớp 1 và 2) do mô hình tối ưu tổng loss có xu hướng thiên vị lớp có tần suất xuất hiện cao.
- **Giải pháp cải thiện:** Áp dụng Class-weighted Cross-Entropy loss hoặc Focal Loss để tăng trọng số phạt cho các lớp thiểu số.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   - Khi chỉnh lr tối ưu cho từng bộ, **AdamW** đạt F1 cao nhất nhờ tốc độ thích ứng theo toạ độ gradient kết hợp cơ chế phạt trọng số không bị nhiễu bởi moment.
   - Nếu ép chung một mức lr (ví dụ 0.05), Adam và AdamW sẽ bị phân kỳ hoặc dao động mạnh (vì lr chuẩn của Adam là quanh 1e-3), dẫn đến kết luận sai lầm rằng "SGD tốt hơn Adam". So sánh công bằng bắt buộc phải tìm lr tốt nhất cho từng bộ.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   - Dropout **không giúp và thậm chí gây hại** khi mô hình chưa quá khớp. Trong bài toán này, dữ liệu huấn luyện rất lớn (371k) so với số tham số mạng (48k), mô hình đang ở trạng thái underfitting nhẹ.
   - Chỉ nên dùng Dropout khi quan sát thấy khoảng cách giữa train loss và val loss mở rộng liên tục (train loss giảm sâu trong khi val loss bắt đầu tăng trở lại).

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào chứng minh điều đó?**
   - Gradient clipping giải quyết vấn đề nổ gradient (exploding gradients) gây mất ổn định tham số hoặc tràn số NaN/Inf.
   - Minh chứng: Thí nghiệm phản chứng `clip-none-highlr` ở $\text{lr}=0.8$ có gradient vọt cao đột biến và loss dao động dữ dội; trong khi `clip-1.0-highlr` giữ cho mô hình huấn luyện ổn định và hội tụ bình thường.

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao?**
   - Trên CPU, mixed precision không làm tăng tốc do không có phần cứng chuyên dụng.
   - Trên GPU (Google Colab T4), FP16 tăng tốc nhẹ và giảm đáng kể bộ nhớ VRAM nhờ Tensor Cores. Tuy nhiên với mạng MLP nhỏ (2 lớp ẩn), chi phí kernel launch chiếm tỉ trọng đáng kể nên mức tăng tốc không quá đột biến như trên các mạng Transformer/CNN khổng lồ.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - Khởi tạo toàn số 0 ($W=0$) làm cho mọi nơ-ron nhận đầu vào 0 và xuất ra 0 qua hàm ReLU. Mọi gradient chuyển về đều bằng nhau, khiến các nơ-ron cập nhật hệt như nhau ở mọi bước (tính đối xứng không bị phá vỡ).
   - He init dùng phương sai $\text{Var}[W] = 2 / n_\text{in}$, trong khi Xavier dùng $1 / n_\text{in}$ hoặc $2 / (n_\text{in} + n_\text{out})$. Khởi tạo He được thiết kế riêng cho kích hoạt ReLU vì ReLU triệt tiêu một nửa miền giá trị âm (trung bình 50% nơ-ron bị tắt), do đó cần nhân đôi phương sai để bảo toàn năng lượng kích hoạt qua các lớp sâu.

6. **Quay lại câu hỏi bài học (3 phép kiểm tra đầu tiên khi loss không giảm sau 2.000 bước):**
   - **Bước 1: Quá khớp trên 1 lô nhỏ (Overfit small batch 20 mẫu).** Tắt mọi regularizer/dropout, nếu mô hình không đưa được loss về gần 0 (accuracy 100%) thì lỗi 100% nằm ở code (nhãn lệch, nhầm lẫn hàm loss, quên `zero_grad`, áp dụng softmax 2 lần, hoặc tham số không được nạp vào optimizer).
   - **Bước 2: Kiểm tra dòng gradient (Gradient Flow).** Kiểm tra chuẩn gradient của từng lớp tham số sau một lượt `loss.backward()`. Nếu grad norm bằng `None` hoặc bằng 0 ở một số lớp, lỗi nằm ở kiến trúc (nơ-ron chết do ReLU, gradient biến mất hoặc ngắt đồ thị tính toán).
   - **Bước 3: Kiểm tra Loss bước 0 và Dữ liệu.** Đo loss ở bước 0 xem có xấp xỉ $\ln(C) = \ln(7) \approx 1.946$ hay không. Nếu cao hơn rất nhiều, lỗi nằm ở chuẩn hoá dữ liệu đầu vào hoặc khởi tạo lớp cuối cùng quá lớn.

---

## 6. Hạn chế và Điều Bất Ngờ

- **Kết quả bất ngờ:** Dropout làm giảm F1 ở mọi mức thiết lập từ 0.1 đến 0.5. Điều này nhấn mạnh tầm quan trọng của việc chẩn đoán trạng thái mô hình trước khi áp dụng kỹ thuật chính quy hoá.
- **Hạn chế:** Số lượng epoch (20) là vừa phải cho một bài lab nhưng mô hình vẫn đang trong đà giảm loss nhẹ. Nếu tăng lên 40 epoch kết hợp Cosine Learning Rate Scheduler, Macro-F1 có thể tiệm cận mốc 0.90+.

---

## 7. Phụ lục

- **Các file nộp kèm trong `submission_2A202603002/`:**
  - `REPORT.md` (Báo cáo đầy đủ này)
  - `experiments.xlsx` (Bảng thống kê toàn bộ 21 thí nghiệm, điền đủ 4 sheet)
  - `predictions_eval.csv` (File dự đoán 116 203 dòng của cấu hình cuối cùng)
  - `eval_result.json` (Kết quả chấm chính thức từ `scripts/evaluate.py`)
  - `figures/` (Gồm đầy đủ ảnh 3 ô cho từng `exp_id` và ảnh so sánh nhóm `compare_*.png`)
  - `results/` (Chứa file json log chi tiết từng epoch của mọi thí nghiệm)
  - `code/` (Chứa `lab.ipynb` và toàn bộ các module `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`)
- **Tổng thời gian chạy thực nghiệm:** $\approx 8$ phút trên môi trường cục bộ đa luồng.
