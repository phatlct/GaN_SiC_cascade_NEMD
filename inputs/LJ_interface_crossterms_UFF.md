# Tham số LJ cho tương tác chéo tại giao diện GaN/SiC (Ga-Si, Ga-C, N-Si, N-C)

## Kết luận chính: CÓ, bắt buộc cần potentials, và đã xác định được bộ đầy đủ

Hệ 4 nguyên tố Ga-N-Si-C cần:
1. Thế năng cho khối GaN (Ga-Ga, Ga-N, N-N)
2. Thế năng cho khối SiC (Si-Si, Si-C, C-C)
3. Thế năng cho tương tác CHÉO tại giao diện (Ga-Si, Ga-C, N-Si, N-C) — đây là phần **không có sẵn công khai** cho cặp GaN-SiC, đúng như rủi ro đã nêu trong đề xuất ban đầu.

## Bộ giải pháp đã xác nhận qua tài liệu thật

### 1. GaN.tersoff — ĐÃ TẢI, nguồn xác thực
Nord, J.; Albe, K.; Erhart, P.; Nordlund, K. *J. Phys.: Condens. Matter* 15, 5649 (2003).
File chính thức phân phối kèm LAMMPS (Sandia National Labs, contributor Xiaowang Zhou):
https://raw.githubusercontent.com/lammps/lammps/develop/potentials/GaN.tersoff
→ Đã lưu tại `01_Code_MD_Simulation/02_Force_Field/GaN.tersoff`

### 2. SiC.tersoff — ĐÃ TẢI, nguồn xác thực
Tersoff, J. *Phys. Rev. B* 39, 5566–5568 (1989), hiệu đính PRB 41, 3248.
File chính thức phân phối kèm LAMMPS (contributor Aidan Thompson, Sandia):
https://raw.githubusercontent.com/lammps/lammps/develop/potentials/SiC.tersoff
→ Đã lưu tại `01_Code_MD_Simulation/02_Force_Field/SiC.tersoff`

**Quan trọng**: Đổi từ đề xuất ban đầu (Vashishta cho SiC) sang dùng Tersoff cho cả 2 vật liệu, vì:
- Cả GaN.tersoff và SiC.tersoff đều là cùng dạng hàm Tersoff → tương thích khi khai báo trong LAMMPS bằng 2 instance riêng của `pair_style hybrid/overlay tersoff tersoff`.
- Vashishta (dạng hàm khác Tersoff hoàn toàn) không thể kết hợp trực tiếp với Tersoff bằng cùng 1 pair_coeff mixing — sẽ phức tạp hơn khi cần tương tác chéo.

### 3. Tương tác chéo giao diện (Ga-Si, Ga-C, N-Si, N-C) — dùng Lennard-Jones "lớp keo" (interfacial glue layer)

**Căn cứ phương pháp (không tự bịa)**: tra cứu thực tế cho thấy đây là cách làm chuẩn trong các bài báo GaN/diamond đã công bố — "tương tác giữa nguyên tử GaN và diamond tại giao diện được đặc trưng bằng thế năng Lennard-Jones" trong khi khối GaN dùng EAM và khối diamond dùng Tersoff (theo tổng hợp từ nhiều bài báo GaN/diamond MD, xem bibliography.md mục B). Tài liệu cũng ghi nhận rõ đây là giải pháp có đánh đổi độ chính xác ("mixing different types of potentials... usually accompanied by a decrease in simulation accuracy") — CẦN NÊU RÕ HẠN CHẾ NÀY khi viết Methodology.

**Tham số LJ cụ thể** — tính từ quy tắc kết hợp (combining rule) chuẩn của UFF (Universal Force Field, Rappé et al., *J. Am. Chem. Soc.* 1992, 114, 10024–10035), lấy dữ liệu gốc từ bảng tham số UFF (nguồn: Open Babel UFF.prm, https://raw.githubusercontent.com/openbabel/openbabel/master/data/UFF.prm — triển khai mã nguồn mở công khai của bảng UFF gốc):

| Nguyên tố | x1 (Rmin, Å) | D1 (kcal/mol) |
|---|---|---|
| C  | 3.851 | 0.105 |
| N  | 3.660 | 0.069 |
| Si | 4.295 | 0.402 |
| Ga | 4.383 | 0.415 |

Quy tắc kết hợp UFF (geometric mean): x1_ij = sqrt(x1_i × x1_j), D1_ij = sqrt(D1_i × D1_j)
Chuyển đổi sang LAMMPS `pair_style lj/cut` (units metal, eV/Angstrom):
sigma = x1_ij / 2^(1/6);  epsilon [eV] = D1_ij [kcal/mol] × 0.0433641

**Kết quả tính toán (đã kiểm tra bằng Python)**:

| Cặp | x1_ij (Å) | D1_ij (kcal/mol) | sigma (Å) | epsilon (eV) |
|---|---|---|---|---|
| Ga-Si | 4.3388 | 0.4084 | 3.8654 | 0.01771 |
| Ga-C  | 4.1084 | 0.2087 | 3.6602 | 0.00905 |
| N-Si  | 3.9648 | 0.1665 | 3.5322 | 0.00722 |
| N-C   | 3.7543 | 0.0851 | 3.3447 | 0.00369 |

## CẢNH BÁO QUAN TRỌNG — mức độ tin cậy của tham số LJ này

Đây là ước lượng bằng quy tắc kết hợp UFF tổng quát (dùng cho hoá học phân tử, KHÔNG fit riêng cho vật liệu rắn/ceramic). Đây là điểm khởi đầu hợp lý để chạy thử nghiệm (proof-of-concept) nhưng **PHẢI hiệu chỉnh lại trước khi dùng cho production/paper chính thức**, cụ thể:
1. Kiểm tra năng lượng bám dính giao diện (adhesion/interfacial energy) tính được có hợp lý so với năng lượng liên kết thực nghiệm hoặc DFT không (nếu có tài liệu).
2. Nếu có thể, tính DFT cho vài cấu hình giao diện nhỏ (slab mỏng) để fit lại epsilon/sigma thay vì dùng UFF thô.
3. Xem xét quét độ nhạy (sensitivity scan): chạy NEMD với epsilon dao động ±30-50% để đánh giá TBC nhạy thế nào với tham số giao diện — đây cũng là một phần kết quả đáng báo cáo trong bài (uncertainty quantification).
4. Trích dẫn rõ trong Methodology: tham số liên kết chéo GaN-SiC KHÔNG có trong tài liệu công bố tính đến thời điểm tra cứu (04/08/2026), là hạn chế đã biết và đã xử lý bằng phương pháp gần đúng UFF + LJ, tương tự cách tiếp cận cho giao diện GaN/diamond trong y văn.
