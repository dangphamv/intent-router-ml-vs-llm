# Hướng dẫn gán nhãn

Bối cảnh: tin nhắn khách hàng gửi vào kênh hỗ trợ của một phần mềm SaaS quản lý dự án (gói Free / Pro / Business / Enterprise, tính tiền theo seat).

| Nhãn | Định nghĩa | Ví dụ |
|---|---|---|
| `pricing` | Hỏi giá, gói, hạn mức, giảm giá, phương thức thanh toán, báo giá **trước khi mua / nâng cấp**. | "Gói Pro có bao nhiêu seat?" |
| `complaint` | Bày tỏ bất mãn, phản ánh sai sót (trừ tiền sai, hỗ trợ chậm, sự cố gây thiệt hại), đòi hoàn tiền / bồi thường — **không** yêu cầu chấm dứt dịch vụ. | "Sao tháng này bị trừ tiền 2 lần vậy?" |
| `cancellation` | Muốn hủy, hạ gói, bớt seat, tạm ngưng, tắt gia hạn, xóa / đóng tài khoản; hoặc hỏi thủ tục / hệ quả của việc hủy. | "Làm sao để tắt tự động gia hạn?" |
| `tech_support` | Hỏi cách dùng, báo lỗi, đăng nhập, tích hợp, API — giọng trung tính, cần được hướng dẫn / sửa. | "API trả về lỗi 401 dù token còn hạn" |
| `other` | Chào hỏi, cảm ơn, góp ý tính năng, tuyển dụng, hợp tác, câu ngoài lề, tin vô nghĩa. | "Cảm ơn nhiều nhé" |

## Quy tắc phân xử

1. **Có ý định hủy → `cancellation`**, kể cả khi kèm lời phàn nàn ("Dịch vụ tệ quá, hủy luôn cho tôi").
2. **Lỗi kỹ thuật**: giọng trung tính, cần hướng dẫn → `tech_support`; trọng tâm là bức xúc / đòi trách nhiệm → `complaint`.
3. **Tiền**: hỏi giá trước khi mua → `pricing`; bị tính sai / phí bất ngờ → `complaint`.
4. Góp ý tính năng, khen ngợi → `other`.
