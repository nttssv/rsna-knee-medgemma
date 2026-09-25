# RunPod temporary control key — hướng dẫn cho Luna / Sol / Astra

Đã đối chiếu UI và đường API ngày **25-09-2026, Asia/Singapore**. Mục tiêu: cấp quyền shutdown cho **đúng một pod được người dùng chọn**, lưu key kín, chạy phiên đã duyệt và vô hiệu hóa key sau khi dừng. Đây là SOP vận hành; không phải spending approval và không cho phép tạo/start pod.

## 1. Đọc trước khi thao tác

- Đọc task hiện tại, Pod ID trong console/Connect, session approval và deadline. Không suy ra Pod ID từ tên; không lấy ID/SSH/key của pod lịch sử làm mặc định.
- Phân biệt **approval ngân sách** với **approval tạo credential cấp tài khoản**. Nếu người dùng đã cho phép đúng hành động tạo key tạm trong phiên này, tiếp tục; không hỏi lại từng lệnh. Nếu chưa có, hỏi một lần ngay trước Create: “Cho phép tạo key tạm `<name>`, GraphQL Read/Write cấp tài khoản, AI API None, chỉ dùng read/stop pod `<id>` trong phiên và vô hiệu hóa sau khi dừng?”
- Nếu đã có key được phép dùng trong phiên, kiểm tra key đó trước. **HTTP 403 không tự chứng minh key hỏng.** Làm theo [chẩn đoán 403/1010](RUNPOD_API_TROUBLESHOOTING.md); không tạo vòng lặp key mới vì probe sai header.
- `Restricted` + GraphQL Read/Write vẫn là quyền mạnh **cấp tài khoản**, không phải IAM giới hạn một pod. Giới hạn Pod ID được thực thi bằng script/session của chúng ta. Không quảng cáo quyền này là “pod-scoped”.
- “Temporary” là quy trình vô hiệu hóa sau phiên; **không có TTL tự động được xác minh**. Không để key Enabled sau khi xong. Không cần key AI API/Hugging Face để stop RunPod.

## 2. Tạo đúng quyền trên tab RunPod đang có

Dùng browser tool, không mở cửa sổ mới. Mỗi bước đọc UI hiện tại; không dùng lại element index từ phiên trước.

1. Mở [Credentials](https://console.runpod.io/user/credentials) trong tab hiện có → **API Keys → Create API Key**.
2. Name: tên riêng có ngày giờ, ví dụ `qwen-control-YYYYMMDDTHHMMSSZ`. Tên không chứa credential.
3. **API key options → Restricted**.
4. Dưới **api.runpod.io/graphql → Read / Write**. UI phải ghi `Full access to api.runpod.io/graphql`.
5. Dưới **api.runpod.ai → None**. UI phải ghi `No access to api.runpod.ai`.
6. Kiểm tra cả hai trước **Create**. Không chọn All; Read only không đủ cho shutdown mutation.
7. Sau khi Create thành công, dùng nút **Copy** của key mới. Không paste vào chat, `exec_command`, source, URL/query string, shell history hoặc Markdown. Không in snapshot/screenshot chứa key nguyên văn. Nếu cần đọc UI bằng CUA, lấy trạng thái với `emit:false` rồi che chuỗi credential trước khi hiển thị; không gọi `nodeRepl.write()` trên key/clipboard.

Nếu UI khác hoặc không thể xác định đúng key mới, dừng bước nhập credential và báo field còn thiếu; không đoán hoặc cấp All cho nhanh.

## 3. Lưu local bằng một tab có sẵn — không Save As, không cửa sổ mới

Helper [receive_runpod_key.py](../scripts/receive_runpod_key.py) chỉ nhận **một** key qua loopback. Không gọi RunPod, không mở browser, không đọc clipboard, không start GPU. Helper mới này đã được kiểm tra CPU với key **SYNTHETIC**; chưa dùng để cấp key live trong phiên vừa hoàn tất (phiên đó tái sử dụng key đã được duyệt).

Từ root repo, chạy ngay trước khi paste (thời hạn 120 giây):

```sh
python3 scripts/receive_runpod_key.py --name qwen-control-YYYYMMDDTHHMMSSZ
```

Thay tên timestamp bằng tên thực. Tool sẽ in **một URL `http://127.0.0.1:<port>/`**, không chứa secret. Lấy URL thật từ output, không đoán port và không dùng port viewer cũ. Nếu CLI đang chờ, để process tiếp tục; không chạy lại helper.

1. Sau khi Copy key trên RunPod, dùng **tab local có sẵn** để mở URL helper. Nếu chưa có tab local, tái sử dụng tab RunPod sau khi đã Copy; không tạo cửa sổ/tab mới chỉ để lưu key.
2. Dùng browser tool focus ô **Temporary key**, rồi phím Paste của browser/OS (`Meta+V` trên macOS). Key đi từ clipboard vào ô password, không qua nội dung tool call. **Không đọc/echo clipboard để truyền sang shell.**
3. Click **Save privately**. Kiểm tra trang hiện `Stored locally; no key shown. Intake is now closed.` và process kết thúc với `SAVED_PRIVATE_FILE`.
4. Key được lưu tại `~/.config/rsna-knee/temporary-control/<name>.key`, file `0600`, thư mục user-owned `0700`. Helper từ chối overwrite/symlink, host/origin sai, nonce sai, payload quá dài; tự đóng sau một lần save hoặc hết hạn. Không có endpoint đọc lại key.
5. Xóa clipboard qua browser clipboard API sau khi xác nhận save, nếu API khả dụng. Không dùng shell clipboard hoặc mở ứng dụng khác. Trở lại Credentials; không lưu password vào browser password manager.

Nếu hết hạn, chưa có file và chưa save: chạy lại **local intake**, không cần tạo lại RunPod key nếu bản đã Copy vẫn còn. Nếu không còn key và modal một lần đã đóng: vô hiệu hóa key vừa mất trước khi tạo replacement được phép; không giữ hàng loạt key Enabled. Không ghi đè credential file cũ.

Kiểm tra file **chỉ metadata**, không `cat`:

```sh
export RUNPOD_KEY_FILE="$HOME/.config/rsna-knee/temporary-control/qwen-control-YYYYMMDDTHHMMSSZ.key"
export RUNPOD_POD_ID='EXACT_USER_SELECTED_POD_ID'
python3 - <<'PY'
import os, stat
from pathlib import Path
p=Path(os.environ['RUNPOD_KEY_FILE'])
s=p.lstat()
assert not p.is_symlink() and stat.S_ISREG(s.st_mode)
assert stat.S_IMODE(s.st_mode)==0o600 and s.st_uid==os.getuid()
assert 24 <= s.st_size <= 255
print('PRIVATE_KEY_FILE_READY')
PY
```

Không dùng `export RUNPOD_API_KEY=<literal>`, `curl -H 'Authorization: ...literal...'`, `?api_key=...`, `set -x`, hoặc debug request headers. Chỉ truyền **đường dẫn file** cho scripts.

## 4. Kiểm tra transport trước setup dài

Chạy nguyên [read-only diagnostic](RUNPOD_API_TROUBLESHOOTING.md#read-only-diagnostic-used-successfully), dùng hai biến ở trên. Request phải có:

```text
POST https://api.runpod.io/graphql
Authorization: Bearer <đọc kín từ file>
Content-Type: application/json
User-Agent: rsna-qwen-evidence-selection/1.0
HTTP timeout: 15 seconds
```

Chỉ query đúng pod: `id`, `name`, `desiredStatus`, GPU/count, Secure cloud/region, disk/volume và rates được provider trả. Response có `errors` dù HTTP 200 vẫn là **thất bại**. Không in full pod REST object: có thể chứa Jupyter password/environment secrets. Chỉ lưu/in allowlist field cần thiết trong private state.

| Quan sát | Cách xử lý |
|---|---|
| 403 + Cloudflare `1010`, probe thiếu application User-Agent | Thêm đúng User-Agent như frozen `stop_at.py`; chạy lại read-only probe. Không kết luận invalid key. |
| 401 hoặc GraphQL `UNAUTHORIZED` sau khi đã có đúng header | Kiểm tra key còn Enabled, quyền GraphQL, đúng credential file; không thay bằng key tích hợp pod chưa được kiểm chứng. |
| HTTP 200 nhưng thiếu/null trường cấu hình | Ghi unknown; xác nhận console/Connect. Không điền từ proposal. |
| Read thành công | Xác nhận được read route, **chưa chứng minh một live stop đã thành công**. |
| `costPerHr` vẫn khác 0 trên pod EXITED | Đây có thể là rate cấu hình; kiểm tra trạng thái cùng console current total. Không suy billing từ field đó một mình. |
| Timeout/lỗi sau stop mutation | Kết quả chưa biết: query trạng thái/console trước. Không gửi mutation lại mù quáng. |

Giá storage/current total cần signed-in console; hardware/VRAM/package versions cần kiểm tra thực trên pod. Không copy giá/cache/region từ phiên cũ.

## 5. Watchdog và stop — chỉ dùng runner đã duyệt

Dùng [A/B runner hiện có](../analysis/qwen_evidence_selection_v1/README.md), exact execution-plan/session approval mới, và [SOP phiên chạy thành công](../analysis/qwen_evidence_selection_v1/results/2026-09-25_200748_SGT/SOP.md). Không mở provisioning milestone hay tạo stopped receipt giả.

- Trước setup/cache: xác nhận **đường stop ngoài pod** bằng operator console/API; arm shutdown worker local độc lập và watchdog pod-local bằng `stop_at.py --arm`. Same-pod `/proc` liveness được runner kiểm tra trên pod; PID local không thay thế được PID pod-local.
- Transfer credential qua SSH được xác thực, chỉ tới private `0600` file cần cho watchdog. Không bỏ vào tar source/report/result, command literal hoặc Git. Chỉ tạo session record từ approval thật, không tự reset T0/deadline khi đổi máy hoặc bắt đầu inference.
- Dùng hard inference supervisor, copy/shutdown reserve và số generations của runner. Key creation không cho phép đổi GPU, tăng budget, sửa prompt/parser hoặc generation retry.
- Hoàn tất/thất bại: copy/hash kết quả nếu còn thời gian; **stop ngay**, không chờ hết ngân sách. Deadline shutdown ưu tiên hơn copy.

Lệnh stop đã chạy thành công trong phiên thật (chỉ gọi khi đến bước shutdown đúng pod đã duyệt):

```sh
python3 - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0,'analysis/qwen_evidence_selection_v1/scripts')
from stop_at import read_key, stop_once
result=stop_once(os.environ['RUNPOD_POD_ID'], read_key(Path(os.environ['RUNPOD_KEY_FILE'])))
print(result)  # Chỉ pod ID, trạng thái, timestamp; không có key.
PY
```

`stop_once` gửi một mutation có timeout, không retry tự động. Không stop/start chỉ để test trên pod đang làm việc.

## 6. Đóng phiên và thu hồi quyền

1. Query **đúng Pod ID** độc lập: phải EXITED/STOPPED. Refresh console: Compute/Container storage Not running, **Total $0.00/hour**. “Stop request accepted” hoặc Python đã thoát chưa đủ.
2. Ghi timestamp/readback/console observation private. Chỉ sau xác nhận này mới tạo `operator-verified-stopped.marker` để watchdog local thoát; kiểm tra result `cancelled_after_operator_verified_stop`.
3. Trong Credentials, tìm **đúng tên key phiên này** → menu hàng đó → **Disable key → Yes**. Refresh/đọc UI xác nhận **Disabled**. Không thao tác key khác. Disable là vô hiệu hóa có thể đảo ngược, không phải xóa vĩnh viễn; không tự Enable lại cho phiên sau.
4. Giữ credential ngoài Git và ngoài artifact archive. Có thể xóa local credential đúng file sau khi đã disable nếu phạm vi dọn dẹp cho phép; không xóa theo glob toàn bộ key/history. Không stop/start pod để dọn credential trong container đã dừng.
5. Ghi kết quả planned/attempted/completed/failed/unrun, artifact hashes, cost estimate vs invoice, provider-stop proof và key Disabled. Không đánh dấu hoàn tất nếu thiếu bằng chứng shutdown.

## Checklist bàn giao cực ngắn

```text
[ ] Current user approval covers exact resource/session; separate key-access approval satisfied
[ ] Exact pod ID/Connect confirmed; original clock/budget retained
[ ] Key name unique; Restricted / GraphQL Read-Write / AI API None
[ ] Owner-only file saved; no secret printed, copied to Git, URL or result archive
[ ] Exact-pod read succeeds with application User-Agent; GraphQL errors checked
[ ] Resource/rates and current total independently checked; no invented fields
[ ] Outside-pod stop available; local backstop and required pod watchdog armed
[ ] Run uses approved frozen adapter; no retry/fallback; copy reserve preserved
[ ] Outputs copied and hashed; provider EXITED and console $0/hour verified
[ ] Correct temporary key Disabled; watchdog cancelled only after verified stop
```

**Evidence:** The 25-09-2026 live A40 run used the existing approved control key, completed 20/20 Qwen generations, copied/verified 33 files, stopped successfully via unchanged `stop_at.stop_once()`, showed $0/hour on console, and disabled that key. The new loopback intake helper has four CPU tests (synthetic save, file permissions/overwrite/symlink refusal, origin/nonce checks, and expiry); it is not presented as a live RunPod key-creation test.
