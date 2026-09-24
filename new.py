import os
import json
import re

def process_folder_names(dataset_path, output_json="taco_val_data.json"):
    # Định nghĩa Regex cho 3 dạng tên folder
    # - Group 1: Bắt tiền tố (ap_TownXX, interactive, runner_TownXX)
    # - Group 2 (.*): Bắt phần giữa, có thể chứa nhiều dấu '_' (Tham lam / Greedy)
    # - Group 3 ([^_]+$): Bắt phần ID cuối cùng sau dấu '_' cùng
    patterns = [
        re.compile(r"^(ap_Town\d+)_(.*)_([^_]+)$"),
        re.compile(r"^(interactive)_(.*)_([^_]+)$"),
        re.compile(r"^(runner_Town\d+)_(.*)_([^_]+)$")
    ]

    converted_list = []

    # Kiểm tra đường dẫn tồn tại
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Đường dẫn '{dataset_path}' không tồn tại.")
        return

    # Quét qua các thư mục con trong path
    for item_name in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item_name)

        # Chỉ thao tác với folder, bỏ qua các file
        if os.path.isdir(item_path):
            for pattern in patterns:
                match = pattern.match(item_name)
                if match:
                    # Ráp 3 group lại bằng dấu '/' theo format mong muốn
                    new_format = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
                    converted_list.append(new_format)
                    break # Tìm thấy form khớp thì ngưng duyệt pattern khác cho folder này

    # Ghi mảng (list) kết quả ra file JSON
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(converted_list, json_file, indent=4)

    print(f"✅ Đã xử lý xong! Tìm thấy {len(converted_list)} folder hợp lệ.")
    print(f"📁 Dữ liệu đã được xuất ra file: {output_json}")

# --- CÁCH SỬ DỤNG ---
# Thay '/path/to/your/dataset' bằng đường dẫn thực tế chứa các folder con của bạn
process_folder_names(r'D:\Documents\Study\NYCU\Action-slot\taco_eval\action_slot\num_slots64_obj_maskFalse\plot_occlusion_0')