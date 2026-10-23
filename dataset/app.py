"""Interactive Streamlit Data & Slot Attention Inspector for TACO Dataset."""

import os
import sys
from pathlib import Path
import io
import json
import pickle
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import lmdb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

# Setup page layout
st.set_page_config(
    page_title="TACO Dataset & Slot Attention Explorer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Canonical TACO Actor and Ego Labels
from utils.taco_labels import TACO_ACTOR_LABELS, TACO_EGO_LABELS

# Custom CSS for modern styling, zoom modal, and scrolling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-ego { background-color: #E3F2FD; color: #1565C0; border: 1px solid #BBDEFB; }
    .badge-actor { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
    .badge-info { background-color: #F3E5F5; color: #6A1B9A; border: 1px solid #E1BEE7; }
    
    /* Horizontal frame thumbnail scroll container */
    .frame-scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 12px;
        padding: 10px 0px 18px 0px;
        scrollbar-width: thin;
    }
    .frame-scroll-container::-webkit-scrollbar {
        height: 8px;
    }
    .frame-scroll-container::-webkit-scrollbar-thumb {
        background-color: #B0BEC5;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Connecting to LMDB database...")
def get_lmdb_env(lmdb_path: str):
    """Cache and share LMDB read-only environment across reruns."""
    if not os.path.exists(lmdb_path):
        return None
    actual_path = lmdb_path
    if os.path.isdir(lmdb_path):
        for cand in os.listdir(lmdb_path):
            full_cand = os.path.join(lmdb_path, cand)
            if cand.endswith(".lmdb") or os.path.exists(os.path.join(full_cand, "data.mdb")):
                actual_path = full_cand
                break
    return lmdb.open(actual_path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=128)


@st.cache_data(show_spinner="Parsing dataset index and metadata...")
def load_split_index_and_labels(dataset_dir: str, split: str) -> Tuple[Dict, Dict, List[str]]:
    """Parse JSON mappings and build structured scenario metadata.

    Returns:
        scenario_to_frames: dict mapping scenario path to sorted [(filename, frame_idx), ...]
        labels_dict: dict mapping scenario path to {'agents': [...], 'ego': int}
        scenarios_list: list of scenario paths
    """
    labels_path = os.path.join(os.path.dirname(__file__), f"taco_{split}_label.json")
    labels_dict = {}
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)

    # Search for path_to_idx.json
    possible_idx_files = [
        os.path.join(dataset_dir, "taco_path_to_idx.json"),
        os.path.join(dataset_dir, f"taco_{split}_path_to_idx.json"),
        os.path.join(dataset_dir, "path_to_idx.json"),
    ]
    path_to_idx = {}
    for p in possible_idx_files:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                path_to_idx = json.load(f)
            break

    from collections import defaultdict
    scenario_to_frames = defaultdict(list)
    for full_path, lmdb_idx in path_to_idx.items():
        parts = full_path.split('/')
        if 'rgb' in parts:
            rgb_pos = parts.index('rgb')
            var_scenario = parts[rgb_pos - 1]
            scenario_id = parts[rgb_pos - 3]
            cat_or_map = parts[rgb_pos - 4]
            key = f"{cat_or_map}/{scenario_id}/{var_scenario}"
            frame_file = parts[-1]
            scenario_to_frames[key].append((frame_file, lmdb_idx, full_path))
        else:
            key = "/".join(parts[7:11]) if len(parts) >= 11 else full_path
            scenario_to_frames[key].append((parts[-1], lmdb_idx, full_path))

    # Sort frames inside each scenario chronologically
    for k in scenario_to_frames:
        scenario_to_frames[k].sort(key=lambda x: x[0])

    # Canonical scenario list
    scenarios_list = [k for k in labels_dict.keys() if k in scenario_to_frames]
    if not scenarios_list:
        scenarios_list = sorted(list(scenario_to_frames.keys()))

    return dict(scenario_to_frames), labels_dict, scenarios_list


def read_frame_image(env, frame_idx: int) -> Optional[Image.Image]:
    """Read raw RGB image from LMDB by index key."""
    if env is None:
        return None
    try:
        with env.begin(write=False) as txn:
            byteflow = txn.get(f"{frame_idx:08}".encode("ascii"))
            if byteflow is None:
                return None
            entry = pickle.loads(byteflow)
            frame_bytes = entry["frame"] if isinstance(entry, dict) and "frame" in entry else entry
            if isinstance(frame_bytes, bytes):
                return Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            else:
                return Image.fromarray(np.uint8(frame_bytes)).convert("RGB")
    except Exception:
        return None


def generate_attention_overlay(frame_np: np.ndarray, attn_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Superimpose attention heatmap (Turbo / Jet colormap) over original video frame."""
    H, W = frame_np.shape[:2]
    # Resize heatmap to match original frame dimension
    attn_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_LINEAR)
    # Normalize 0..1
    min_v, max_v = attn_resized.min(), attn_resized.max()
    if max_v > min_v:
        attn_norm = (attn_resized - min_v) / (max_v - min_v)
    else:
        attn_norm = np.zeros_like(attn_resized)

    heatmap_uint8 = np.uint8(255 * attn_norm)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * frame_np)
    return blended


# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("## ⚙️ Cấu Hình Dữ Liệu")

split_choice = st.sidebar.radio(
    "1. Chọn Tập Dữ Liệu (Split):",
    options=["train", "val"],
    index=0,
    format_func=lambda x: "🚆 Train Data" if x == "train" else "🧪 Validation Data",
)

default_base_path = f"D:/apps/Datasets/TACO/{split_choice}"
lmdb_folder = st.sidebar.text_input(
    f"2. Đường Dẫn Folder Chứa LMDB & Index ({split_choice}):",
    value=default_base_path,
    help="Thư mục chứa file taco_{split}.lmdb và taco_path_to_idx.json",
)

# Initialize data
scenario_to_frames, labels_dict, scenarios_list = {}, {}, []
lmdb_env = None

if os.path.exists(lmdb_folder):
    lmdb_env = get_lmdb_env(lmdb_folder)
    scenario_to_frames, labels_dict, scenarios_list = load_split_index_and_labels(lmdb_folder, split_choice)
else:
    st.sidebar.error(f"❌ Không tìm thấy thư mục: `{lmdb_folder}`")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Tải Attention Matrix (.npz)")
uploaded_npz = st.sidebar.file_uploader(
    "Tải file attention_matrix.npz (Tùy chọn):",
    type=["npz"],
    help="Tệp tin được lưu từ quá trình huấn luyện/đánh giá chứa ma trận Attention của 64 Slots",
)
manual_npz_path = st.sidebar.text_input(
    "Hoặc nhập đường dẫn file .npz trên đĩa:",
    value="checkpoints/attention_vis/epoch_5_attention_matrix.npz" if os.path.exists("checkpoints/attention_vis/epoch_5_attention_matrix.npz") else "",
)

# Load Attention Matrix if provided
attention_data = None
attention_frames = None
attention_metadata = {}
loaded_npz_source = None

if uploaded_npz is not None:
    try:
        npz = np.load(uploaded_npz, allow_pickle=True)
        attention_data = npz["attention"]
        if "frames" in npz.files:
            attention_frames = npz["frames"]
        if "metadata" in npz.files:
            attention_metadata = npz["metadata"].item() if npz["metadata"].ndim == 0 else {}
        loaded_npz_source = "Uploaded File"
    except Exception as e:
        st.sidebar.error(f"Lỗi đọc file uploaded npz: {e}")
elif manual_npz_path and os.path.exists(manual_npz_path):
    try:
        npz = np.load(manual_npz_path, allow_pickle=True)
        attention_data = npz["attention"]
        if "frames" in npz.files:
            attention_frames = npz["frames"]
        if "metadata" in npz.files:
            attention_metadata = npz["metadata"].item() if npz["metadata"].ndim == 0 else {}
        loaded_npz_source = manual_npz_path
    except Exception as e:
        st.sidebar.error(f"Lỗi đọc file npz từ đường dẫn: {e}")

if attention_data is not None:
    if attention_data.ndim == 5:
        # (B, T, S, H, W)
        batch_size_attn = attention_data.shape[0]
        selected_b = st.sidebar.selectbox("Batch Sample Index:", list(range(batch_size_attn)), index=0)
        curr_attn = attention_data[selected_b]  # (T, S, H, W)
    else:
        curr_attn = attention_data  # (T, S, H, W)

    T_attn, S_attn, H_attn, W_attn = curr_attn.shape
    st.sidebar.success(f"✅ Loaded Attention Matrix: T={T_attn}, Slots={S_attn}, Resolution={H_attn}x{W_attn}")

    selected_slot = st.sidebar.selectbox(
        "🎯 Chọn Slot để xem Heatmap (0 - 63):",
        options=list(range(min(64, S_attn))),
        index=0,
        format_func=lambda s: f"Slot #{s:02d}: {TACO_ACTOR_LABELS[s] if s < len(TACO_ACTOR_LABELS) else f'Class_{s}'}",
    )
    overlay_alpha = st.sidebar.slider("Độ đậm nhạt Heatmap Overlay (Alpha):", 0.0, 1.0, 0.55, 0.05)
else:
    selected_slot = None
    curr_attn = None
    overlay_alpha = 0.55


# --- MAIN CONTENT AREA ---
st.markdown('<div class="main-header">TACO Video Dataset & Slot Attention Inspector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Trực quan hóa toàn bộ frames từ LMDB theo <code>path_to_idx</code>, điều hướng tuần tự, zoom chi tiết và khám phá cơ chế chú ý của 64 Slots.</div>',
    unsafe_allow_html=True,
)

if not scenarios_list:
    st.warning("⚠️ Chưa tìm thấy kịch bản video nào. Vui lòng kiểm tra lại đường dẫn folder LMDB bên thanh điều khiển.")
    st.stop()

# Scenario Navigation State
if "scenario_idx" not in st.session_state:
    st.session_state.scenario_idx = 0

total_scenarios = len(scenarios_list)

# Navigation Controls
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1.2, 1.2, 3.5, 2.5])

with col_nav1:
    if st.button("⬅️ Kịch bản trước", use_container_width=True):
        if st.session_state.scenario_idx > 0:
            st.session_state.scenario_idx -= 1
            st.rerun()

with col_nav2:
    if st.button("Kịch bản sau ➡️", use_container_width=True):
        if st.session_state.scenario_idx < total_scenarios - 1:
            st.session_state.scenario_idx += 1
            st.rerun()

with col_nav3:
    selected_idx = st.number_input(
        f"Chỉ định Index Video (0 - {total_scenarios - 1}):",
        min_value=0,
        max_value=total_scenarios - 1,
        value=st.session_state.scenario_idx,
        step=1,
    )
    if selected_idx != st.session_state.scenario_idx:
        st.session_state.scenario_idx = selected_idx
        st.rerun()

with col_nav4:
    selected_scenario_dropdown = st.selectbox(
        "Tìm kiếm theo Scenario Path:",
        options=scenarios_list,
        index=st.session_state.scenario_idx,
    )
    if selected_scenario_dropdown != scenarios_list[st.session_state.scenario_idx]:
        st.session_state.scenario_idx = scenarios_list.index(selected_scenario_dropdown)
        st.rerun()

current_scenario = scenarios_list[st.session_state.scenario_idx]
frames_meta = scenario_to_frames.get(current_scenario, [])
total_frames = len(frames_meta)

# Display Scenario Information & Ground Truth Labels
st.markdown("---")
col_info1, col_info2 = st.columns([2.5, 3.5])

scenario_label_info = labels_dict.get(current_scenario, {})
ego_action_id = scenario_label_info.get("ego", None)
ego_name = TACO_EGO_LABELS[ego_action_id] if (ego_action_id is not None and ego_action_id < len(TACO_EGO_LABELS)) else "Unknown"

raw_actors = scenario_label_info.get("agents", scenario_label_info.get("actor", []))
active_actor_names = []
if isinstance(raw_actors, list):
    for idx_cls, val in enumerate(raw_actors):
        if val > 0:
            lbl = TACO_ACTOR_LABELS[idx_cls] if idx_cls < len(TACO_ACTOR_LABELS) else f"Actor_{idx_cls}"
            active_actor_names.append((idx_cls, lbl))

with col_info1:
    st.markdown(f"**Scenario Path:** `{current_scenario}`")
    st.markdown(f"**Tổng số Frames tìm thấy:** `{total_frames} frames` | **Dataset Split:** `{split_choice.upper()}`")
    st.markdown(f'<span class="badge badge-ego">🚗 Ego Action [{ego_action_id}]: {ego_name}</span>', unsafe_allow_html=True)

with col_info2:
    st.markdown("**🏷️ Active Ground-Truth Actor Actions:**")
    if active_actor_names:
        badges_html = "".join([f'<span class="badge badge-actor">[{idx}] {name}</span>' for idx, name in active_actor_names])
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-info">Không có nhãn Actor dương tính (No Active Actors)</span>', unsafe_allow_html=True)


# --- LOAD ALL FRAMES ---
st.markdown("### 🎞️ Toàn Bộ Các Frames Trong Video Clip")

if total_frames == 0:
    st.info("Không có frame nào được lập chỉ mục cho kịch bản này.")
else:
    # Load raw PIL images
    images_list = []
    for f_name, lmdb_idx, full_p in frames_meta:
        img = read_frame_image(lmdb_env, lmdb_idx)
        images_list.append((f_name, lmdb_idx, full_p, img))

    # Frame selection radio / state
    if "selected_frame_idx" not in st.session_state:
        st.session_state.selected_frame_idx = 0
    if st.session_state.selected_frame_idx >= total_frames:
        st.session_state.selected_frame_idx = 0

    st.markdown(
        f"*(Hiển thị {total_frames} frames liên quan. Kéo thanh cuộn ngang bên dưới để xem tuần tự, nhấn chọn frame để xem kích thước lớn)*"
    )

    # 1. SCROLLABLE HORIZONTAL THUMBNAIL GALLERY
    # We display thumbnails in columns inside a scrollable container
    thumb_cols = st.columns(min(8, total_frames))
    cols_per_row = 8
    
    # Quick thumbnail strip
    for i in range(0, total_frames, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < total_frames:
                f_name, lmdb_idx, full_p, img = images_list[idx]
                with cols[j]:
                    caption_str = f"#{idx:02d}: {f_name}"
                    if curr_attn is not None and idx < curr_attn.shape[0]:
                        # Overlay attention on thumbnail if available
                        attn_slice = curr_attn[idx, selected_slot]
                        np_img = np.array(img) if img is not None else np.zeros((128, 384, 3), dtype=np.uint8)
                        display_img = generate_attention_overlay(np_img, attn_slice, alpha=overlay_alpha)
                    else:
                        display_img = img

                    if display_img is not None:
                        st.image(display_img, caption=caption_str, use_container_width=True)
                    else:
                        st.error("Frame Error")

                    if st.button(f"🔍 Xem #{idx:02d}", key=f"btn_select_f_{idx}", use_container_width=True):
                        st.session_state.selected_frame_idx = idx
                        st.rerun()

    # 2. DETAILED SINGLE-FRAME INSPECTION & ZOOM
    st.markdown("---")
    st.markdown(f"### 🔎 Chi Tiết & Phóng To Frame `#{st.session_state.selected_frame_idx:02d}`")

    target_fname, target_lmdb_idx, target_full_path, target_img = images_list[st.session_state.selected_frame_idx]

    col_detail_img, col_detail_ctrl = st.columns([3.5, 1.5])

    with col_detail_ctrl:
        st.markdown(f"**Tên file frame:** `{target_fname}`")
        st.markdown(f"**LMDB Key ID:** `{target_lmdb_idx:08d}`")
        st.caption(f"Đường dẫn gốc: `{target_full_path}`")

        zoom_scale = st.slider("Cấp độ phóng to ảnh (Zoom Level):", min_value=1.0, max_value=3.5, value=1.0, step=0.25)
        show_attn_toggle = st.checkbox("Bật lớp phủ Attention Heatmap", value=(curr_attn is not None))

        # Direct navigation between frames
        col_prev_f, col_next_f = st.columns(2)
        with col_prev_f:
            if st.button("⏮️ Frame trước", use_container_width=True):
                if st.session_state.selected_frame_idx > 0:
                    st.session_state.selected_frame_idx -= 1
                    st.rerun()
        with col_next_f:
            if st.button("Frame sau ⏭️", use_container_width=True):
                if st.session_state.selected_frame_idx < total_frames - 1:
                    st.session_state.selected_frame_idx += 1
                    st.rerun()

    with col_detail_img:
        if target_img is not None:
            base_np = np.array(target_img)
            # Apply attention overlay if requested
            if show_attn_toggle and curr_attn is not None:
                if st.session_state.selected_frame_idx < curr_attn.shape[0]:
                    attn_slice = curr_attn[st.session_state.selected_frame_idx, selected_slot]
                    blended_np = generate_attention_overlay(base_np, attn_slice, alpha=overlay_alpha)
                    final_display_img = Image.fromarray(blended_np)
                    st.caption(f"🔥 Hiển thị lớp phủ Attention của **Slot #{selected_slot:02d} ({TACO_ACTOR_LABELS[selected_slot]})**")
                else:
                    st.warning("Số frames trong file .npz ít hơn số frames của kịch bản hiện tại.")
                    final_display_img = target_img
            else:
                final_display_img = target_img

            # Apply Zoom Scaling if > 1.0
            if zoom_scale > 1.0:
                w, h = final_display_img.size
                new_w, new_h = int(w * zoom_scale), int(h * zoom_scale)
                final_display_img = final_display_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            st.image(
                final_display_img,
                caption=f"Frame #{st.session_state.selected_frame_idx:02d} - {target_fname} (Zoom {zoom_scale:.2f}x)",
                use_container_width=True,
            )
        else:
            st.error("Không thể giải mã dữ liệu ảnh từ LMDB.")


# --- 64 SLOTS FULL ATTENTION MATRIX BROWSER ---
if curr_attn is not None:
    st.markdown("---")
    st.markdown("### 🧩 Khám Phá Toàn Diện 64 Slots Attention Matrix")
    st.markdown(
        "Xem toàn bộ phân bố chú ý của tất cả các slots trên kịch bản video này để đánh giá tính diễn giải và mức độ tập trung của từng slot."
    )

    with st.expander("📊 Hiển thị Lưới 64 Slots Heatmap (Tất cả các Slot)", expanded=False):
        slot_cols = st.columns(4)
        num_slots_display = min(64, curr_attn.shape[1])
        t_sample_idx = min(st.session_state.selected_frame_idx, curr_attn.shape[0] - 1)

        for s_idx in range(num_slots_display):
            with slot_cols[s_idx % 4]:
                s_name = TACO_ACTOR_LABELS[s_idx] if s_idx < len(TACO_ACTOR_LABELS) else f"Slot_{s_idx}"
                # Slot heatmap at selected frame
                s_attn_slice = curr_attn[t_sample_idx, s_idx]

                if target_img is not None:
                    base_np = np.array(target_img)
                    s_blended = generate_attention_overlay(base_np, s_attn_slice, alpha=overlay_alpha)
                    st.image(
                        s_blended,
                        caption=f"Slot #{s_idx:02d}: {s_name}",
                        use_container_width=True,
                    )
