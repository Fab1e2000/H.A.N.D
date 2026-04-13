import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

try:
    import matplotlib
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:  # pragma: no cover
    matplotlib = None
    FigureCanvasTkAgg = None
    Figure = None
    Poly3DCollection = None

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover
    serial = None
    list_ports = None


class Kalman1D:
    def __init__(self, process_var: float = 1e-3, measure_var: float = 2e-2) -> None:
        self.process_var = float(process_var)
        self.measure_var = float(measure_var)
        self.x = 0.0
        self.p = 1.0
        self.initialized = False

    def reset(self) -> None:
        self.x = 0.0
        self.p = 1.0
        self.initialized = False

    def update(self, z: float) -> float:
        zf = float(z)
        if not self.initialized:
            self.x = zf
            self.p = 1.0
            self.initialized = True
            return self.x

        self.p = self.p + self.process_var
        k = self.p / (self.p + self.measure_var)
        self.x = self.x + k * (zf - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


class ServoControllerApp:
    @dataclass(frozen=True)
    class ChannelMeta:
        label: str
        joint_type: str

    ACTIVE_CHANNELS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    LATERAL_CHANNELS = {2, 6, 10, 14}
    CHANNEL_META = {
        0: ChannelMeta("拇指-远指节", "distal"),
        1: ChannelMeta("拇指-近指节", "proximal"),
        2: ChannelMeta("拇指-侧摆节", "lateral"),
        4: ChannelMeta("食指-远指节", "distal"),
        5: ChannelMeta("食指-近指节", "proximal"),
        6: ChannelMeta("食指-侧摆节", "lateral"),
        8: ChannelMeta("中指-远指节", "distal"),
        9: ChannelMeta("中指-近指节", "proximal"),
        10: ChannelMeta("中指-侧摆节", "lateral"),
        12: ChannelMeta("无名指-远指节", "distal"),
        13: ChannelMeta("无名指-近指节", "proximal"),
        14: ChannelMeta("无名指-侧摆节", "lateral"),
    }
    LEFT_PANEL_WIDTH = 920
    CHANNEL_BLOCK_HEIGHT = 300
    CALIB_BLOCK_HEIGHT = 95
    CONTROL_BLOCK_HEIGHT = 355
    BLOCK_V_GAP = 10
    LEFT_BLOCKS_TOTAL_HEIGHT = CHANNEL_BLOCK_HEIGHT + CALIB_BLOCK_HEIGHT + CONTROL_BLOCK_HEIGHT + BLOCK_V_GAP * 2
    MODEL_PANEL_SIZE = LEFT_BLOCKS_TOTAL_HEIGHT
    VIDEO_PANEL_SIZE = MODEL_PANEL_SIZE
    BIO_PANEL_WIDTH = VIDEO_PANEL_SIZE // 2
    MODEL_REDRAW_MS = 50
    TABLE_REFRESH_MS = 120
    VIDEO_REFRESH_MS = 33
    CAMERA_INDEX = 0
    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    PALM_ORIGIN = (0.0, 0.0, 0.0)
    FINGER_LINEWIDTH_3D = 6
    FINGER_MARKER_SIZE_3D = 48
    FINGER_LINEWIDTH_2D = 6
    FINGER_MARKER_SIZE_2D = 48
    JOINT_COLOR = "#000000"
    SERVO_ANGLE_MIN = 0.0
    SERVO_ANGLE_MAX = 180.0
    FINGER_GROUPS = (
        ("拇指", (0, 1, 2)),
        ("食指", (4, 5, 6)),
        ("中指", (8, 9, 10)),
        ("无名指", (12, 13, 14)),
    )
    PINCH_FINGERS = ("index", "middle", "ring")
    PINCH_FINGER_LABELS = {
        "index": "食指",
        "middle": "中指",
        "ring": "无名指",
    }
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    FINGERS_LANDMARKS = {
        "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
        "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    }
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    )

    @staticmethod
    def _runtime_base_dir() -> Path:
        # In frozen mode, keep configs beside the executable for persistence.
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    def _default_ui_layout(self) -> dict[str, int]:
        return {
            "window_width": 2400,
            "window_height": 1200,
            "min_width": 2400,
            "min_height": 1200,
            "left_zone_width": 600,
            "middle_zone_width": 900,
            "right_zone_width": 900,
            "zone_height": 1100,
            "channel_block_height": 250,
            "calib_block_height": 110,
            "control_block_height": 350,
            "bio_calib_block_height": 310,
            "block_v_gap": 10,
            "model_panel_size": 850,
            "video_panel_size": 850,
            "body_padding": 12,
            "camera_index": 0,
            "video_refresh_ms": 33,
            "table_refresh_ms": 120,
            "model_redraw_ms": 50,
            "control_scale_length": 620,
            "landmark_process_var": 0.0006,
            "landmark_measure_var": 0.0025,
            "distance_process_var": 0.0008,
            "distance_measure_var": 0.003,
            "joint_process_var": 0.8,
            "joint_measure_var": 7.0,
            "serial_panel_height": 220,
            "serial_log_max_lines": 300,
            "tracking_send_interval_ms": 70,
            "tracking_log_interval_ms": 180,
        }

    def _normalize_ui_layout(self, raw: object) -> dict[str, int | float]:
        normalized = self._default_ui_layout()
        if not isinstance(raw, dict):
            return normalized
        non_negative_int_keys = {"camera_index"}
        positive_float_keys = {
            "landmark_process_var",
            "landmark_measure_var",
            "distance_process_var",
            "distance_measure_var",
            "joint_process_var",
            "joint_measure_var",
        }
        for key, default in normalized.items():
            val = raw.get(key)
            if key in non_negative_int_keys:
                try:
                    normalized[key] = max(0, int(val))
                except Exception:
                    normalized[key] = default
                continue
            if key in positive_float_keys:
                try:
                    normalized[key] = max(1e-9, float(val))
                except Exception:
                    normalized[key] = default
                continue
            try:
                v = int(val)
                normalized[key] = max(1, v)
            except Exception:
                normalized[key] = default
        return normalized

    def _load_ui_layout(self) -> dict[str, int]:
        if not self.ui_layout_file.exists():
            default = self._default_ui_layout()
            try:
                with self.ui_layout_file.open("w", encoding="utf-8") as f:
                    json.dump(default, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return default

        try:
            with self.ui_layout_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            return self._default_ui_layout()
        return self._normalize_ui_layout(loaded)

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PC-Arduino Servo Controller")
        base_dir = self._runtime_base_dir()
        self.ui_layout_file = base_dir / "ui_layout.json"
        self.ui_layout = self._load_ui_layout()

        self.window_width = self.ui_layout["window_width"]
        self.window_height = self.ui_layout["window_height"]
        self.min_width = self.ui_layout["min_width"]
        self.min_height = self.ui_layout["min_height"]
        self.left_zone_width = self.ui_layout["left_zone_width"]
        self.middle_zone_width = self.ui_layout["middle_zone_width"]
        self.right_zone_width = self.ui_layout["right_zone_width"]
        self.zone_height = self.ui_layout["zone_height"]
        self.channel_block_height = self.ui_layout["channel_block_height"]
        self.calib_block_height = self.ui_layout["calib_block_height"]
        self.control_block_height = self.ui_layout["control_block_height"]
        self.bio_calib_block_height = self.ui_layout["bio_calib_block_height"]
        self.block_v_gap = self.ui_layout["block_v_gap"]
        self.left_blocks_total_height = (
            self.channel_block_height
            + self.calib_block_height
            + self.control_block_height
            + self.bio_calib_block_height
            + self.block_v_gap * 3
        )
        self.model_panel_size = self.ui_layout["model_panel_size"]
        self.video_panel_size = self.ui_layout["video_panel_size"]
        self.body_padding = self.ui_layout["body_padding"]
        self.camera_index = int(self.ui_layout["camera_index"])
        self.video_refresh_ms = int(self.ui_layout["video_refresh_ms"])
        self.table_refresh_ms = int(self.ui_layout["table_refresh_ms"])
        self.model_redraw_ms = int(self.ui_layout["model_redraw_ms"])
        self.control_scale_length = int(self.ui_layout["control_scale_length"])
        self.landmark_process_var = float(self.ui_layout["landmark_process_var"])
        self.landmark_measure_var = float(self.ui_layout["landmark_measure_var"])
        self.distance_process_var = float(self.ui_layout["distance_process_var"])
        self.distance_measure_var = float(self.ui_layout["distance_measure_var"])
        self.joint_process_var = float(self.ui_layout["joint_process_var"])
        self.joint_measure_var = float(self.ui_layout["joint_measure_var"])
        self.serial_panel_height = int(self.ui_layout["serial_panel_height"])
        self.serial_log_max_lines = int(self.ui_layout["serial_log_max_lines"])
        self.tracking_send_interval_ms = int(self.ui_layout["tracking_send_interval_ms"])
        self.tracking_log_interval_ms = int(self.ui_layout["tracking_log_interval_ms"])

        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.minsize(self.min_width, self.min_height)
        self.root.resizable(False, False)

        self.ser = None
        self.send_after_id = None
        self.table_after_id = None
        self.last_payload_sent = ""

        self.port_var = tk.StringVar(value="COM3")
        self.baud_var = tk.StringVar(value="115200")
        self.channel_vars = {ch: tk.BooleanVar(value=(ch == 0)) for ch in self.ACTIVE_CHANNELS}
        self.real_angle_vars = {ch: tk.StringVar(value="关节角度: --") for ch in self.ACTIVE_CHANNELS}
        self.flex_var = tk.IntVar(value=50)
        self.calib_channel_var = tk.StringVar(value="0")
        self.calib_min_var = tk.IntVar(value=0)
        self.calib_max_var = tk.IntVar(value=180)
        self.realtime_info_var = tk.StringVar(value="请选择至少一个舵机")
        self.status_var = tk.StringVar(value="未连接")

        self.calibration_file = base_dir / "servo_calibration.json"
        self.hand_model_file = base_dir / "hand_model_config.json"
        self.distance_calibration_file = base_dir / "distance_calibration.json"
        self.hand_landmarker_file = base_dir / "hand_landmarker.task"
        self.calibration = self._load_calibration()
        self.hand_model_config = self._load_hand_model_config()
        self.distance_calibration = self._load_distance_calibration()
        self.channel_real_angles = {ch: self._map_flex_to_real_angle(ch, self.flex_var.get()) for ch in self.ACTIVE_CHANNELS}
        self.control_table_items: dict[int, str] = {}
        self.model_figure = None
        self.model_axis = None
        self.model_axes_2d: dict[str, object] = {}
        self.model_canvas = None
        self.model_redraw_after_id = None
        self.model_finger_order: tuple[str, ...] = ()
        self.model_finger_artists: list[dict[str, object]] = []
        self.model_palm_artist_3d = None
        self.model_palm_artists_xy: list[object] = []
        self.model_palm_artists_xz: list[object] = []
        self.model_palm_artists_yz: list[object] = []
        self.track_enabled = False
        self.video_after_id = None
        self.video_canvas = None
        self.video_canvas_image_id = None
        self.video_canvas_text_id = None
        self.video_image_ref: ImageTk.PhotoImage | None = None
        self.serial_log_text: tk.Text | None = None
        self.video_cap = None
        self.camera_backend_name = ""
        self.camera_open_error = ""
        self.hand_backend_type = "none"
        self.hand_backend = None
        self.last_thumb_distances: dict[str, float] = {}
        self.tracking_channel_states: dict[int, dict[str, object]] = {}
        self.image_landmark_filters = self._build_landmark_filters(
            process_var=self.landmark_process_var,
            measure_var=self.landmark_measure_var,
        )
        self.world_landmark_filters = self._build_landmark_filters(
            process_var=self.landmark_process_var,
            measure_var=self.landmark_measure_var,
        )
        self.distance_filters = {
            finger: Kalman1D(process_var=self.distance_process_var, measure_var=self.distance_measure_var)
            for finger in self.PINCH_FINGERS
        }
        self.tracking_angle_filters = {
            ch: Kalman1D(process_var=self.joint_process_var, measure_var=self.joint_measure_var)
            for ch in self.ACTIVE_CHANNELS
        }
        self.tracking_last_send_ts = 0.0
        self.tracking_last_log_ts = 0.0
        self.tracking_last_error_ts = 0.0
        self.manual_controls: list[object] = []

        self._configure_theme()
        self._init_handtracking_backend()
        self._build_ui()
        self._refresh_real_angle_labels()
        self._load_selected_calibration()
        self._refresh_ports()
        self._schedule_video_frame()

    def _configure_theme(self) -> None:
        style = ttk.Style()
        for theme_name in ("vista", "xpnative", "clam"):
            if theme_name in style.theme_names():
                style.theme_use(theme_name)
                break
        style.configure("TLabelFrame", padding=8)
        style.configure("TButton", padding=(10, 4))
        style.configure("TLabel", padding=(1, 1))

        if matplotlib is not None:
            matplotlib.rcParams["font.sans-serif"] = [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ]
            matplotlib.rcParams["axes.unicode_minus"] = False

    def _load_hand_model_config(self) -> dict[str, dict[str, object]]:
        default = {
            "fingers": {
                "thumb": {
                    "label": "拇指",
                    "channels": {"distal": 0, "proximal": 1, "lateral": 2},
                    "root": [-35.0, -5.0, 0.0],
                    "base_direction": [0.9, 0.4, 0.0],
                    "segment_lengths": [30.0, 24.0, 18.0],
                },
                "index": {
                    "label": "食指",
                    "channels": {"distal": 4, "proximal": 5, "lateral": 6},
                    "root": [-12.0, 0.0, 0.0],
                    "base_direction": [0.0, 1.0, 0.0],
                    "segment_lengths": [34.0, 26.0, 20.0],
                },
                "middle": {
                    "label": "中指",
                    "channels": {"distal": 8, "proximal": 9, "lateral": 10},
                    "root": [8.0, 0.0, 0.0],
                    "base_direction": [0.0, 1.0, 0.0],
                    "segment_lengths": [36.0, 28.0, 22.0],
                },
                "ring": {
                    "label": "无名指",
                    "channels": {"distal": 12, "proximal": 13, "lateral": 14},
                    "root": [27.0, -2.0, 0.0],
                    "base_direction": [0.0, 1.0, 0.0],
                    "segment_lengths": [33.0, 25.0, 19.0],
                },
            }
        }

        if not self.hand_model_file.exists():
            return default

        try:
            with self.hand_model_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("fingers"), dict):
                return loaded
        except Exception:
            pass
        return default

    def _default_distance_calibration(self) -> dict[str, dict[str, float | None]]:
        return {
            finger: {
                "far": None,
                "near": None,
                "lateral_far": None,
                "lateral_near": None,
                "thumb_lateral_far": None,
                "thumb_lateral_near": None,
            }
            for finger in self.PINCH_FINGERS
        }

    def _normalize_distance_calibration(self, raw: object) -> dict[str, dict[str, float | None]]:
        normalized = self._default_distance_calibration()
        if not isinstance(raw, dict):
            return normalized

        for finger in self.PINCH_FINGERS:
            item = raw.get(finger)
            if not isinstance(item, dict):
                continue
            for key in normalized[finger].keys():
                val = item.get(key)
                if val is None:
                    normalized[finger][key] = None
                    continue
                try:
                    normalized[finger][key] = float(val)
                except Exception:
                    normalized[finger][key] = None
        return normalized

    def _save_distance_calibration(self) -> None:
        payload = self._normalize_distance_calibration(self.distance_calibration)
        with self.distance_calibration_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_distance_calibration(self) -> dict[str, dict[str, float | None]]:
        if not self.distance_calibration_file.exists():
            default = self._default_distance_calibration()
            try:
                with self.distance_calibration_file.open("w", encoding="utf-8") as f:
                    json.dump(default, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return default

        try:
            with self.distance_calibration_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            return self._default_distance_calibration()
        return self._normalize_distance_calibration(loaded)

    def _ensure_hand_landmarker_model(self) -> Path:
        if self.hand_landmarker_file.exists():
            return self.hand_landmarker_file
        urllib.request.urlretrieve(self.HAND_MODEL_URL, self.hand_landmarker_file)
        return self.hand_landmarker_file

    def _init_handtracking_backend(self) -> None:
        # Prefer legacy solutions API, fallback to tasks API.
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.hand_backend = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
                self.hand_backend_type = "solutions"
                return

            from mediapipe.tasks import python as mp_tasks

            model_path = self._ensure_hand_landmarker_model()
            options = mp_tasks.vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_tasks.vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.hand_backend = mp_tasks.vision.HandLandmarker.create_from_options(options)
            self.hand_backend_type = "tasks"
        except Exception:
            self.hand_backend = None
            self.hand_backend_type = "none"

    def _open_video_capture(self) -> bool:
        if self.video_cap is not None and self.video_cap.isOpened():
            self.camera_open_error = ""
            return True
        self.camera_backend_name = ""

        backends: list[tuple[str, int | None]] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DSHOW", int(cv2.CAP_DSHOW)))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("MSMF", int(cv2.CAP_MSMF)))
        if hasattr(cv2, "CAP_ANY"):
            backends.append(("ANY", int(cv2.CAP_ANY)))
        backends.append(("DEFAULT", None))

        tried: list[str] = []
        for backend_name, backend_id in backends:
            try:
                if backend_id is None:
                    cap_try = cv2.VideoCapture(self.camera_index)
                else:
                    cap_try = cv2.VideoCapture(self.camera_index, backend_id)

                if cap_try.isOpened():
                    self.video_cap = cap_try
                    self.camera_backend_name = backend_name
                    self.camera_open_error = ""
                    return True

                tried.append(f"{backend_name}:closed")
                cap_try.release()
            except Exception as exc:
                tried.append(f"{backend_name}:{exc}")

        self.camera_open_error = "; ".join(tried)
        self.video_cap = None
        return False

    def _extract_hand_landmarks(self, frame_rgb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.hand_backend is None:
            return None, None
        if self.hand_backend_type == "solutions":
            result = self.hand_backend.process(frame_rgb)
            if result.multi_hand_landmarks and result.multi_hand_world_landmarks:
                image_hand = result.multi_hand_landmarks[0]
                world_hand = result.multi_hand_world_landmarks[0]
                image_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in image_hand.landmark], dtype=np.float32)
                world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in world_hand.landmark], dtype=np.float32)
                return image_landmarks, world_landmarks
            return None, None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.hand_backend.detect(mp_image)
        if result.hand_landmarks and result.hand_world_landmarks:
            image_hand = result.hand_landmarks[0]
            world_hand = result.hand_world_landmarks[0]
            image_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in image_hand], dtype=np.float32)
            world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in world_hand], dtype=np.float32)
            return image_landmarks, world_landmarks
        return None, None

    def _build_landmark_filters(self, process_var: float, measure_var: float) -> dict[tuple[int, int], Kalman1D]:
        filters: dict[tuple[int, int], Kalman1D] = {}
        for landmark_idx in range(21):
            for axis in range(3):
                filters[(landmark_idx, axis)] = Kalman1D(process_var=process_var, measure_var=measure_var)
        return filters

    def _apply_landmark_filters(
        self,
        landmarks: np.ndarray,
        filters: dict[tuple[int, int], Kalman1D],
    ) -> np.ndarray:
        filtered = np.empty_like(landmarks)
        for i in range(landmarks.shape[0]):
            for axis in range(landmarks.shape[1]):
                filtered[i, axis] = filters[(i, axis)].update(float(landmarks[i, axis]))
        return filtered

    def _reset_tracking_filters(self) -> None:
        for filt in self.image_landmark_filters.values():
            filt.reset()
        for filt in self.world_landmark_filters.values():
            filt.reset()
        for filt in self.distance_filters.values():
            filt.reset()
        for filt in self.tracking_angle_filters.values():
            filt.reset()

    def _build_thumb_pair_distances(self, world_landmarks: np.ndarray) -> dict[str, float]:
        thumb_primary = world_landmarks[self.FINGERS_LANDMARKS["thumb"][3]]
        thumb_dip_joint = world_landmarks[self.THUMB_IP]
        blended: dict[str, float] = {}
        for finger in self.PINCH_FINGERS:
            finger_tip = world_landmarks[self.FINGERS_LANDMARKS[finger][3]]
            d_primary = float(np.linalg.norm(thumb_primary - finger_tip))
            d_dip = float(np.linalg.norm(thumb_dip_joint - finger_tip))
            blended[finger] = 0.5 * d_primary + 0.5 * d_dip
        return blended

    def _joint_limit_from_calibration(self, channel: int, fallback: tuple[float, float]) -> tuple[float, float]:
        cfg = self.calibration.get(channel)
        if not isinstance(cfg, dict):
            return fallback
        joint_cfg = cfg.get("joint")
        if not isinstance(joint_cfg, dict):
            return fallback
        try:
            lo = float(joint_cfg.get("min"))
            hi = float(joint_cfg.get("max"))
        except Exception:
            return fallback
        if lo <= hi:
            return (lo, hi)
        return (hi, lo)

    def _alpha_from_distance(self, finger: str, dist: float) -> float:
        cal = self.distance_calibration.get(finger, {})
        far = cal.get("far")
        near = cal.get("near")
        if far is None or near is None or abs(float(far) - float(near)) < 1e-6:
            far = 0.14
            near = 0.025
        far_v = float(far)
        near_v = float(near)
        if far_v < near_v:
            far_v, near_v = near_v, far_v
        alpha = (far_v - float(dist)) / max(far_v - near_v, 1e-6)
        return float(np.clip(alpha, 0.0, 1.0))

    def _interpolate_lateral_from_calibration(
        self,
        finger: str,
        alpha: float,
        finger_limits: tuple[float, float],
        thumb_limits: tuple[float, float],
    ) -> tuple[float, float]:
        cal = self.distance_calibration.get(finger, {})
        f_far = cal.get("lateral_far")
        f_near = cal.get("lateral_near")
        t_far = cal.get("thumb_lateral_far")
        t_near = cal.get("thumb_lateral_near")

        if f_far is None or f_near is None:
            finger_target = finger_limits[0] + (finger_limits[1] - finger_limits[0]) * alpha
        else:
            finger_target = float(f_far) + (float(f_near) - float(f_far)) * alpha

        if t_far is None or t_near is None:
            thumb_target = 0.0
        else:
            thumb_target = float(t_far) + (float(t_near) - float(t_far)) * alpha

        finger_target = float(np.clip(finger_target, finger_limits[0], finger_limits[1]))
        thumb_target = float(np.clip(thumb_target, thumb_limits[0], thumb_limits[1]))
        return finger_target, thumb_target

    def _map_real_angle_to_flex(self, channel: int, joint_real_angle: float) -> float:
        cfg = self.calibration.get(channel, self._normalize_channel_calibration(channel, {}))
        mapping = cfg.get("mapping", []) if isinstance(cfg, dict) else []
        points: list[tuple[float, float]] = []
        if isinstance(mapping, list):
            for item in mapping:
                if not isinstance(item, dict):
                    continue
                try:
                    degree = float(item.get("degree"))
                    angle = float(item.get("joint_angle"))
                except Exception:
                    continue
                points.append((degree, angle))

        if len(points) < 2:
            lo, hi = self._joint_real_angle_range(channel)
            span = max(hi - lo, 1e-6)
            return float(np.clip((float(joint_real_angle) - lo) * 100.0 / span, 0.0, 100.0))

        points.sort(key=lambda x: x[1])
        x = float(joint_real_angle)
        if x <= points[0][1]:
            return float(np.clip(points[0][0], 0.0, 100.0))
        if x >= points[-1][1]:
            return float(np.clip(points[-1][0], 0.0, 100.0))

        for idx in range(1, len(points)):
            d0, a0 = points[idx - 1]
            d1, a1 = points[idx]
            if x <= a1:
                span = a1 - a0
                if abs(span) <= 1e-9:
                    return float(np.clip(d1, 0.0, 100.0))
                ratio = (x - a0) / span
                return float(np.clip(d0 + (d1 - d0) * ratio, 0.0, 100.0))
        return float(np.clip(points[-1][0], 0.0, 100.0))

    def _tracking_state_from_joint_angle(self, channel: int, joint_real_angle: float) -> dict[str, object]:
        filtered_angle = float(self.tracking_angle_filters[channel].update(joint_real_angle))
        lo, hi = self._joint_real_angle_range(channel)
        clamped = float(np.clip(filtered_angle, min(lo, hi), max(lo, hi)))
        flex = int(round(self._map_real_angle_to_flex(channel, clamped)))
        flex = max(0, min(100, flex))
        cfg = self.calibration.get(channel, self._normalize_channel_calibration(channel, {}))
        servo_cfg = cfg.get("servo", {}) if isinstance(cfg, dict) else {}
        min_angle = int(servo_cfg.get("min", 0))
        max_angle = int(servo_cfg.get("max", 180))
        mapped_angle = self._map_flex_to_angle(channel, flex)
        return {
            "channel": channel,
            "flex": flex,
            "mapped_angle": mapped_angle,
            "joint_real": clamped,
            "joint_init": self._joint_initial_angle(channel),
            "joint_rot": self._joint_rotation_angle(channel, clamped),
            "min": min_angle,
            "max": max_angle,
            "direction": "反向" if min_angle > max_angle else "正向",
            "command": f"S,{channel},{mapped_angle}",
        }

    def _solve_tracking_channel_states(self, world_landmarks: np.ndarray) -> dict[int, dict[str, object]]:
        states: dict[int, dict[str, object]] = {}
        pair_distances = self._build_thumb_pair_distances(world_landmarks)
        pinch_alphas: dict[str, float] = {
            finger: self._alpha_from_distance(finger, pair_distances.get(finger, 0.14))
            for finger in self.PINCH_FINGERS
        }

        fingers_cfg = self.hand_model_config.get("fingers", {})
        if not isinstance(fingers_cfg, dict):
            return states

        thumb_cfg_any = fingers_cfg.get("thumb", {})
        thumb_channels_any = thumb_cfg_any.get("channels", {}) if isinstance(thumb_cfg_any, dict) else {}
        thumb_lateral_ref_ch = int(thumb_channels_any.get("lateral", 2)) if isinstance(thumb_channels_any, dict) else 2

        for finger in self.PINCH_FINGERS:
            finger_cfg = fingers_cfg.get(finger, {})
            channels = finger_cfg.get("channels", {}) if isinstance(finger_cfg, dict) else {}
            if not isinstance(channels, dict):
                continue
            alpha = pinch_alphas.get(finger, 0.0)

            proximal_ch = int(channels.get("proximal", -1))
            distal_ch = int(channels.get("distal", -1))
            lateral_ch = int(channels.get("lateral", -1))

            proximal_lim = self._joint_limit_from_calibration(proximal_ch, (0.0, 120.0))
            distal_lim = self._joint_limit_from_calibration(distal_ch, (0.0, 120.0))
            prox_target = float(np.clip(proximal_lim[0] + (proximal_lim[1] - proximal_lim[0]) * alpha, proximal_lim[0], proximal_lim[1]))
            dist_target = float(np.clip(distal_lim[0] + (distal_lim[1] - distal_lim[0]) * alpha, distal_lim[0], distal_lim[1]))

            if proximal_ch in self.ACTIVE_CHANNELS:
                states[proximal_ch] = self._tracking_state_from_joint_angle(proximal_ch, prox_target)
            if distal_ch in self.ACTIVE_CHANNELS:
                states[distal_ch] = self._tracking_state_from_joint_angle(distal_ch, dist_target)

            lateral_lim = self._joint_limit_from_calibration(lateral_ch, (-20.0, 20.0))
            thumb_lat_lim = self._joint_limit_from_calibration(thumb_lateral_ref_ch, (-20.0, 20.0))
            finger_lat_target, _thumb_candidate = self._interpolate_lateral_from_calibration(
                finger,
                alpha,
                lateral_lim,
                thumb_lat_lim,
            )
            if lateral_ch in self.ACTIVE_CHANNELS:
                states[lateral_ch] = self._tracking_state_from_joint_angle(lateral_ch, finger_lat_target)

        thumb_cfg = fingers_cfg.get("thumb", {})
        thumb_channels = thumb_cfg.get("channels", {}) if isinstance(thumb_cfg, dict) else {}
        if isinstance(thumb_channels, dict):
            thumb_alpha = max(pinch_alphas.values()) if pinch_alphas else 0.0
            thumb_prox_ch = int(thumb_channels.get("proximal", -1))
            thumb_dist_ch = int(thumb_channels.get("distal", -1))
            thumb_lat_ch = int(thumb_channels.get("lateral", -1))

            thumb_prox_lim = self._joint_limit_from_calibration(thumb_prox_ch, (0.0, 120.0))
            thumb_dist_lim = self._joint_limit_from_calibration(thumb_dist_ch, (0.0, 120.0))
            thumb_lat_lim = self._joint_limit_from_calibration(thumb_lat_ch, (-20.0, 20.0))
            thumb_prox_target = float(np.clip(thumb_prox_lim[0] + (thumb_prox_lim[1] - thumb_prox_lim[0]) * thumb_alpha, thumb_prox_lim[0], thumb_prox_lim[1]))
            thumb_dist_target = float(np.clip(thumb_dist_lim[0] + (thumb_dist_lim[1] - thumb_dist_lim[0]) * thumb_alpha, thumb_dist_lim[0], thumb_dist_lim[1]))

            dominant_finger = max(pinch_alphas, key=pinch_alphas.get) if pinch_alphas else "index"
            dominant_alpha = pinch_alphas.get(dominant_finger, 0.0)
            dominant_cfg = fingers_cfg.get(dominant_finger, {})
            dominant_channels = dominant_cfg.get("channels", {}) if isinstance(dominant_cfg, dict) else {}
            dominant_lat_ch = int(dominant_channels.get("lateral", -1)) if isinstance(dominant_channels, dict) else -1
            dominant_lateral_lim = self._joint_limit_from_calibration(
                dominant_lat_ch,
                (-20.0, 20.0),
            )
            _dom_finger_lat, thumb_lat_target = self._interpolate_lateral_from_calibration(
                dominant_finger,
                dominant_alpha,
                dominant_lateral_lim,
                thumb_lat_lim,
            )

            if thumb_prox_ch in self.ACTIVE_CHANNELS:
                states[thumb_prox_ch] = self._tracking_state_from_joint_angle(thumb_prox_ch, thumb_prox_target)
            if thumb_dist_ch in self.ACTIVE_CHANNELS:
                states[thumb_dist_ch] = self._tracking_state_from_joint_angle(thumb_dist_ch, thumb_dist_target)
            if thumb_lat_ch in self.ACTIVE_CHANNELS:
                states[thumb_lat_ch] = self._tracking_state_from_joint_angle(thumb_lat_ch, thumb_lat_target)

        return states

    def _draw_hand_overlay(self, frame_bgr: np.ndarray, image_landmarks: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        pts2d = image_landmarks.copy()
        pts2d[:, 0] *= w
        pts2d[:, 1] *= h

        for a, b in self.HAND_CONNECTIONS:
            pa = tuple(np.round(pts2d[a, :2]).astype(int))
            pb = tuple(np.round(pts2d[b, :2]).astype(int))
            cv2.line(frame_bgr, pa, pb, (0, 220, 255), 2, cv2.LINE_AA)
        for idx in range(21):
            p = tuple(np.round(pts2d[idx, :2]).astype(int))
            cv2.circle(frame_bgr, p, 4, (60, 255, 60), -1, cv2.LINE_AA)
        return frame_bgr

    def _default_joint_range(self, channel: int) -> tuple[float, float]:
        if channel in self.LATERAL_CHANNELS:
            return (-15.0, 15.0)
        return (10.0, 90.0)

    def _generate_linear_mapping_points(
        self,
        servo_min: float,
        servo_max: float,
        joint_min: float,
        joint_max: float,
    ) -> list[dict[str, float]]:
        degrees = (0.0, 100.0)
        points: list[dict[str, float]] = []
        for degree in degrees:
            ratio = degree / 100.0
            points.append(
                {
                    "degree": degree,
                    "servo_angle": servo_min + (servo_max - servo_min) * ratio,
                    "joint_angle": joint_min + (joint_max - joint_min) * ratio,
                }
            )
        return points

    def _normalize_channel_calibration(self, channel: int, item: object) -> dict[str, object]:
        default_joint_min, default_joint_max = self._default_joint_range(channel)
        servo_min = 0.0
        servo_max = 180.0
        joint_min = default_joint_min
        joint_max = default_joint_max
        mapping_points: list[dict[str, float]] = []

        if isinstance(item, dict):
            if isinstance(item.get("servo"), dict):
                servo_min = float(item["servo"].get("min", servo_min))
                servo_max = float(item["servo"].get("max", servo_max))
            else:
                servo_min = float(item.get("min", servo_min))
                servo_max = float(item.get("max", servo_max))

            if isinstance(item.get("joint"), dict):
                joint_min = float(item["joint"].get("min", joint_min))
                joint_max = float(item["joint"].get("max", joint_max))

            raw_points = item.get("mapping")
            if isinstance(raw_points, list):
                for point in raw_points:
                    if not isinstance(point, dict):
                        continue
                    try:
                        degree = max(0.0, min(100.0, float(point.get("degree", 0.0))))
                        servo_angle = float(point.get("servo_angle"))
                        joint_angle = float(point.get("joint_angle"))
                    except (TypeError, ValueError):
                        continue
                    mapping_points.append(
                        {
                            "degree": degree,
                            "servo_angle": servo_angle,
                            "joint_angle": joint_angle,
                        }
                    )

        servo_min = max(self.SERVO_ANGLE_MIN, min(self.SERVO_ANGLE_MAX, servo_min))
        servo_max = max(self.SERVO_ANGLE_MIN, min(self.SERVO_ANGLE_MAX, servo_max))
        joint_min = max(-180.0, min(180.0, joint_min))
        joint_max = max(-180.0, min(180.0, joint_max))

        if not mapping_points:
            mapping_points = self._generate_linear_mapping_points(servo_min, servo_max, joint_min, joint_max)

        mapping_points.sort(key=lambda p: float(p["degree"]))
        return {
            "servo": {"min": int(round(servo_min)), "max": int(round(servo_max))},
            "joint": {"min": joint_min, "max": joint_max},
            "mapping": mapping_points,
        }

    def _load_calibration(self) -> dict[int, dict[str, object]]:
        default = {
            idx: self._normalize_channel_calibration(idx, {"min": 0, "max": 180}) for idx in self.ACTIVE_CHANNELS
        }
        if not self.calibration_file.exists():
            return default

        try:
            with self.calibration_file.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return default

        if not isinstance(raw, dict):
            return default

        for ch in self.ACTIVE_CHANNELS:
            default[ch] = self._normalize_channel_calibration(ch, raw.get(str(ch), {}))

        return default

    def _save_calibration(self) -> None:
        payload = {str(ch): self.calibration[ch] for ch in self.ACTIVE_CHANNELS}
        try:
            with self.calibration_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def _get_calibration_channel(self) -> int | None:
        try:
            channel = int(self.calib_channel_var.get())
        except ValueError:
            messagebox.showwarning("输入错误", "舵机编号不在可用通道中")
            return None

        if channel not in self.ACTIVE_CHANNELS:
            messagebox.showwarning("输入错误", "舵机编号不在可用通道中")
            return None
        return channel

    def _load_selected_calibration(self, *_args) -> None:
        channel = self._get_calibration_channel()
        if channel is None:
            return
        cfg = self.calibration[channel]
        self.calib_min_var.set(int(cfg["servo"]["min"]))
        self.calib_max_var.set(int(cfg["servo"]["max"]))
        self._update_realtime_info()

    def _apply_calibration(self) -> None:
        channel = self._get_calibration_channel()
        if channel is None:
            return

        min_angle = self.calib_min_var.get()
        max_angle = self.calib_max_var.get()
        if (
            min_angle < int(self.SERVO_ANGLE_MIN)
            or min_angle > int(self.SERVO_ANGLE_MAX)
            or max_angle < int(self.SERVO_ANGLE_MIN)
            or max_angle > int(self.SERVO_ANGLE_MAX)
        ):
            messagebox.showwarning(
                "输入错误",
                f"最小/最大角度必须在 {int(self.SERVO_ANGLE_MIN)}-{int(self.SERVO_ANGLE_MAX)} 之间",
            )
            return

        joint_cfg = self.calibration[channel].get("joint", {})
        joint_min = float(joint_cfg.get("min", self._default_joint_range(channel)[0]))
        joint_max = float(joint_cfg.get("max", self._default_joint_range(channel)[1]))
        self.calibration[channel] = {
            "servo": {"min": min_angle, "max": max_angle},
            "joint": {"min": joint_min, "max": joint_max},
            "mapping": self._generate_linear_mapping_points(min_angle, max_angle, joint_min, joint_max),
        }
        self._save_calibration()
        self.status_var.set(f"已保存标定: 舵机{channel} [{min_angle}, {max_angle}]")
        self._update_realtime_info()
        self.send_command()

    def _interpolate_mapping_value(self, channel: int, flex_percent: float, value_key: str) -> float:
        cfg = self.calibration.get(channel, self._normalize_channel_calibration(channel, {}))
        points = cfg.get("mapping", []) if isinstance(cfg, dict) else []
        parsed_points: list[tuple[float, float]] = []

        if isinstance(points, list):
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    degree = max(0.0, min(100.0, float(point.get("degree", 0.0))))
                    value = float(point.get(value_key))
                except (TypeError, ValueError):
                    continue
                parsed_points.append((degree, value))

        if not parsed_points:
            servo_cfg = cfg.get("servo", {}) if isinstance(cfg, dict) else {}
            joint_cfg = cfg.get("joint", {}) if isinstance(cfg, dict) else {}
            if value_key == "servo_angle":
                min_val = float(servo_cfg.get("min", 0.0))
                max_val = float(servo_cfg.get("max", 180.0))
            else:
                default_joint_min, default_joint_max = self._default_joint_range(channel)
                min_val = float(joint_cfg.get("min", default_joint_min))
                max_val = float(joint_cfg.get("max", default_joint_max))
            flex = max(0.0, min(100.0, float(flex_percent)))
            return min_val + (max_val - min_val) * (flex / 100.0)

        parsed_points.sort(key=lambda item: item[0])
        flex = max(0.0, min(100.0, float(flex_percent)))

        if flex <= parsed_points[0][0]:
            return parsed_points[0][1]
        if flex >= parsed_points[-1][0]:
            return parsed_points[-1][1]

        for idx in range(1, len(parsed_points)):
            left_degree, left_value = parsed_points[idx - 1]
            right_degree, right_value = parsed_points[idx]
            if flex <= right_degree:
                span = right_degree - left_degree
                if span <= 1e-9:
                    return right_value
                ratio = (flex - left_degree) / span
                return left_value + (right_value - left_value) * ratio

        return parsed_points[-1][1]

    def _map_flex_to_angle(self, channel: int, flex_percent: int) -> int:
        # Serial command angle must be strictly linear from servo min/max calibration.
        flex = max(0.0, min(100.0, float(flex_percent)))
        cfg = self.calibration.get(channel, self._normalize_channel_calibration(channel, {}))
        servo_cfg = cfg.get("servo", {}) if isinstance(cfg, dict) else {}
        min_angle = float(servo_cfg.get("min", 0.0))
        max_angle = float(servo_cfg.get("max", 180.0))
        mapped = min_angle + (max_angle - min_angle) * (flex / 100.0)
        mapped = max(self.SERVO_ANGLE_MIN, min(self.SERVO_ANGLE_MAX, mapped))
        return int(round(mapped))

    def _selected_channels(self) -> list[int]:
        return [ch for ch in self.ACTIVE_CHANNELS if self.channel_vars[ch].get()]

    def _channel_label(self, channel: int) -> str:
        meta = self.CHANNEL_META.get(channel)
        return meta.label if meta else ""

    def _joint_type(self, channel: int) -> str:
        if channel in self.LATERAL_CHANNELS:
            return "lateral"
        meta = self.CHANNEL_META.get(channel)
        return meta.joint_type if meta else "lateral"

    def _calculate_channel_state(self, channel: int, flex_percent: int) -> dict[str, object]:
        flex = max(0, min(100, int(flex_percent)))
        cfg = self.calibration.get(channel, self._normalize_channel_calibration(channel, {}))
        servo_cfg = cfg.get("servo", {})
        min_angle = int(servo_cfg.get("min", 0))
        max_angle = int(servo_cfg.get("max", 180))
        mapped_angle = self._map_flex_to_angle(channel, flex)
        joint_real = self._map_flex_to_real_angle(channel, flex)
        joint_init = self._joint_initial_angle(channel)
        joint_rot = self._joint_rotation_angle(channel, joint_real)
        return {
            "channel": channel,
            "flex": flex,
            "mapped_angle": mapped_angle,
            "joint_real": joint_real,
            "joint_init": joint_init,
            "joint_rot": joint_rot,
            "min": min_angle,
            "max": max_angle,
            "direction": "反向" if min_angle > max_angle else "正向",
            "command": f"S,{channel},{mapped_angle}",
        }

    def _joint_real_angle_range(self, channel: int) -> tuple[float, float]:
        cfg = self.calibration.get(channel, self._normalize_channel_calibration(channel, {}))
        joint_cfg = cfg.get("joint", {}) if isinstance(cfg, dict) else {}
        default_min, default_max = self._default_joint_range(channel)
        min_real = float(joint_cfg.get("min", default_min))
        max_real = float(joint_cfg.get("max", default_max))
        return (min_real, max_real)

    def _map_flex_to_real_angle(self, channel: int, flex_percent: int) -> float:
        return self._interpolate_mapping_value(channel, float(flex_percent), "joint_angle")

    def _joint_initial_angle(self, channel: int) -> float:
        # Baseline rule: lateral starts at flex=50; others start at flex=0.
        baseline_flex = 50 if self._joint_type(channel) == "lateral" else 0
        return self._map_flex_to_real_angle(channel, baseline_flex)

    def _joint_rotation_angle(self, channel: int, joint_real_angle: float) -> float:
        return joint_real_angle

    def _joint_model_bend_angle(self, channel: int, joint_real_angle: float) -> float:
        return self._joint_rotation_angle(channel, joint_real_angle)

    def _format_real_angle_text(self, channel: int, angle: float) -> str:
        if self._joint_type(channel) == "lateral":
            return f"关节角度: {angle:+.1f}°"
        return f"关节角度: {angle:.1f}°"

    def _refresh_real_angle_labels(self, schedule_model_redraw: bool = True) -> None:
        for ch in self.ACTIVE_CHANNELS:
            angle = self.channel_real_angles[ch]
            self.real_angle_vars[ch].set(self._format_real_angle_text(ch, angle))
        if schedule_model_redraw:
            self._schedule_model_redraw()

    def _update_real_angles_from_flex(self, flex_percent: int, channels: list[int] | None = None) -> None:
        targets = channels if channels is not None else list(self.ACTIVE_CHANNELS)
        for ch in targets:
            self.channel_real_angles[ch] = self._map_flex_to_real_angle(ch, flex_percent)

    def _schedule_model_redraw(self) -> None:
        if self.model_redraw_after_id is not None:
            return
        self.model_redraw_after_id = self.root.after(self.model_redraw_ms, self._flush_model_redraw)

    def _schedule_table_refresh(self) -> None:
        # In tracking mode, frames arrive faster than table_refresh_ms.
        # If we always cancel-and-reschedule, refresh can starve forever.
        if self.track_enabled:
            if self.table_after_id is None:
                self.table_after_id = self.root.after(self.table_refresh_ms, self._flush_table_refresh)
            return

        if self.table_after_id is not None:
            self.root.after_cancel(self.table_after_id)
        self.table_after_id = self.root.after(self.table_refresh_ms, self._flush_table_refresh)

    def _flush_table_refresh(self) -> None:
        self.table_after_id = None
        self._refresh_control_table()

    def _flush_model_redraw(self) -> None:
        self.model_redraw_after_id = None
        self._update_3d_model_view()

    def _vector_norm(self, vec: tuple[float, float, float]) -> tuple[float, float, float]:
        length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
        if length < 1e-9:
            return (0.0, 1.0, 0.0)
        return (vec[0] / length, vec[1] / length, vec[2] / length)

    def _rotate_around_x(self, vec: tuple[float, float, float], angle_deg: float) -> tuple[float, float, float]:
        rad = math.radians(angle_deg)
        c = math.cos(rad)
        s = math.sin(rad)
        return (vec[0], vec[1] * c - vec[2] * s, vec[1] * s + vec[2] * c)

    def _cross(self, a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _rotate_around_axis(
        self,
        vec: tuple[float, float, float],
        axis: tuple[float, float, float],
        angle_deg: float,
    ) -> tuple[float, float, float]:
        n = self._vector_norm(axis)
        rad = math.radians(angle_deg)
        c = math.cos(rad)
        s = math.sin(rad)
        dot = vec[0] * n[0] + vec[1] * n[1] + vec[2] * n[2]
        cross = self._cross(n, vec)
        return (
            vec[0] * c + cross[0] * s + n[0] * dot * (1.0 - c),
            vec[1] * c + cross[1] * s + n[1] * dot * (1.0 - c),
            vec[2] * c + cross[2] * s + n[2] * dot * (1.0 - c),
        )

    def _bend_toward_pos_x(self, direction: tuple[float, float, float], bend_deg: float) -> tuple[float, float, float]:
        target = (1.0, 0.0, 0.0)
        axis = self._cross(direction, target)
        axis_len = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        if axis_len < 1e-9:
            return self._vector_norm(direction)
        return self._vector_norm(self._rotate_around_axis(direction, axis, bend_deg))

    def _parse_vec3(self, value: object, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            return fallback
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return fallback

    def _parse_lengths(self, value: object, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            return fallback
        try:
            return (
                max(1.0, float(value[0])),
                max(1.0, float(value[1])),
                max(1.0, float(value[2])),
            )
        except Exception:
            return fallback

    def _finger_points(self, finger_cfg: dict[str, object]) -> list[tuple[float, float, float]]:
        root = self._parse_vec3(finger_cfg.get("root"), (0.0, 0.0, 0.0))
        base_direction = self._vector_norm(self._parse_vec3(finger_cfg.get("base_direction"), (0.0, 1.0, 0.0)))
        lengths = self._parse_lengths(finger_cfg.get("segment_lengths"), (30.0, 24.0, 18.0))
        channels = finger_cfg.get("channels", {})
        if not isinstance(channels, dict):
            channels = {}

        distal_ch = int(channels.get("distal", -1))
        proximal_ch = int(channels.get("proximal", -1))
        lateral_ch = int(channels.get("lateral", -1))

        lateral_initial = self._joint_initial_angle(lateral_ch) if lateral_ch in self.ACTIVE_CHANNELS else 0.0
        proximal_initial = self._joint_initial_angle(proximal_ch) if proximal_ch in self.ACTIVE_CHANNELS else 0.0
        distal_initial = self._joint_initial_angle(distal_ch) if distal_ch in self.ACTIVE_CHANNELS else 0.0

        lateral_real = float(self.channel_real_angles.get(lateral_ch, lateral_initial))
        proximal_real = float(self.channel_real_angles.get(proximal_ch, proximal_initial))
        distal_real = float(self.channel_real_angles.get(distal_ch, distal_initial))

        lateral_deg = self._joint_model_bend_angle(lateral_ch, lateral_real)
        proximal_bend_deg = self._joint_model_bend_angle(proximal_ch, proximal_real)
        distal_bend_deg = self._joint_model_bend_angle(distal_ch, distal_real)

        # MCP has only lateral motion: rotate the whole finger base around +X.
        seg1_dir = self._vector_norm(self._rotate_around_x(base_direction, lateral_deg))

        # Use a stable flexion axis so 100% pose does not collapse into a singular direction.
        flex_axis = self._cross(seg1_dir, (1.0, 0.0, 0.0))
        axis_len = math.sqrt(flex_axis[0] * flex_axis[0] + flex_axis[1] * flex_axis[1] + flex_axis[2] * flex_axis[2])
        if axis_len < 1e-9:
            flex_axis = self._cross(seg1_dir, (0.0, 0.0, 1.0))
            axis_len = math.sqrt(flex_axis[0] * flex_axis[0] + flex_axis[1] * flex_axis[1] + flex_axis[2] * flex_axis[2])
        if axis_len < 1e-9:
            flex_axis = (0.0, 1.0, 0.0)

        seg2_dir = self._vector_norm(self._rotate_around_axis(seg1_dir, flex_axis, proximal_bend_deg))
        seg3_dir = self._vector_norm(self._rotate_around_axis(seg2_dir, flex_axis, distal_bend_deg))

        p0 = root
        p1 = (p0[0] + seg1_dir[0] * lengths[0], p0[1] + seg1_dir[1] * lengths[0], p0[2] + seg1_dir[2] * lengths[0])
        p2 = (p1[0] + seg2_dir[0] * lengths[1], p1[1] + seg2_dir[1] * lengths[1], p1[2] + seg2_dir[2] * lengths[1])
        p3 = (p2[0] + seg3_dir[0] * lengths[2], p2[1] + seg3_dir[1] * lengths[2], p2[2] + seg3_dir[2] * lengths[2])
        return [p0, p1, p2, p3]

    def _build_3d_model_panel(self, parent: ttk.Frame) -> None:
        model_frame = ttk.LabelFrame(
            parent,
            text="三维模型显示区块",
            padding=10,
            width=self.model_panel_size,
            height=self.model_panel_size,
        )
        model_frame.pack(anchor="n", pady=(0, 0))
        model_frame.pack_propagate(False)

        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(model_frame, text="未安装 matplotlib，无法显示三维模型\n请执行: pip install matplotlib").pack(
                padx=12, pady=12, anchor="w"
            )
            return

        self.model_figure = Figure(figsize=(8.0, 8.0), dpi=100)
        grid = self.model_figure.add_gridspec(2, 2, wspace=0.18, hspace=0.2)
        self.model_axis = self.model_figure.add_subplot(grid[0, 0], projection="3d")
        self.model_axes_2d = {
            "xy": self.model_figure.add_subplot(grid[0, 1]),
            "xz": self.model_figure.add_subplot(grid[1, 0]),
            "yz": self.model_figure.add_subplot(grid[1, 1]),
        }
        self.model_canvas = FigureCanvasTkAgg(self.model_figure, master=model_frame)
        self.model_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._schedule_model_redraw()

    def _build_serial_output_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="串口发送区块",
            padding=8,
            width=self.middle_zone_width,
            height=self.serial_panel_height,
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(self.block_v_gap, 0))
        frame.pack_propagate(False)

        text = tk.Text(frame, wrap="none", state=tk.DISABLED, height=8)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scroll.set)
        self.serial_log_text = text

    def _append_serial_log(self, line: str) -> None:
        if self.serial_log_text is None:
            return
        text = self.serial_log_text
        text.configure(state=tk.NORMAL)
        text.insert(tk.END, line + "\n")
        try:
            total = int(text.index("end-1c").split(".")[0])
            if total > self.serial_log_max_lines:
                drop = total - self.serial_log_max_lines
                text.delete("1.0", f"{drop + 1}.0")
        except Exception:
            pass
        text.see(tk.END)
        text.configure(state=tk.DISABLED)

    def _set_scatter_3d_points(self, scatter_artist: object, points: list[tuple[float, float, float]]) -> None:
        if scatter_artist is None:
            return
        px = [p[0] for p in points]
        py = [p[1] for p in points]
        pz = [p[2] for p in points]
        scatter_artist._offsets3d = (px, py, pz)

    def _set_scatter_2d_points(self, scatter_artist: object, points: list[tuple[float, float]]) -> None:
        if scatter_artist is None:
            return
        scatter_artist.set_offsets(points)

    def _ensure_palm_polygon_count(self, axis: object, artists: list[object], target_count: int, color: str, alpha: float) -> None:
        while len(artists) < target_count:
            polygon = axis.fill([], [], color=color, alpha=alpha)[0]
            artists.append(polygon)
        while len(artists) > target_count:
            polygon = artists.pop()
            polygon.remove()

    def _update_palm_surface(
        self,
        root_points: list[tuple[float, float, float]],
    ) -> None:
        if len(root_points) < 2:
            if self.model_palm_artist_3d is not None and hasattr(self.model_palm_artist_3d, "set_verts"):
                self.model_palm_artist_3d.set_verts([])
            self._ensure_palm_polygon_count(self.model_axes_2d.get("xy"), self.model_palm_artists_xy, 0, "#e8c39e", 0.18)
            self._ensure_palm_polygon_count(self.model_axes_2d.get("xz"), self.model_palm_artists_xz, 0, "#e8c39e", 0.18)
            self._ensure_palm_polygon_count(self.model_axes_2d.get("yz"), self.model_palm_artists_yz, 0, "#e8c39e", 0.18)
            return

        origin = self.PALM_ORIGIN
        palm_faces = [[origin, root_points[idx], root_points[idx + 1]] for idx in range(len(root_points) - 1)]
        if not palm_faces:
            return

        palm_color = "#e8c39e"
        palm_edge = "#b7865c"

        if self.model_palm_artist_3d is not None and hasattr(self.model_palm_artist_3d, "set_verts"):
            self.model_palm_artist_3d.set_verts(palm_faces)

        ax_xy = self.model_axes_2d.get("xy")
        ax_xz = self.model_axes_2d.get("xz")
        ax_yz = self.model_axes_2d.get("yz")

        if ax_xy is not None:
            self._ensure_palm_polygon_count(ax_xy, self.model_palm_artists_xy, len(palm_faces), palm_color, 0.18)
            for idx, face in enumerate(palm_faces):
                self.model_palm_artists_xy[idx].set_xy([(p[0], p[1]) for p in face])
        if ax_xz is not None:
            self._ensure_palm_polygon_count(ax_xz, self.model_palm_artists_xz, len(palm_faces), palm_color, 0.18)
            for idx, face in enumerate(palm_faces):
                self.model_palm_artists_xz[idx].set_xy([(p[0], p[2]) for p in face])
        if ax_yz is not None:
            self._ensure_palm_polygon_count(ax_yz, self.model_palm_artists_yz, len(palm_faces), palm_color, 0.18)
            for idx, face in enumerate(palm_faces):
                self.model_palm_artists_yz[idx].set_xy([(p[1], p[2]) for p in face])

    def _init_model_artists(self, finger_items: list[tuple[str, dict[str, object]]]) -> None:
        if self.model_axis is None:
            return

        ax = self.model_axis
        ax_xy = self.model_axes_2d.get("xy")
        ax_xz = self.model_axes_2d.get("xz")
        ax_yz = self.model_axes_2d.get("yz")

        ax.clear()
        if ax_xy is not None:
            ax_xy.clear()
        if ax_xz is not None:
            ax_xz.clear()
        if ax_yz is not None:
            ax_yz.clear()

        self.model_finger_artists = []
        palette = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
        for idx, (_finger_key, finger_cfg) in enumerate(finger_items):
            color = palette[idx % len(palette)]
            line3d = ax.plot([], [], [], color=color, linewidth=self.FINGER_LINEWIDTH_3D)[0]
            scatter3d = ax.scatter([], [], [], color=self.JOINT_COLOR, s=self.FINGER_MARKER_SIZE_3D)
            label_text = str(finger_cfg.get("label", "手指"))
            text3d = ax.text(0.0, 0.0, 0.0, label_text, color=color, fontsize=8)

            line_xy = scatter_xy = None
            line_xz = scatter_xz = None
            line_yz = scatter_yz = None
            if ax_xy is not None:
                line_xy = ax_xy.plot([], [], color=color, linewidth=self.FINGER_LINEWIDTH_2D)[0]
                scatter_xy = ax_xy.scatter([], [], color=self.JOINT_COLOR, s=self.FINGER_MARKER_SIZE_2D)
            if ax_xz is not None:
                line_xz = ax_xz.plot([], [], color=color, linewidth=self.FINGER_LINEWIDTH_2D)[0]
                scatter_xz = ax_xz.scatter([], [], color=self.JOINT_COLOR, s=self.FINGER_MARKER_SIZE_2D)
            if ax_yz is not None:
                line_yz = ax_yz.plot([], [], color=color, linewidth=self.FINGER_LINEWIDTH_2D)[0]
                scatter_yz = ax_yz.scatter([], [], color=self.JOINT_COLOR, s=self.FINGER_MARKER_SIZE_2D)

            self.model_finger_artists.append(
                {
                    "line3d": line3d,
                    "scatter3d": scatter3d,
                    "text3d": text3d,
                    "line_xy": line_xy,
                    "scatter_xy": scatter_xy,
                    "line_xz": line_xz,
                    "scatter_xz": scatter_xz,
                    "line_yz": line_yz,
                    "scatter_yz": scatter_yz,
                }
            )

        self.model_finger_order = tuple(key for key, _ in finger_items)
        self.model_palm_artists_xy = []
        self.model_palm_artists_xz = []
        self.model_palm_artists_yz = []
        if Poly3DCollection is not None:
            self.model_palm_artist_3d = Poly3DCollection(
                [],
                facecolors="#e8c39e",
                edgecolors="#b7865c",
                linewidths=0.8,
                alpha=0.45,
            )
            ax.add_collection3d(self.model_palm_artist_3d)

    def _update_3d_model_view(self) -> None:
        if self.model_axis is None or self.model_canvas is None:
            return

        ax = self.model_axis
        ax_xy = self.model_axes_2d.get("xy")
        ax_xz = self.model_axes_2d.get("xz")
        ax_yz = self.model_axes_2d.get("yz")

        fingers = self.hand_model_config.get("fingers", {})
        if not isinstance(fingers, dict):
            fingers = {}

        finger_items = [(key, cfg) for key, cfg in fingers.items() if isinstance(cfg, dict)]
        if self.model_finger_order != tuple(key for key, _ in finger_items):
            self._init_model_artists(finger_items)
        if not self.model_finger_artists:
            return

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        root_points: list[tuple[float, float, float]] = []

        for idx, (_key, finger_cfg) in enumerate(finger_items):
            points = self._finger_points(finger_cfg)
            root_points.append(points[0])
            artists = self.model_finger_artists[idx]

            px = [p[0] for p in points]
            py = [p[1] for p in points]
            pz = [p[2] for p in points]
            xs.extend(px)
            ys.extend(py)
            zs.extend(pz)

            artists["line3d"].set_data_3d(px, py, pz)
            self._set_scatter_3d_points(artists["scatter3d"], points)

            if ax_xy is not None:
                artists["line_xy"].set_data(px, py)
                self._set_scatter_2d_points(artists["scatter_xy"], [(p[0], p[1]) for p in points])
            if ax_xz is not None:
                artists["line_xz"].set_data(px, pz)
                self._set_scatter_2d_points(artists["scatter_xz"], [(p[0], p[2]) for p in points])
            if ax_yz is not None:
                artists["line_yz"].set_data(py, pz)
                self._set_scatter_2d_points(artists["scatter_yz"], [(p[1], p[2]) for p in points])

            artists["text3d"].set_position((points[0][0], points[0][1]))
            artists["text3d"].set_3d_properties(points[0][2] + 2.0, zdir="z")

        self._update_palm_surface(root_points)
        xs.append(self.PALM_ORIGIN[0])
        ys.append(self.PALM_ORIGIN[1])
        zs.append(self.PALM_ORIGIN[2])

        if not xs:
            xs = [0.0]
            ys = [0.0]
            zs = [0.0]

        ax.set_title("三维视图", fontsize=9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)

        x_mid = (min(xs) + max(xs)) / 2.0
        y_mid = (min(ys) + max(ys)) / 2.0
        z_mid = (min(zs) + max(zs)) / 2.0
        radius = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 60.0) / 2.0 + 10.0
        ax.set_xlim(x_mid - radius, x_mid + radius)
        ax.set_ylim(y_mid - radius, y_mid + radius)
        ax.set_zlim(z_mid - radius, z_mid + radius)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.0, 1.0, 1.0))
        # Initial view target: X toward left-lower, Y right, Z up.
        ax.view_init(elev=18, azim=20)

        if ax_xy is not None:
            ax_xy.set_title("俯视图 (X-Y)", fontsize=8)
            ax_xy.set_xlabel("X")
            ax_xy.set_ylabel("Y")
            ax_xy.set_xlim(x_mid - radius, x_mid + radius)
            ax_xy.set_ylim(y_mid - radius, y_mid + radius)
            ax_xy.set_aspect("equal", adjustable="box")
            ax_xy.grid(True, alpha=0.5)

        if ax_xz is not None:
            ax_xz.set_title("侧视图 (X-Z)", fontsize=8)
            ax_xz.set_xlabel("X")
            ax_xz.set_ylabel("Z")
            ax_xz.set_xlim(x_mid - radius, x_mid + radius)
            ax_xz.set_ylim(z_mid - radius, z_mid + radius)
            ax_xz.set_aspect("equal", adjustable="box")
            ax_xz.grid(True, alpha=0.5)

        if ax_yz is not None:
            ax_yz.set_title("正视图 (Y-Z)", fontsize=8)
            ax_yz.set_xlabel("Y")
            ax_yz.set_ylabel("Z")
            ax_yz.set_xlim(y_mid - radius, y_mid + radius)
            ax_yz.set_ylim(z_mid - radius, z_mid + radius)
            ax_yz.set_aspect("equal", adjustable="box")
            ax_yz.grid(True, alpha=0.5)

        self.model_canvas.draw_idle()

    def _draw_separator(self, canvas: tk.Canvas, width: int) -> None:
        canvas.delete("all")
        canvas.create_line(0, 1, max(0, width - 2), 1, dash=(4, 3), fill="#7f8c8d")

    def _update_realtime_summary(self) -> None:
        if self.track_enabled:
            solved = len(self.tracking_channel_states)
            if solved <= 0:
                self.realtime_info_var.set("追踪模式 | 等待手部")
            else:
                self.realtime_info_var.set(f"追踪模式 | 实时逆解 {solved} 路")
            return

        channels = self._selected_channels()
        if not channels:
            self.realtime_info_var.set("请选择至少一个舵机")
            return

        flex_percent = self.flex_var.get()
        self.realtime_info_var.set(f"幅度 {flex_percent}% | 已选 {len(channels)} 路")

    def _refresh_control_table(self) -> None:
        channels = list(self.tracking_channel_states.keys()) if self.track_enabled else self._selected_channels()
        if not channels:
            if hasattr(self, "control_table"):
                for item_id in self.control_table.get_children():
                    self.control_table.delete(item_id)
                self.control_table_items.clear()
            return

        flex_percent = self.flex_var.get()

        if not hasattr(self, "control_table"):
            return

        selected_set = set(channels)
        for ch, item_id in list(self.control_table_items.items()):
            if ch not in selected_set:
                self.control_table.delete(item_id)
                del self.control_table_items[ch]

        for ch in channels:
            state = self.tracking_channel_states.get(ch) if self.track_enabled else self._calculate_channel_state(ch, flex_percent)
            if state is None:
                continue
            values = (
                str(ch),
                self._channel_label(ch),
                f"{int(state['flex'])}%" if self.track_enabled else f"{flex_percent}%",
                f"{state['mapped_angle']}°",
                f"{state['joint_real']:+.1f}°" if self._joint_type(ch) == "lateral" else f"{state['joint_real']:.1f}°",
                f"{state['joint_init']:+.1f}°" if self._joint_type(ch) == "lateral" else f"{state['joint_init']:.1f}°",
                f"{state['joint_rot']:+.1f}°" if self._joint_type(ch) == "lateral" else f"{state['joint_rot']:.1f}°",
                f"[{state['min']}, {state['max']}]",
                str(state["direction"]),
            )
            item_id = self.control_table_items.get(ch)
            if item_id:
                self.control_table.item(item_id, values=values)
            else:
                self.control_table_items[ch] = self.control_table.insert("", "end", values=values)

    def _update_realtime_info(self) -> None:
        self._update_realtime_summary()
        self._refresh_control_table()

    def _refresh_ports(self) -> None:
        if list_ports is None:
            self.port_combo["values"] = []
            return

        ports = [p.device for p in list_ports.comports()]
        self.port_combo["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    def _toggle_connection(self) -> None:
        if self.ser and self.ser.is_open:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        if serial is None:
            messagebox.showerror("缺少依赖", "未安装 pyserial，请先执行: pip install pyserial")
            return

        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("提示", "请先选择串口")
            return

        try:
            baud = int(self.baud_var.get())
            self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.2)
            self.status_var.set(f"已连接: {port}")
            self.connect_btn.configure(text="断开")
        except Exception as exc:
            messagebox.showerror("连接失败", str(exc))

    def _disconnect(self) -> None:
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.last_payload_sent = ""
        self.status_var.set("未连接")
        self.connect_btn.configure(text="连接")

    def _shutdown_tracking(self) -> None:
        if self.video_after_id is not None:
            try:
                self.root.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None
        if self.video_cap is not None:
            try:
                self.video_cap.release()
            except Exception:
                pass
            self.video_cap = None
        if self.hand_backend is not None and hasattr(self.hand_backend, "close"):
            try:
                self.hand_backend.close()
            except Exception:
                pass

    def close(self) -> None:
        self._shutdown_tracking()
        self._disconnect()

    def _on_channel_changed(self) -> None:
        if self.track_enabled:
            return
        self._update_realtime_summary()
        self._schedule_table_refresh()
        self.send_command()

    def _select_all_channels(self) -> None:
        if self.track_enabled:
            return
        for var in self.channel_vars.values():
            var.set(True)
        self._update_realtime_summary()
        self._schedule_table_refresh()
        self.send_command()

    def _clear_all_channels(self) -> None:
        if self.track_enabled:
            return
        for var in self.channel_vars.values():
            var.set(False)
        self._update_realtime_summary()
        self._schedule_table_refresh()
        self.send_command()

    def _on_angle_moved(self, _value: str) -> None:
        if self.track_enabled:
            return
        # Debounce slider changes to avoid flooding serial while dragging.
        flex_percent = self.flex_var.get()
        self._update_real_angles_from_flex(flex_percent, self._selected_channels())
        self._refresh_real_angle_labels(schedule_model_redraw=True)
        self._update_realtime_summary()
        self._schedule_table_refresh()
        if self.send_after_id:
            self.root.after_cancel(self.send_after_id)
        self.send_after_id = self.root.after(45, self.send_command)

    def _on_scale_mousewheel(self, event: tk.Event) -> str:
        if self.track_enabled:
            return "break"
        step = 0
        if hasattr(event, "delta") and event.delta != 0:
            step = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) == 4:
            step = 1
        elif getattr(event, "num", None) == 5:
            step = -1

        if step != 0:
            current = self.flex_var.get()
            new_value = max(0, min(100, current + step))
            if new_value != current:
                self._set_angle_and_send(new_value)
        return "break"

    def _set_angle_and_send(self, flex_percent: int) -> None:
        if self.track_enabled:
            return
        self.flex_var.set(flex_percent)
        self._update_real_angles_from_flex(flex_percent, self._selected_channels())
        self._refresh_real_angle_labels(schedule_model_redraw=True)
        self._update_realtime_summary()
        self._schedule_table_refresh()
        self.send_command()

    def send_command(self) -> None:
        if self.track_enabled:
            return
        self.send_after_id = None
        channels = self._selected_channels()
        flex_percent = max(0, min(100, int(self.flex_var.get())))
        if flex_percent != self.flex_var.get():
            self.flex_var.set(flex_percent)
        self._update_realtime_summary()

        if not channels:
            self.status_var.set("请至少选择一个舵机")
            self.last_payload_sent = ""
            return

        states = [self._calculate_channel_state(channel, flex_percent) for channel in channels]
        for state in states:
            channel = int(state["channel"])
            self.channel_real_angles[channel] = float(state["joint_real"])

        cmd_list = [str(state["command"]) for state in states]
        self._refresh_real_angle_labels(schedule_model_redraw=True)
        payload = "\n".join(cmd_list) + "\n"

        if not (self.ser and self.ser.is_open):
            preview = ", ".join(str(ch) for ch in channels[:5])
            if len(channels) > 5:
                preview += "..."
            self.status_var.set(f"未连接 | 已选{len(channels)}路: [{preview}] | 幅度: {flex_percent}%")
            return

        if payload == self.last_payload_sent:
            return

        try:
            self.ser.write(payload.encode("ascii"))
            self.last_payload_sent = payload
            stamp = time.strftime("%H:%M:%S")
            for cmd in cmd_list:
                self._append_serial_log(f"[{stamp}] {cmd}")
            self.status_var.set(f"已发送 {len(channels)} 路 | 幅度: {flex_percent}%")
        except Exception as exc:
            self.status_var.set("发送失败")
            messagebox.showerror("串口发送错误", str(exc))

    def _send_tracking_commands(self) -> None:
        if not self.track_enabled:
            return
        if not self.tracking_channel_states:
            self.last_payload_sent = ""
            return

        now = time.monotonic()
        if (now - self.tracking_last_send_ts) * 1000.0 < float(self.tracking_send_interval_ms):
            return

        ordered_channels = [ch for ch in self.ACTIVE_CHANNELS if ch in self.tracking_channel_states]
        cmd_list = [str(self.tracking_channel_states[ch]["command"]) for ch in ordered_channels]
        payload = "\n".join(cmd_list) + "\n"

        if not (self.ser and self.ser.is_open):
            self.last_payload_sent = ""
            return

        if payload == self.last_payload_sent:
            return

        try:
            self.ser.write(payload.encode("ascii"))
            self.tracking_last_send_ts = now
            self.last_payload_sent = payload
            if (now - self.tracking_last_log_ts) * 1000.0 >= float(self.tracking_log_interval_ms):
                stamp = time.strftime("%H:%M:%S")
                preview = ", ".join(cmd_list[:3])
                if len(cmd_list) > 3:
                    preview += ", ..."
                self._append_serial_log(f"[{stamp}] TRACK {len(cmd_list)}路 | {preview}")
                self.tracking_last_log_ts = now
            self.status_var.set(f"追踪发送 {len(cmd_list)} 路")
        except Exception:
            self.status_var.set("追踪发送失败")
            # Avoid modal dialogs in frame loop; they can block UI and appear as freezes.
            if (now - self.tracking_last_error_ts) >= 1.0:
                self._append_serial_log("[ERR] 追踪串口发送失败")
                self.tracking_last_error_ts = now

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill=tk.X)

        ttk.Label(top, text="串口:").grid(row=0, column=0, padx=(2, 6))
        self.port_combo = ttk.Combobox(top, width=12, textvariable=self.port_var, state="readonly")
        self.port_combo.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(top, text="刷新", command=self._refresh_ports).grid(row=0, column=2, padx=(0, 12))

        ttk.Label(top, text="波特率:").grid(row=0, column=3, padx=(0, 6))
        ttk.Entry(top, width=10, textvariable=self.baud_var).grid(row=0, column=4, padx=(0, 12))

        self.connect_btn = ttk.Button(top, text="连接", command=self._toggle_connection)
        self.connect_btn.grid(row=0, column=5, padx=(0, 12))
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=6, sticky="w")
        top.grid_columnconfigure(6, weight=1)

        body = ttk.Frame(self.root, padding=self.body_padding)
        body.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(body, width=self.left_zone_width, height=self.zone_height)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH)
        left_panel.pack_propagate(False)

        middle_panel = ttk.Frame(
            body,
            width=self.middle_zone_width,
            height=self.zone_height,
        )
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(12, 0))
        middle_panel.pack_propagate(False)

        right_panel = ttk.Frame(
            body,
            width=self.right_zone_width,
            height=self.zone_height,
        )
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(12, 0))
        right_panel.pack_propagate(False)

        channel_frame = ttk.LabelFrame(
            left_panel,
            text="选择区块",
            padding=10,
            width=self.left_zone_width,
            height=self.channel_block_height,
        )
        channel_frame.pack(fill=tk.X, pady=(0, self.block_v_gap))
        channel_frame.pack_propagate(False)

        for finger_index, (finger_name, channels) in enumerate(self.FINGER_GROUPS):
            base_row = finger_index * 3
            ttk.Label(channel_frame, text=finger_name).grid(
                row=base_row,
                column=0,
                padx=6,
                pady=6,
                sticky="w",
            )

            for joint_index, channel in enumerate(channels):
                btn = ttk.Checkbutton(
                    channel_frame,
                    text=f"{channel} {self._channel_label(channel)}",
                    variable=self.channel_vars[channel],
                    command=self._on_channel_changed,
                )
                btn.grid(row=base_row, column=joint_index + 1, padx=8, pady=6, sticky="w")
                self.manual_controls.append(btn)
                ttk.Label(channel_frame, textvariable=self.real_angle_vars[channel]).grid(
                    row=base_row + 1,
                    column=joint_index + 1,
                    padx=24,
                    pady=(0, 6),
                    sticky="w",
                )

            if finger_index < len(self.FINGER_GROUPS) - 1:
                sep = tk.Canvas(
                    channel_frame,
                    height=2,
                    highlightthickness=0,
                    bd=0,
                    background=self.root.cget("bg"),
                )
                sep.grid(row=base_row + 2, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 2))
                sep.bind("<Configure>", lambda event, canvas=sep: self._draw_separator(canvas, event.width))

        channel_frame.grid_columnconfigure(0, weight=0)
        channel_frame.grid_columnconfigure(1, weight=1)
        channel_frame.grid_columnconfigure(2, weight=1)
        channel_frame.grid_columnconfigure(3, weight=1)

        select_row = ttk.Frame(channel_frame)
        select_row.grid(row=len(self.FINGER_GROUPS) * 3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        select_all_btn = ttk.Button(select_row, text="全选", command=self._select_all_channels)
        select_all_btn.pack(side=tk.LEFT, padx=(0, 6))
        clear_all_btn = ttk.Button(select_row, text="清空", command=self._clear_all_channels)
        clear_all_btn.pack(side=tk.LEFT)
        self.manual_controls.extend([select_all_btn, clear_all_btn])

        calib_frame = ttk.LabelFrame(
            left_panel,
            text="标定区块",
            padding=10,
            width=self.left_zone_width,
            height=self.calib_block_height,
        )
        calib_frame.pack(fill=tk.X, pady=(0, self.block_v_gap))
        calib_frame.pack_propagate(False)

        ttk.Label(calib_frame, text="舵机编号:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        calib_combo = ttk.Combobox(
            calib_frame,
            width=8,
            textvariable=self.calib_channel_var,
            state="readonly",
            values=[str(i) for i in self.ACTIVE_CHANNELS],
        )
        calib_combo.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        calib_combo.bind("<<ComboboxSelected>>", self._load_selected_calibration)

        ttk.Label(calib_frame, text="最小角度:").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Spinbox(
            calib_frame,
            from_=int(self.SERVO_ANGLE_MIN),
            to=int(self.SERVO_ANGLE_MAX),
            width=8,
            textvariable=self.calib_min_var,
        ).grid(
            row=0, column=3, padx=4, pady=4, sticky="w"
        )

        ttk.Label(calib_frame, text="最大角度:").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        ttk.Spinbox(
            calib_frame,
            from_=int(self.SERVO_ANGLE_MIN),
            to=int(self.SERVO_ANGLE_MAX),
            width=8,
            textvariable=self.calib_max_var,
        ).grid(
            row=0, column=5, padx=4, pady=4, sticky="w"
        )

        ttk.Button(calib_frame, text="保存标定", command=self._apply_calibration).grid(
            row=0, column=6, padx=10, pady=4, sticky="w"
        )

        angle_frame = ttk.LabelFrame(
            left_panel,
            text="控制区块",
            padding=10,
            width=self.left_zone_width,
            height=self.control_block_height,
        )
        angle_frame.pack(fill=tk.X, pady=(0, self.block_v_gap))
        angle_frame.pack_propagate(False)

        self.angle_scale = tk.Scale(
            angle_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=self.control_scale_length,
            resolution=1,
            variable=self.flex_var,
            command=self._on_angle_moved,
        )
        self.angle_scale.pack(pady=(8, 12))
        self.angle_scale.bind("<MouseWheel>", self._on_scale_mousewheel)
        self.angle_scale.bind("<Button-4>", self._on_scale_mousewheel)
        self.angle_scale.bind("<Button-5>", self._on_scale_mousewheel)

        ttk.Label(
            angle_frame,
            textvariable=self.realtime_info_var,
            justify=tk.LEFT,
            anchor="w",
            wraplength=860,
        ).pack(fill=tk.X, padx=8, pady=(0, 10))

        self.control_table = ttk.Treeview(
            angle_frame,
            columns=("channel", "joint", "servo_degree", "servo_angle", "joint_angle", "joint_init", "joint_rot", "range", "direction"),
            show="headings",
            height=7,
        )
        self.control_table.heading("channel", text="舵机")
        self.control_table.heading("joint", text="关节")
        self.control_table.heading("servo_degree", text="旋转程度")
        self.control_table.heading("servo_angle", text="舵机角度")
        self.control_table.heading("joint_angle", text="关节角度")
        self.control_table.heading("joint_init", text="关节初始角")
        self.control_table.heading("joint_rot", text="关节旋转角")
        self.control_table.heading("range", text="标定范围")
        self.control_table.heading("direction", text="方向")
        self.control_table.column("channel", width=70, anchor="center")
        self.control_table.column("joint", width=170, anchor="center")
        self.control_table.column("servo_degree", width=85, anchor="center")
        self.control_table.column("servo_angle", width=85, anchor="center")
        self.control_table.column("joint_angle", width=85, anchor="center")
        self.control_table.column("joint_init", width=95, anchor="center")
        self.control_table.column("joint_rot", width=95, anchor="center")
        self.control_table.column("range", width=95, anchor="center")
        self.control_table.column("direction", width=80, anchor="center")
        self.control_table.pack(fill=tk.X, padx=8, pady=(0, 8))

        control_row = ttk.Frame(angle_frame)
        control_row.pack()
        send_btn = ttk.Button(control_row, text="发送当前幅度", command=self.send_command)
        send_btn.pack(side=tk.LEFT, padx=6)
        center_btn = ttk.Button(control_row, text="回中位 (50%)", command=lambda: self._set_angle_and_send(50))
        center_btn.pack(side=tk.LEFT, padx=6)
        self.manual_controls.extend([self.angle_scale, send_btn, center_btn])

        bio_holder = ttk.Frame(left_panel, width=self.left_zone_width, height=self.bio_calib_block_height)
        bio_holder.pack(fill=tk.X)
        bio_holder.pack_propagate(False)
        self._build_bio_calibration_panel(bio_holder, self.left_zone_width, self.bio_calib_block_height)

        model_holder = ttk.Frame(middle_panel, width=self.middle_zone_width, height=self.model_panel_size)
        model_holder.pack(fill=tk.X, anchor="n")
        model_holder.pack_propagate(False)
        self._build_3d_model_panel(model_holder)

        serial_holder = ttk.Frame(middle_panel, width=self.middle_zone_width, height=self.serial_panel_height)
        serial_holder.pack(fill=tk.BOTH, expand=True)
        serial_holder.pack_propagate(False)
        self._build_serial_output_panel(serial_holder)

        video_holder = ttk.Frame(right_panel, width=self.right_zone_width, height=self.zone_height)
        video_holder.pack(fill=tk.BOTH, expand=True)
        video_holder.pack_propagate(False)
        self._build_video_panel(video_holder)

    def _build_video_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="视频流区块",
            padding=8,
            width=self.video_panel_size,
            height=self.video_panel_size,
        )
        frame.pack(anchor="n", pady=(0, 0))
        frame.pack_propagate(False)
        self.video_canvas = tk.Canvas(frame, bg="#101010", highlightthickness=0, bd=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self._show_no_signal()

    def _build_bio_calibration_panel(self, parent: ttk.Frame, panel_width: int, panel_height: int) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="生物标定区块",
            padding=10,
            width=panel_width,
            height=panel_height,
        )
        frame.pack(fill=tk.BOTH, expand=True)
        frame.pack_propagate(False)

        row = 0
        for finger in self.PINCH_FINGERS:
            ttk.Label(frame, text=f"{self.PINCH_FINGER_LABELS[finger]} 标定").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
            row += 1
            ttk.Button(frame, text="最远标定", command=lambda f=finger: self._capture_bio_far(f)).grid(
                row=row,
                column=0,
                sticky="ew",
                padx=(0, 6),
                pady=(0, 6),
            )
            ttk.Button(frame, text="最近标定", command=lambda f=finger: self._capture_bio_near(f)).grid(
                row=row,
                column=1,
                sticky="ew",
                pady=(0, 6),
            )
            row += 1

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 10))
        row += 1
        self.track_btn = ttk.Button(frame, text="手部追踪", command=self._toggle_hand_tracking)
        self.track_btn.grid(row=row, column=0, columnspan=2, sticky="ew")

        row += 1
        self.track_info_var = tk.StringVar(value="状态: 停止")
        ttk.Label(frame, textvariable=self.track_info_var, justify=tk.LEFT, wraplength=max(100, panel_width - 30)).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

    def _capture_bio_far(self, finger: str) -> None:
        if not self.last_thumb_distances:
            self.status_var.set("生物标定失败：未检测到手")
            return
        self.distance_calibration[finger]["far"] = float(self.last_thumb_distances.get(finger, 0.0))
        try:
            self._save_distance_calibration()
            self.status_var.set(f"{self.PINCH_FINGER_LABELS[finger]} 最远标定已保存")
        except Exception as exc:
            self.status_var.set(f"{self.PINCH_FINGER_LABELS[finger]} 最远标定保存失败")
            messagebox.showerror("标定保存失败", str(exc))

    def _capture_bio_near(self, finger: str) -> None:
        if not self.last_thumb_distances:
            self.status_var.set("生物标定失败：未检测到手")
            return
        self.distance_calibration[finger]["near"] = float(self.last_thumb_distances.get(finger, 0.0))
        try:
            self._save_distance_calibration()
            self.status_var.set(f"{self.PINCH_FINGER_LABELS[finger]} 最近标定已保存")
        except Exception as exc:
            self.status_var.set(f"{self.PINCH_FINGER_LABELS[finger]} 最近标定保存失败")
            messagebox.showerror("标定保存失败", str(exc))

    def _toggle_hand_tracking(self) -> None:
        self.track_enabled = not self.track_enabled
        if self.track_enabled:
            self.track_btn.configure(text="停止追踪")
            self.track_info_var.set(f"状态: 运行中 ({self.hand_backend_type})")
            self.tracking_channel_states.clear()
            self._reset_tracking_filters()
            self.tracking_last_send_ts = 0.0
            self.tracking_last_log_ts = 0.0
            self.tracking_last_error_ts = 0.0
            self._set_manual_controls_enabled(False)
        else:
            self.track_btn.configure(text="手部追踪")
            self.track_info_var.set("状态: 停止")
            self.last_thumb_distances = {}
            self.tracking_channel_states.clear()
            self.last_payload_sent = ""
            self._reset_tracking_filters()
            self.tracking_last_send_ts = 0.0
            self.tracking_last_log_ts = 0.0
            self.tracking_last_error_ts = 0.0
            self._set_manual_controls_enabled(True)
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self._show_no_signal()
        self._update_realtime_summary()
        self._schedule_table_refresh()

    def _show_no_signal(self) -> None:
        if self.video_canvas is None:
            return
        self.video_canvas.delete("all")
        w = max(1, self.video_canvas.winfo_width())
        h = max(1, self.video_canvas.winfo_height())
        self.video_canvas.create_rectangle(0, 0, w, h, fill="#101010", outline="")
        self.video_canvas_text_id = self.video_canvas.create_text(
            w // 2,
            h // 2,
            text="NO SIGNAL",
            fill="#c0c0c0",
            font=("Segoe UI", 24, "bold"),
        )
        self.video_canvas_image_id = None
        self.video_image_ref = None

    def _render_video_frame(self, frame_bgr: np.ndarray) -> None:
        if self.video_canvas is None:
            return
        canvas_w = max(160, self.video_canvas.winfo_width())
        canvas_h = max(120, self.video_canvas.winfo_height())

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = frame_rgb.shape[:2]
        scale = min(canvas_w / max(src_w, 1), canvas_h / max(src_h, 1))
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)
        y0 = (canvas_h - new_h) // 2
        x0 = (canvas_w - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

        image = Image.fromarray(canvas)
        photo = ImageTk.PhotoImage(image=image)
        if self.video_canvas_image_id is None:
            self.video_canvas.delete("all")
            self.video_canvas_image_id = self.video_canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo)
        else:
            self.video_canvas.itemconfig(self.video_canvas_image_id, image=photo)
            self.video_canvas.coords(self.video_canvas_image_id, canvas_w // 2, canvas_h // 2)
        self.video_image_ref = photo

    def _schedule_video_frame(self) -> None:
        if self.video_after_id is not None:
            return
        self.video_after_id = self.root.after(self.video_refresh_ms, self._update_video_frame)

    def _set_manual_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in self.manual_controls:
            try:
                widget.configure(state=state)
            except Exception:
                pass

    def _update_video_frame(self) -> None:
        self.video_after_id = None

        if not self.track_enabled:
            self._show_no_signal()
            self._schedule_video_frame()
            return

        if self.hand_backend is None:
            self.track_info_var.set("状态: 缺少MediaPipe后端")
            self._show_no_signal()
            self._schedule_video_frame()
            return

        if not self._open_video_capture():
            if self.camera_open_error:
                self.track_info_var.set(f"状态: 摄像头不可用 ({self.camera_open_error})")
            else:
                self.track_info_var.set("状态: 摄像头不可用")
            self._show_no_signal()
            self._schedule_video_frame()
            return

        ok, frame = self.video_cap.read()
        if not ok:
            self.track_info_var.set("状态: 读取失败")
            try:
                self.video_cap.release()
            except Exception:
                pass
            self.video_cap = None
            self._show_no_signal()
            self._schedule_video_frame()
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_landmarks, world_landmarks = self._extract_hand_landmarks(frame_rgb)
            if image_landmarks is not None and world_landmarks is not None:
                image_landmarks = self._apply_landmark_filters(image_landmarks, self.image_landmark_filters)
                world_landmarks = self._apply_landmark_filters(world_landmarks, self.world_landmark_filters)
                frame = self._draw_hand_overlay(frame, image_landmarks)
                raw_distances = self._build_thumb_pair_distances(world_landmarks)
                self.last_thumb_distances = {
                    finger: self.distance_filters[finger].update(raw_distances.get(finger, 0.0))
                    for finger in self.PINCH_FINGERS
                }
                self.tracking_channel_states = self._solve_tracking_channel_states(world_landmarks)
                for ch, state in self.tracking_channel_states.items():
                    self.channel_real_angles[ch] = float(state["joint_real"])
                self._refresh_real_angle_labels(schedule_model_redraw=True)
                self._schedule_table_refresh()
                self._update_realtime_summary()
                self._send_tracking_commands()
                self.track_info_var.set("状态: 检测到手")
            else:
                self.last_thumb_distances = {}
                self.tracking_channel_states.clear()
                self.last_payload_sent = ""
                self._schedule_table_refresh()
                self._update_realtime_summary()
                self._reset_tracking_filters()
                self.track_info_var.set("状态: 等待手部")
        except Exception as exc:
            self.tracking_channel_states.clear()
            self.last_payload_sent = ""
            self._schedule_table_refresh()
            self._update_realtime_summary()
            self._reset_tracking_filters()
            self.track_info_var.set(f"状态: 追踪异常 {exc}")

        self._render_video_frame(frame)
        self._schedule_video_frame()


def main() -> None:
    root = tk.Tk()
    app = ServoControllerApp(root)

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
