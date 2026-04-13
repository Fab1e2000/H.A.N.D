from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvasTkAgg = None
    Figure = None


CAMERA_INDEX = 0
WINDOW_TITLE = "Hand Pose Viewer"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILE = Path(__file__).resolve().parent / "hand_landmarker.task"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CONFIG_FILE = PROJECT_ROOT / "hand_model_config.json"
SERVO_CALIBRATION_FILE = PROJECT_ROOT / "servo_calibration.json"
DISTANCE_CALIBRATION_FILE = Path(__file__).resolve().parent / "distance_calibration.json"
WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 980
FRAME_INTERVAL_MS = 16
RIGHT_PANEL_WIDTH = 470
BODY_PAD = 12
BODY_GAP = 12
MODEL_PLOT_HEIGHT = 340
JOINT_TABLE_HEIGHT = 210

# 21-point hand landmarks (common ordering)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP = 17

FINGERS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
}

PINCH_FINGERS = ("index", "middle", "ring")
PINCH_FINGER_LABELS = {
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


class Kalman1D:
    def __init__(self, process_var: float = 1.2, measure_var: float = 9.0) -> None:
        self.q = float(process_var)
        self.r = float(measure_var)
        self.x = 0.0
        self.p = 1.0
        self.initialized = False

    def update(self, z: float) -> float:
        measurement = float(z)
        if not self.initialized:
            self.x = measurement
            self.p = 1.0
            self.initialized = True
            return self.x

        # Predict step for constant-state model.
        self.p = self.p + self.q

        # Update step.
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1.0 - k) * self.p
        return self.x

    def reset(self) -> None:
        self.x = 0.0
        self.p = 1.0
        self.initialized = False


def ensure_hand_landmarker_model() -> Path:
    if MODEL_FILE.exists():
        return MODEL_FILE

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as exc:
        raise RuntimeError(
            "当前 MediaPipe 不包含 solutions API，且下载 hand_landmarker.task 失败。"
            "请手动下载该模型并放到 TEST/minimal_hand_project/hand_landmarker.task"
        ) from exc

    return MODEL_FILE


def create_mediapipe_backend() -> tuple[str, object]:
    # Legacy API path.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        return ("solutions", hands)

    # Tasks API path.
    try:
        # Preferred import path for tasks API.
        from mediapipe.tasks import python as mp_tasks
        vision = mp_tasks.vision
        base_options_cls = mp_tasks.BaseOptions
    except Exception:
        try:
            # Fallback for alternative package export structures.
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions as base_options_cls
        except Exception as exc:
            raise RuntimeError("无法初始化 MediaPipe 后端：既没有 solutions，也无法导入 tasks API") from exc

    model_path = ensure_hand_landmarker_model()
    options = vision.HandLandmarkerOptions(
        base_options=base_options_cls(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    return ("tasks", landmarker)


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v)) + 1e-8


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = _safe_norm(v1) * _safe_norm(v2)
    cosv = float(np.dot(v1, v2) / denom)
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def signed_angle_on_plane(v_ref: np.ndarray, v: np.ndarray, plane_normal: np.ndarray) -> float:
    n = plane_normal / _safe_norm(plane_normal)

    v_ref_p = v_ref - np.dot(v_ref, n) * n
    v_p = v - np.dot(v, n) * n

    if _safe_norm(v_ref_p) < 1e-6 or _safe_norm(v_p) < 1e-6:
        return 0.0

    v_ref_p = v_ref_p / _safe_norm(v_ref_p)
    v_p = v_p / _safe_norm(v_p)

    unsigned = angle_between(v_ref_p, v_p)
    sign = float(np.sign(np.dot(n, np.cross(v_ref_p, v_p))))
    return unsigned * sign


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    points = landmarks.copy()
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def compute_palm_axes(pts3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrist = pts3d[WRIST]
    v_index = pts3d[INDEX_MCP] - wrist
    v_pinky = pts3d[PINKY_MCP] - wrist

    palm_normal = np.cross(v_index, v_pinky)
    if _safe_norm(palm_normal) < 1e-6:
        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    palm_forward = pts3d[MIDDLE_MCP] - wrist
    return palm_normal, palm_forward


def compute_finger_angles(pts3d: np.ndarray, mcp: int, pip: int, dip: int) -> tuple[float, float]:
    # Flexion at PIP approximates finger bend.
    v1 = pts3d[mcp] - pts3d[pip]
    v2 = pts3d[dip] - pts3d[pip]
    flexion = angle_between(v1, v2)

    # Lateral swing at MCP as signed in-plane angle against palm forward axis.
    palm_normal, palm_forward = compute_palm_axes(pts3d)
    finger_axis = pts3d[pip] - pts3d[mcp]
    lateral = signed_angle_on_plane(palm_forward, finger_axis, palm_normal)
    return flexion, lateral


def rotate_around_axis(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis_norm = axis / _safe_norm(axis)
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    v = vec.astype(np.float64)
    return (v * c) + (np.cross(axis_norm, v) * s) + (axis_norm * np.dot(axis_norm, v) * (1.0 - c))


def load_hand_model_config() -> dict[str, dict[str, object]]:
    default = {
        "fingers": {
            "thumb": {
                "label": "thumb",
                "root": [0.0, 27.0, 18.0],
                "base_direction": [0.0, 1.0, -1.0],
                "segment_lengths": [17.0, 30.0, 43.0],
            },
            "index": {
                "label": "index",
                "root": [0.0, 25.0, 50.0],
                "base_direction": [0.0, 0.0, 1.0],
                "segment_lengths": [17.0, 30.0, 43.0],
            },
            "middle": {
                "label": "middle",
                "root": [0.0, 8.0, 50.0],
                "base_direction": [0.0, -1.0, 8.0],
                "segment_lengths": [17.0, 30.0, 43.0],
            },
            "ring": {
                "label": "ring",
                "root": [0.0, -8.0, 50.0],
                "base_direction": [0.0, -1.0, 6.0],
                "segment_lengths": [17.0, 30.0, 43.0],
            },
        }
    }

    if not MODEL_CONFIG_FILE.exists():
        return default

    try:
        with MODEL_CONFIG_FILE.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and isinstance(loaded.get("fingers"), dict):
            return loaded
    except Exception:
        pass
    return default


def load_servo_calibration() -> dict[str, object]:
    if not SERVO_CALIBRATION_FILE.exists():
        return {}

    try:
        with SERVO_CALIBRATION_FILE.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return {}


def _default_distance_calibration() -> dict[str, dict[str, float | None]]:
    return {
        finger: {
            "far": None,
            "near": None,
            "lateral_far": None,
            "lateral_near": None,
            "thumb_lateral_far": None,
            "thumb_lateral_near": None,
        }
        for finger in PINCH_FINGERS
    }


def _normalize_distance_calibration(raw: object) -> dict[str, dict[str, float | None]]:
    normalized = _default_distance_calibration()
    if not isinstance(raw, dict):
        return normalized

    for finger in PINCH_FINGERS:
        item = raw.get(finger)
        if not isinstance(item, dict):
            continue
        for key in (
            "far",
            "near",
            "lateral_far",
            "lateral_near",
            "thumb_lateral_far",
            "thumb_lateral_near",
        ):
            val = item.get(key)
            if val is None:
                normalized[finger][key] = None
                continue
            try:
                normalized[finger][key] = float(val)
            except Exception:
                normalized[finger][key] = None
    return normalized


def save_distance_calibration(data: dict[str, dict[str, float | None]]) -> None:
    payload = _normalize_distance_calibration(data)
    with DISTANCE_CALIBRATION_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_distance_calibration() -> dict[str, dict[str, float | None]]:
    if not DISTANCE_CALIBRATION_FILE.exists():
        default = _default_distance_calibration()
        try:
            save_distance_calibration(default)
        except Exception:
            pass
        return default

    try:
        with DISTANCE_CALIBRATION_FILE.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return _default_distance_calibration()

    return _normalize_distance_calibration(loaded)


def _joint_limit_from_calibration(calibration: dict[str, object], channel: int, fallback: tuple[float, float]) -> tuple[float, float]:
    item = calibration.get(str(channel))
    if not isinstance(item, dict):
        return fallback

    joint = item.get("joint")
    if not isinstance(joint, dict):
        return fallback

    try:
        lo = float(joint.get("min"))
        hi = float(joint.get("max"))
    except Exception:
        return fallback

    if lo <= hi:
        return (lo, hi)
    return (hi, lo)


def build_finger_joint_limits(
    hand_model_config: dict[str, dict[str, object]],
    servo_calibration: dict[str, object],
) -> dict[str, dict[str, tuple[float, float]]]:
    limits: dict[str, dict[str, tuple[float, float]]] = {}
    fingers_cfg = hand_model_config.get("fingers", {})

    if not isinstance(fingers_cfg, dict):
        return limits

    for finger_key in FINGERS:
        finger_cfg = fingers_cfg.get(finger_key, {})
        channels = finger_cfg.get("channels", {}) if isinstance(finger_cfg, dict) else {}
        if not isinstance(channels, dict):
            channels = {}

        distal_ch = int(channels.get("distal", -1))
        proximal_ch = int(channels.get("proximal", -1))
        lateral_ch = int(channels.get("lateral", -1))
        mcp_flex_ch = int(channels.get("mcp_flex", proximal_ch))

        finger_limits = {
            "mcp_flex": _joint_limit_from_calibration(servo_calibration, mcp_flex_ch, (0.0, 120.0)),
            "proximal": _joint_limit_from_calibration(servo_calibration, proximal_ch, (0.0, 120.0)),
            "distal": _joint_limit_from_calibration(servo_calibration, distal_ch, (0.0, 120.0)),
            "lateral": _joint_limit_from_calibration(servo_calibration, lateral_ch, (-20.0, 20.0)),
        }
        limits[finger_key] = finger_limits

    return limits


def _joint_angle_at_point(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    return angle_between(v1, v2)


def _to_flexion_degrees(raw_joint_angle: float) -> float:
    # Geometric joint angle is near 180 deg when fully straight;
    # convert to flexion convention: straight=0 deg, bending=positive.
    return float(np.clip(180.0 - float(raw_joint_angle), 0.0, 180.0))


def extract_mcp_pip_flexion_angles(pts3d: np.ndarray, finger_key: str) -> dict[str, float]:
    # Finger chain points ordered from proximal to distal in FINGERS.
    p0, p1, p2, p3 = FINGERS[finger_key]

    # MCP-equivalent flexion is applied at proximal-middle connection: p0-p1-p2.
    mcp_raw = _joint_angle_at_point(pts3d[p0], pts3d[p1], pts3d[p2])
    mcp_flex = _to_flexion_degrees(mcp_raw)

    # PIP-equivalent flexion is applied at middle-distal connection: p1-p2-p3.
    pip_raw = _joint_angle_at_point(pts3d[p1], pts3d[p2], pts3d[p3])
    pip_flex = _to_flexion_degrees(pip_raw)
    return {
        "mcp_flex": float(np.clip(mcp_flex, 0.0, 150.0)),
        "pip_flex": float(np.clip(pip_flex, 0.0, 150.0)),
    }


def reconstruct_finger_points(
    root: np.ndarray,
    base_direction: np.ndarray,
    lengths: tuple[float, float, float],
    mcp_flex_deg: float,
    proximal_deg: float,
    lateral_deg: float,
) -> np.ndarray:
    dir0 = base_direction / _safe_norm(base_direction)

    lateral_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dir_lat = rotate_around_axis(dir0, lateral_axis, lateral_deg)

    flex_axis = np.cross(dir_lat, lateral_axis)
    if _safe_norm(flex_axis) < 1e-6:
        flex_axis = np.cross(dir_lat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if _safe_norm(flex_axis) < 1e-6:
        flex_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Root to proximal segment keeps base direction; flexion starts from proximal-middle joint.
    seg1_dir = dir_lat
    seg2_dir = rotate_around_axis(seg1_dir, flex_axis, mcp_flex_deg)
    seg3_dir = rotate_around_axis(seg2_dir, flex_axis, proximal_deg)

    l1, l2, l3 = lengths
    p0 = root.astype(np.float64)
    p1 = p0 + seg1_dir * l1
    p2 = p1 + seg2_dir * l2
    p3 = p2 + seg3_dir * l3
    return np.vstack((p0, p1, p2, p3))


def build_angle_filters() -> dict[tuple[str, str], Kalman1D]:
    filters: dict[tuple[str, str], Kalman1D] = {}
    for finger_name in FINGERS:
        filters[(finger_name, "flexion")] = Kalman1D(process_var=1.1, measure_var=10.0)
        filters[(finger_name, "lateral")] = Kalman1D(process_var=0.8, measure_var=7.0)
    return filters


def build_landmark_filters() -> dict[tuple[int, int], Kalman1D]:
    filters: dict[tuple[int, int], Kalman1D] = {}
    for landmark_idx in range(21):
        for axis in range(3):
            filters[(landmark_idx, axis)] = Kalman1D(process_var=0.0006, measure_var=0.0025)
    return filters


def apply_landmark_filters(
    landmarks: np.ndarray,
    filters: dict[tuple[int, int], Kalman1D],
) -> np.ndarray:
    filtered = np.empty_like(landmarks)
    for i in range(landmarks.shape[0]):
        for axis in range(landmarks.shape[1]):
            filtered[i, axis] = filters[(i, axis)].update(float(landmarks[i, axis]))
    return filtered


def reset_filter_bank(filters: dict[tuple[int, int] | tuple[str, str], Kalman1D]) -> None:
    for filt in filters.values():
        filt.reset()


def draw_hand_overlay(
    frame: np.ndarray,
    image_landmarks: np.ndarray,
    world_landmarks: np.ndarray,
    angle_filters: dict[tuple[str, str], Kalman1D],
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    h, w = frame.shape[:2]
    pts2d = to_pixels(image_landmarks, w, h)
    angle_rows: dict[str, tuple[float, float]] = {}

    # Draw full hand skeleton (21 landmarks + palm links).
    for a, b in HAND_CONNECTIONS:
        pa = tuple(np.round(pts2d[a, :2]).astype(int))
        pb = tuple(np.round(pts2d[b, :2]).astype(int))
        cv2.line(frame, pa, pb, (0, 220, 255), 2, cv2.LINE_AA)

    for idx in range(21):
        p = tuple(np.round(pts2d[idx, :2]).astype(int))
        cv2.circle(frame, p, 4, (60, 255, 60), -1, cv2.LINE_AA)

    # Angle estimation now includes thumb + index/middle/ring.
    for name, (_base, mcp, pip, _tip) in FINGERS.items():
        flexion_raw, lateral_raw = compute_finger_angles(world_landmarks, mcp, pip, _tip)
        flexion = angle_filters[(name, "flexion")].update(flexion_raw)
        lateral = angle_filters[(name, "lateral")].update(lateral_raw)
        angle_rows[name] = (flexion, lateral)

    return frame, angle_rows

class HandPoseViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(1420, 860)
        self.root.resizable(False, False)

        self.backend_type, self.backend = create_mediapipe_backend()
        self.angle_filters = build_angle_filters()
        self.ik_filters = {
            (finger, "mcp_flex"): Kalman1D(process_var=1.1, measure_var=10.0)
            for finger in FINGERS
        }
        self.ik_filters.update(
            {
                (finger, "proximal"): Kalman1D(process_var=1.1, measure_var=10.0)
                for finger in FINGERS
            }
        )
        self.ik_filters.update(
            {
                (finger, "distal"): Kalman1D(process_var=1.0, measure_var=9.0)
                for finger in FINGERS
            }
        )
        self.ik_filters.update(
            {
                (finger, "lateral"): Kalman1D(process_var=0.9, measure_var=8.0)
                for finger in FINGERS
            }
        )
        self.image_landmark_filters = build_landmark_filters()
        self.world_landmark_filters = build_landmark_filters()
        self.hand_model_config = load_hand_model_config()
        self.servo_calibration = load_servo_calibration()
        self.finger_joint_limits = build_finger_joint_limits(self.hand_model_config, self.servo_calibration)
        self.last_thumb_distances: dict[str, float] = {}
        self.last_thumb_distances_primary: dict[str, float] = {}
        self.last_thumb_distances_dip: dict[str, float] = {}
        self.last_solved_angles: dict[str, tuple[float, float]] = {}
        self.last_joint_dofs: dict[str, tuple[float, float, float]] = {}
        self.distance_calibration = load_distance_calibration()
        self.last_model_points: dict[str, np.ndarray] = {}

        self.cap = None
        if hasattr(cv2, "CAP_DSHOW"):
            cap_try = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if cap_try.isOpened():
                self.cap = cap_try
            else:
                cap_try.release()
        if self.cap is None:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头索引 {CAMERA_INDEX}")

        self.status_var = tk.StringVar(value="Running")
        self.backend_var = tk.StringVar(value=f"Backend: {self.backend_type}")
        self.detect_var = tk.StringVar(value="Hand: not detected")
        self.calibration_status_vars: dict[str, tk.StringVar] = {
            finger: tk.StringVar(value=f"{PINCH_FINGER_LABELS[finger]}: 未标定") for finger in PINCH_FINGERS
        }
        self.video_image_ref: ImageTk.PhotoImage | None = None
        self.video_canvas: tk.Canvas | None = None
        self.video_canvas_image_id: int | None = None
        self.model_figure = None
        self.model_axis = None
        self.model_canvas = None
        self.joint_table: ttk.Treeview | None = None

        self._configure_theme()
        self._build_ui()
        self._refresh_calibration_status_texts()
        self._schedule_next_frame()

    def _configure_theme(self) -> None:
        style = ttk.Style()
        for theme_name in ("vista", "xpnative", "clam"):
            if theme_name in style.theme_names():
                style.theme_use(theme_name)
                break
        style.configure("TLabelFrame", padding=8)
        style.configure("TButton", padding=(10, 4))
        style.configure("TLabel", padding=(1, 1))

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill=tk.X)

        ttk.Label(top, textvariable=self.backend_var).grid(row=0, column=0, padx=(2, 16), sticky="w")
        ttk.Label(top, textvariable=self.detect_var).grid(row=0, column=1, padx=(0, 16), sticky="w")
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=2, sticky="w")
        ttk.Button(top, text="重置滤波", command=self._reset_filters).grid(row=0, column=3, padx=(14, 0))
        top.grid_columnconfigure(4, weight=1)

        body = ttk.Frame(self.root, padding=BODY_PAD)
        body.pack(fill=tk.BOTH, expand=True)
        body.pack_propagate(False)

        left_panel = ttk.LabelFrame(body, text="视频流区块", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(body, width=RIGHT_PANEL_WIDTH)
        right_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(BODY_GAP, 0))
        right_panel.pack_propagate(False)

        available_width = WINDOW_WIDTH - BODY_PAD * 2 - BODY_GAP - RIGHT_PANEL_WIDTH
        available_height = WINDOW_HEIGHT - 90 - BODY_PAD * 2
        left_panel.configure(width=max(100, available_width), height=max(100, available_height))
        right_panel.configure(height=max(100, available_height))

        self.video_canvas = tk.Canvas(left_panel, bg="#181818", highlightthickness=0, bd=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        calib_frame = ttk.LabelFrame(right_panel, text="标定功能区块", padding=10)
        calib_frame.pack(fill=tk.X)

        for row, finger in enumerate(PINCH_FINGERS):
            ttk.Button(
                calib_frame,
                text="最远标定",
                command=lambda f=finger: self._capture_far_calibration_for(f),
            ).grid(row=row, column=0, padx=(0, 6), pady=(4, 2), sticky="w")
            ttk.Button(
                calib_frame,
                text="最近标定",
                command=lambda f=finger: self._capture_near_calibration_for(f),
            ).grid(row=row, column=1, padx=(0, 6), pady=(4, 2), sticky="w")

        joint_frame = ttk.LabelFrame(right_panel, text="关节角度区块", padding=8, height=JOINT_TABLE_HEIGHT)
        joint_frame.pack(fill=tk.X, pady=(10, 0))
        joint_frame.pack_propagate(False)
        self._build_joint_table(joint_frame)

        model_frame = ttk.LabelFrame(right_panel, text="三维模型显示区块", padding=8, height=MODEL_PLOT_HEIGHT)
        model_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        model_frame.pack_propagate(False)
        self._build_model_plot(model_frame)

    def _build_joint_table(self, parent: ttk.LabelFrame) -> None:
        columns = ("finger", "mcp_flex", "proximal", "lateral")
        table = ttk.Treeview(parent, columns=columns, show="headings", height=5)
        table.heading("finger", text="手指")
        table.heading("mcp_flex", text="MCP弯曲(°)")
        table.heading("proximal", text="PIP弯曲(°)")
        table.heading("lateral", text="侧摆(°)")

        table.column("finger", width=86, anchor="center")
        table.column("mcp_flex", width=110, anchor="center")
        table.column("proximal", width=110, anchor="center")
        table.column("lateral", width=90, anchor="center")
        table.pack(fill=tk.BOTH, expand=True)

        display_order = ("thumb", "index", "middle", "ring")
        for finger in display_order:
            table.insert("", tk.END, iid=f"joint-{finger}", values=(finger, "--", "--", "--"))

        self.joint_table = table

    def _update_joint_table(self, dofs: dict[str, tuple[float, float, float]]) -> None:
        if self.joint_table is None:
            return
        display_order = ("thumb", "index", "middle", "ring")
        for finger in display_order:
            row = dofs.get(finger)
            if row is None:
                values = (finger, "--", "--", "--")
            else:
                values = (
                    finger,
                    f"{float(row[0]):.1f}",
                    f"{float(row[1]):.1f}",
                    f"{float(row[2]):+.1f}",
                )
            self.joint_table.item(f"joint-{finger}", values=values)

    def _build_model_plot(self, parent: ttk.LabelFrame) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(parent, text="未安装 matplotlib，无法显示三维模型\n请执行: pip install matplotlib").pack(
                fill=tk.BOTH,
                expand=True,
                padx=8,
                pady=8,
            )
            return

        self.model_figure = Figure(figsize=(4.6, 3.2), dpi=100)
        self.model_axis = self.model_figure.add_subplot(111, projection="3d")
        self.model_axis.set_title("Inverse-Solved Hand Model")
        self.model_canvas = FigureCanvasTkAgg(self.model_figure, master=parent)
        self.model_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _reset_filters(self) -> None:
        reset_filter_bank(self.angle_filters)
        reset_filter_bank(self.ik_filters)
        reset_filter_bank(self.image_landmark_filters)
        reset_filter_bank(self.world_landmark_filters)
        self.status_var.set("Filters reset")

    def _capture_far_calibration_for(self, finger: str) -> None:
        if not self.last_thumb_distances:
            self.status_var.set("最远标定失败：未检测到手")
            return
        self.distance_calibration[finger]["far"] = float(self.last_thumb_distances.get(finger, 0.0))
        lat_pair = self._optimize_lateral_pair_for_finger(finger, maximize_distance=True)
        if lat_pair is None:
            self.status_var.set(f"{PINCH_FINGER_LABELS[finger]} 最远标定失败：缺少模型角度")
            return
        finger_lat, thumb_lat = lat_pair
        self.distance_calibration[finger]["lateral_far"] = float(finger_lat)
        self.distance_calibration[finger]["thumb_lateral_far"] = float(thumb_lat)
        try:
            save_distance_calibration(self.distance_calibration)
        except Exception:
            self.status_var.set(f"{PINCH_FINGER_LABELS[finger]} 最远标定保存失败")
            self._refresh_calibration_status_texts()
            return
        self.status_var.set(f"已记录 {PINCH_FINGER_LABELS[finger]} 最远标定")
        self._refresh_calibration_status_texts()

    def _capture_near_calibration_for(self, finger: str) -> None:
        if not self.last_thumb_distances:
            self.status_var.set("最近标定失败：未检测到手")
            return
        self.distance_calibration[finger]["near"] = float(self.last_thumb_distances.get(finger, 0.0))
        lat_pair = self._optimize_lateral_pair_for_finger(finger, maximize_distance=False)
        if lat_pair is None:
            self.status_var.set(f"{PINCH_FINGER_LABELS[finger]} 最近标定失败：缺少模型角度")
            return
        finger_lat, thumb_lat = lat_pair
        self.distance_calibration[finger]["lateral_near"] = float(finger_lat)
        self.distance_calibration[finger]["thumb_lateral_near"] = float(thumb_lat)
        try:
            save_distance_calibration(self.distance_calibration)
        except Exception:
            self.status_var.set(f"{PINCH_FINGER_LABELS[finger]} 最近标定保存失败")
            self._refresh_calibration_status_texts()
            return
        self.status_var.set(f"已记录 {PINCH_FINGER_LABELS[finger]} 最近标定")
        self._refresh_calibration_status_texts()

    def _refresh_calibration_status_texts(self) -> None:
        for finger in PINCH_FINGERS:
            # Status text is intentionally hidden in the calibration block.
            self.calibration_status_vars[finger].set("")

    def _finger_geometry(self, finger: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
        fingers_cfg = self.hand_model_config.get("fingers", {})
        finger_cfg = fingers_cfg.get(finger, {}) if isinstance(fingers_cfg, dict) else {}
        root = np.array(finger_cfg.get("root", [0.0, 0.0, 0.0]), dtype=np.float64)
        base = np.array(finger_cfg.get("base_direction", [0.0, 1.0, 0.0]), dtype=np.float64)
        lengths_raw = finger_cfg.get("segment_lengths", [17.0, 30.0, 43.0])
        if not isinstance(lengths_raw, list) or len(lengths_raw) < 3:
            lengths_raw = [17.0, 30.0, 43.0]
        lengths = (float(lengths_raw[0]), float(lengths_raw[1]), float(lengths_raw[2]))
        return root, base, lengths

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

    def _optimize_lateral_pair_for_finger(self, finger: str, maximize_distance: bool) -> tuple[float, float] | None:
        finger_angles = self.last_solved_angles.get(finger)
        thumb_angles = self.last_solved_angles.get("thumb")
        if finger_angles is None or thumb_angles is None:
            return None

        f_mcp, f_pip = finger_angles
        t_mcp, t_pip = thumb_angles

        f_root, f_base, f_lengths = self._finger_geometry(finger)
        t_root, t_base, t_lengths = self._finger_geometry("thumb")

        f_lim = self.finger_joint_limits.get(finger, {}).get("lateral", (-20.0, 20.0))
        t_lim = self.finger_joint_limits.get("thumb", {}).get("lateral", (-20.0, 20.0))

        best_score = -1e18 if maximize_distance else 1e18
        best_pair = (0.0, 0.0)
        f_candidates = np.linspace(f_lim[0], f_lim[1], 25)
        t_candidates = np.linspace(t_lim[0], t_lim[1], 25)

        for f_lat in f_candidates:
            f_pts = reconstruct_finger_points(f_root, f_base, f_lengths, f_mcp, f_pip, float(f_lat))
            for t_lat in t_candidates:
                t_pts = reconstruct_finger_points(t_root, t_base, t_lengths, t_mcp, t_pip, float(t_lat))
                d = float(np.linalg.norm(f_pts[-1] - t_pts[-1]))
                if maximize_distance:
                    score = d
                    if score > best_score:
                        best_score = score
                        best_pair = (float(f_lat), float(t_lat))
                else:
                    score = d
                    if score < best_score:
                        best_score = score
                        best_pair = (float(f_lat), float(t_lat))

        return best_pair

    def _build_thumb_pair_distances(self, world_landmarks: np.ndarray) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        # Existing point keeps prior behavior (thumb tip), new point is thumb DIP/IP joint.
        thumb_primary = world_landmarks[FINGERS["thumb"][3]]
        thumb_dip_joint = world_landmarks[THUMB_IP]

        blended: dict[str, float] = {}
        primary_dists: dict[str, float] = {}
        dip_dists: dict[str, float] = {}
        for finger in PINCH_FINGERS:
            finger_tip = world_landmarks[FINGERS[finger][3]]
            d_primary = float(np.linalg.norm(thumb_primary - finger_tip))
            d_dip = float(np.linalg.norm(thumb_dip_joint - finger_tip))
            d_blend = 0.5 * d_primary + 0.5 * d_dip
            primary_dists[finger] = d_primary
            dip_dists[finger] = d_dip
            blended[finger] = d_blend

        return blended, primary_dists, dip_dists

    def _alpha_from_distance(self, finger: str, dist: float) -> float:
        cal = self.distance_calibration.get(finger, {})
        far = cal.get("far")
        near = cal.get("near")

        # Fallback defaults in normalized world coordinate scale.
        if far is None or near is None or abs(float(far) - float(near)) < 1e-6:
            far = 0.14
            near = 0.025

        far_v = float(far)
        near_v = float(near)
        if far_v < near_v:
            far_v, near_v = near_v, far_v

        alpha = (far_v - float(dist)) / max(far_v - near_v, 1e-6)
        return float(np.clip(alpha, 0.0, 1.0))

    def _extract_landmarks(self, frame_rgb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        image_landmarks = None
        world_landmarks = None

        if self.backend_type == "solutions":
            result = self.backend.process(frame_rgb)
            if result.multi_hand_landmarks and result.multi_hand_world_landmarks:
                image_hand = result.multi_hand_landmarks[0]
                world_hand = result.multi_hand_world_landmarks[0]
                image_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in image_hand.landmark], dtype=np.float32)
                world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in world_hand.landmark], dtype=np.float32)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.backend.detect(mp_image)
            if result.hand_landmarks and result.hand_world_landmarks:
                image_hand = result.hand_landmarks[0]
                world_hand = result.hand_world_landmarks[0]
                image_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in image_hand], dtype=np.float32)
                world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in world_hand], dtype=np.float32)

        return image_landmarks, world_landmarks

    def _solve_model_from_landmarks(self, world_landmarks: np.ndarray) -> tuple[dict[str, tuple[float, float, float]], dict[str, np.ndarray]]:
        angles_for_table: dict[str, tuple[float, float, float]] = {}
        model_points: dict[str, np.ndarray] = {}
        pair_distances, primary_dists, dip_dists = self._build_thumb_pair_distances(world_landmarks)
        self.last_thumb_distances = dict(pair_distances)
        self.last_thumb_distances_primary = dict(primary_dists)
        self.last_thumb_distances_dip = dict(dip_dists)
        self._refresh_calibration_status_texts()

        # First solve pinch fingers flexion from thumb pair distances.
        pinch_alphas: dict[str, float] = {}
        flex_by_finger: dict[str, tuple[float, float]] = {}
        for finger in PINCH_FINGERS:
            dist = pair_distances.get(finger, 0.14)
            alpha = self._alpha_from_distance(finger, dist)
            pinch_alphas[finger] = alpha

            limits = self.finger_joint_limits.get(finger, {})
            mcp_lim = limits.get("mcp_flex", (0.0, 120.0))
            pip_lim = limits.get("proximal", (0.0, 120.0))

            mcp_target = mcp_lim[0] + (mcp_lim[1] - mcp_lim[0]) * alpha
            pip_target = pip_lim[0] + (pip_lim[1] - pip_lim[0]) * alpha
            mcp_flex = float(np.clip(self.ik_filters[(finger, "mcp_flex")].update(mcp_target), mcp_lim[0], mcp_lim[1]))
            pip_flex = float(np.clip(self.ik_filters[(finger, "proximal")].update(pip_target), pip_lim[0], pip_lim[1]))
            flex_by_finger[finger] = (mcp_flex, pip_flex)

        # Thumb follows strongest pinch demand among index/middle/ring.
        thumb_alpha = max(pinch_alphas.values()) if pinch_alphas else 0.0
        thumb_limits = self.finger_joint_limits.get("thumb", {})
        thumb_mcp_lim = thumb_limits.get("mcp_flex", (0.0, 120.0))
        thumb_pip_lim = thumb_limits.get("proximal", (0.0, 120.0))
        thumb_mcp_target = thumb_mcp_lim[0] + (thumb_mcp_lim[1] - thumb_mcp_lim[0]) * thumb_alpha
        thumb_pip_target = thumb_pip_lim[0] + (thumb_pip_lim[1] - thumb_pip_lim[0]) * thumb_alpha
        thumb_mcp = float(np.clip(self.ik_filters[("thumb", "mcp_flex")].update(thumb_mcp_target), thumb_mcp_lim[0], thumb_mcp_lim[1]))
        thumb_pip = float(np.clip(self.ik_filters[("thumb", "proximal")].update(thumb_pip_target), thumb_pip_lim[0], thumb_pip_lim[1]))
        self.last_solved_angles = {**flex_by_finger, "thumb": (thumb_mcp, thumb_pip)}

        dominant_finger = max(pinch_alphas, key=pinch_alphas.get) if pinch_alphas else "index"
        thumb_lat_lim = thumb_limits.get("lateral", (-20.0, 20.0))
        thumb_lateral_target = 0.0
        lateral_by_finger: dict[str, float] = {"thumb": 0.0}

        for finger in PINCH_FINGERS:
            alpha = pinch_alphas.get(finger, 0.0)
            limits = self.finger_joint_limits.get(finger, {})
            finger_lat_lim = limits.get("lateral", (-20.0, 20.0))
            finger_target, thumb_target_candidate = self._interpolate_lateral_from_calibration(
                finger,
                alpha,
                finger_lat_lim,
                thumb_lat_lim,
            )
            lateral_deg = float(
                np.clip(
                    self.ik_filters[(finger, "lateral")].update(finger_target),
                    finger_lat_lim[0],
                    finger_lat_lim[1],
                )
            )
            lateral_by_finger[finger] = lateral_deg

            if finger == dominant_finger:
                thumb_lateral_target = thumb_target_candidate

        thumb_lateral = float(
            np.clip(
                self.ik_filters[("thumb", "lateral")].update(thumb_lateral_target),
                thumb_lat_lim[0],
                thumb_lat_lim[1],
            )
        )
        lateral_by_finger["thumb"] = thumb_lateral

        for finger in FINGERS:
            flex_pair = self.last_solved_angles.get(finger, (0.0, 0.0))
            angles_for_table[finger] = (float(flex_pair[0]), float(flex_pair[1]), float(lateral_by_finger.get(finger, 0.0)))

        self.last_joint_dofs = dict(angles_for_table)

        for finger in FINGERS:
            mcp_flex, pip_flex, lateral_deg = angles_for_table.get(finger, (0.0, 0.0, 0.0))
            root, base, lengths = self._finger_geometry(finger)

            model_points[finger] = reconstruct_finger_points(
                root=root,
                base_direction=base,
                lengths=lengths,
                mcp_flex_deg=mcp_flex,
                proximal_deg=pip_flex,
                lateral_deg=lateral_deg,
            )

        return angles_for_table, model_points

    def _update_model_plot(self, model_points: dict[str, np.ndarray]) -> None:
        if self.model_axis is None or self.model_canvas is None:
            return

        self.model_axis.clear()
        self.model_axis.set_xlabel("X")
        self.model_axis.set_ylabel("Y")
        self.model_axis.set_zlabel("Z")
        self.model_axis.set_title("Inverse-Solved Hand Model")

        color_map = {"thumb": "#d62728", "index": "#1f77b4", "middle": "#2ca02c", "ring": "#ff7f0e"}
        all_pts: list[np.ndarray] = []
        for finger, pts in model_points.items():
            color = color_map.get(finger, "#444444")
            self.model_axis.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.8)
            self.model_axis.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=18)
            all_pts.append(pts)

        if all_pts:
            merged = np.vstack(all_pts)
            mins = np.min(merged, axis=0)
            maxs = np.max(merged, axis=0)
            center = (mins + maxs) * 0.5
            radius = max(float(np.max(maxs - mins)) * 0.6, 20.0)
            self.model_axis.set_xlim(center[0] - radius, center[0] + radius)
            self.model_axis.set_ylim(center[1] - radius, center[1] + radius)
            self.model_axis.set_zlim(center[2] - radius, center[2] + radius)
            if hasattr(self.model_axis, "set_box_aspect"):
                self.model_axis.set_box_aspect((1.0, 1.0, 1.0))

        self.model_axis.grid(True, alpha=0.4)
        # Match servocontrol.py default orientation: X toward left-lower, Y right, Z up.
        self.model_axis.view_init(elev=18, azim=20)
        self.model_canvas.draw_idle()

    def _render_video(self, frame_bgr: np.ndarray) -> None:
        if self.video_canvas is None:
            return
        label_w = max(160, self.video_canvas.winfo_width())
        label_h = max(120, self.video_canvas.winfo_height())

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = frame_rgb.shape[:2]
        scale = min(label_w / max(src_w, 1), label_h / max(src_h, 1))
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((label_h, label_w, 3), 24, dtype=np.uint8)
        y0 = (label_h - new_h) // 2
        x0 = (label_w - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

        image = Image.fromarray(canvas)
        photo = ImageTk.PhotoImage(image=image)
        if self.video_canvas_image_id is None:
            self.video_canvas_image_id = self.video_canvas.create_image(label_w // 2, label_h // 2, image=photo)
        else:
            self.video_canvas.itemconfig(self.video_canvas_image_id, image=photo)
            self.video_canvas.coords(self.video_canvas_image_id, label_w // 2, label_h // 2)
        self.video_image_ref = photo

    def _schedule_next_frame(self) -> None:
        self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    def _update_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Camera read failed")
            self._schedule_next_frame()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_landmarks, world_landmarks = self._extract_landmarks(frame_rgb)

        angle_rows: dict[str, tuple[float, float, float]] = {}
        model_points: dict[str, np.ndarray] = self.last_model_points
        try:
            detected = image_landmarks is not None and world_landmarks is not None
            if detected:
                image_landmarks = apply_landmark_filters(image_landmarks, self.image_landmark_filters)
                world_landmarks = apply_landmark_filters(world_landmarks, self.world_landmark_filters)
                frame, _ = draw_hand_overlay(frame, image_landmarks, world_landmarks, self.angle_filters)
                angle_rows, model_points = self._solve_model_from_landmarks(world_landmarks)
                self.last_model_points = model_points
                self._update_joint_table(angle_rows)
                self.detect_var.set("Hand: detected")
                self.status_var.set("Running")
            else:
                self.detect_var.set("Hand: not detected")
                self.status_var.set("Running | waiting for hand")
                self.last_thumb_distances = {}
                self.last_thumb_distances_primary = {}
                self.last_thumb_distances_dip = {}
                self.last_solved_angles = {}
                self.last_joint_dofs = {}
                self._update_joint_table({})
                self._refresh_calibration_status_texts()
        except Exception as exc:
            self.detect_var.set("Hand: error")
            self.status_var.set(f"Runtime error: {exc}")

        self._update_model_plot(model_points)
        self._render_video(frame)
        self._schedule_next_frame()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if hasattr(self.backend, "close"):
            self.backend.close()


def main() -> None:
    root = tk.Tk()
    app = HandPoseViewerApp(root)

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
