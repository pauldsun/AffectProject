"""Shared per-frame feature extraction (OpenFace 3.0 + PnP head pose + landmark geometry).
Used by live_demo.py (webcam) and extract_features.py (offline video), so both produce
identical columns. Every feature uses only the current frame: no smoothing, no lookahead."""
import contextlib
from dataclasses import dataclass
from pathlib import Path
import cv2, numpy as np, torch
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "OpenFace-3.0" / "weights"

EMOTIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]  # AffectNet order

# AU head output order is NOT documented. Probable mapping = standard DISFA 8-AU benchmark set:
# smiling sample images fire idx 5 + 6 (AU12 lip corner pull, AU25 lips part), which fits this order
# and not the train_mix_au_dev.py eval_au order. Webcam 2026-11-06: smile -> au_5 confirmed; frown gave no
# AU above 0.5 (AU4 at idx 2 still unconfirmed, check for sub-threshold rises). Columns stay index-named (au_0..au_7)
# so fixing this dict never requires re-extraction.
AU_LABELS = {0: "AU1 inner brow raise", 1: "AU2 outer brow raise", 2: "AU4 brow lower",
             3: "AU6 cheek raise", 4: "AU9 nose wrinkle", 5: "AU12 lip corner pull",
             6: "AU25 lips part", 7: "AU26 jaw drop"}

# WFLW 98-point indices (checked by drawing indices on images/0.jpg). "L"/"R" = image left/right.
EYE_L = dict(outer=60, inner=64, top=(61, 62, 63), bot=(67, 66, 65))
EYE_R = dict(outer=72, inner=68, top=(71, 70, 69), bot=(73, 74, 75))
BROW_L, BROW_R = range(33, 42), range(42, 51)
BROW_INNER_L, BROW_INNER_R = 38, 50
LIP_IN = dict(left=88, right=92, top=(89, 90, 91), bot=(95, 94, 93))

# Generic 3D face for PnP, OpenCV camera axes (x right, y down, z away from camera), arbitrary units.
# A frontal face looking at the camera has rotation = identity, so angles read as offsets from frontal.
PNP_IDX = [54, 16, 60, 72, 76, 82]                       # nose tip, chin, eye outer L/R, mouth corner L/R
PNP_MODEL = np.array([[0, 0, 0], [0, 330, 65],
                      [-225, -170, 135], [225, -170, 135],
                      [-150, 150, 125], [150, 150, 125]], dtype=np.float64)

FEATURE_COLUMNS = (
    ["face_detected", "det_score", "n_faces"]                         # tracking quality
    + [f"emo_{e}" for e in EMOTIONS]                                   # softmax probabilities
    + [f"au_{i}" for i in range(8)]                                    # AU presence scores
    + ["gaze_yaw", "gaze_pitch"]                                       # radians (MPIIGaze convention)
    + ["head_pitch", "head_yaw", "head_roll", "pnp_err"]               # degrees; pnp_err = fit quality
    + ["eye_open", "mouth_open", "brow_raise", "brow_gap"]             # landmark geometry / IOD
    + ["face_x", "face_y", "face_size"]                                # position in frame, 0-1
)

# The multitask model's shared EfficientNet-B0 backbone vector (the input all three heads read, before
# their task-specific layers). Not interpretable, but saved so a learned-representation model can be tried
# later without re-downloading and re-extracting DAiSEE. Stored per frame in the clip's .npz, not the CSV.
EMBED_DIM = 1280


@dataclass
class FrameResult:
    features: dict              # FEATURE_COLUMNS -> float (NaN when no face)
    landmarks: np.ndarray       # (98, 2) pixel coords, or None
    box: np.ndarray             # [x1, y1, x2, y2], or None
    rvec: np.ndarray = None     # PnP rotation/translation + intrinsics, for drawing axes
    tvec: np.ndarray = None
    K: np.ndarray = None
    embedding: np.ndarray = None  # (EMBED_DIM,) float32 backbone vector, or None


def empty_features():
    f = dict.fromkeys(FEATURE_COLUMNS, np.nan)
    f["face_detected"], f["n_faces"] = 0.0, 0.0
    return f


NO_DISTORTION = np.zeros(4)    # uncalibrated webcam/DAiSEE: assume no lens distortion


def camera_matrix(w, h):
    """Approximate pinhole intrinsics (focal = image width, ~53 deg FOV, centered). DAiSEE and the
    webcam are uncalibrated; the approximation mostly affects absolute pitch/yaw scale, not changes."""
    return np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)


def head_pose(lm, w, h):
    """Fit the generic 3D face to 6 landmarks. Returns (pitch, yaw, roll) in degrees, normalized
    reprojection error, rvec, tvec, K. Angles use R = Rz(roll) Ry(yaw) Rx(pitch):
    +pitch = face tilts down, +yaw = face turns toward image left, +roll = clockwise in image."""
    K = camera_matrix(w, h)
    img_pts = lm[PNP_IDX].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(PNP_MODEL, img_pts, K, NO_DISTORTION, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return (np.nan,) * 4 + (None, None, K)
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    proj, _ = cv2.projectPoints(PNP_MODEL, rvec, tvec, K, NO_DISTORTION)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean() / interocular(lm)
    return pitch, yaw, roll, err, rvec, tvec, K


def interocular(lm):
    return np.linalg.norm(lm[EYE_L["outer"]] - lm[EYE_R["outer"]]) + 1e-6


def opening(lm, d):
    """Mean lid/lip gap over corner-to-corner width (eye aspect ratio style)."""
    gap = np.linalg.norm(lm[list(d["top"])] - lm[list(d["bot"])], axis=1).mean()
    a, b = (d["outer"], d["inner"]) if "outer" in d else (d["left"], d["right"])
    return gap / (np.linalg.norm(lm[a] - lm[b]) + 1e-6)


def geometry(lm):
    """Scale-free shape features; all distances divided by interocular distance (IOD)."""
    iod = interocular(lm)
    eye_top = lm[list(EYE_L["top"]) + list(EYE_R["top"])].mean(0)
    brows = lm[list(BROW_L) + list(BROW_R)].mean(0)
    return {
        "eye_open": (opening(lm, EYE_L) + opening(lm, EYE_R)) / 2,       # blinks, drowsy eyes
        "mouth_open": opening(lm, LIP_IN),                                 # talking, yawning
        "brow_raise": np.linalg.norm(brows - eye_top) / iod,              # high = raised, low = lowered
        "brow_gap": np.linalg.norm(lm[BROW_INNER_L] - lm[BROW_INNER_R]) / iod,  # low = brows drawn together
    }


class ArrayFaceDetector(FaceDetector):
    """get_face()/detect_faces() for BGR ndarrays (webcam/video frames), not just file paths.
    Fixes shipped bug: preprocess_image hardcodes cv2.imread (breaks detect-video)."""
    def preprocess_image(self, image, resize=1.0):
        img_raw = image if isinstance(image, np.ndarray) else cv2.imread(image)
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).unsqueeze(0).to(self.device), img_raw


class FeatureExtractor:
    """Loads the three OpenFace 3.0 models once; call .process(frame) per BGR frame."""

    def __init__(self, device="cpu", min_score=0.5):
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be 'cpu', 'cuda' or 'mps'")
        self.min_score = min_score
        # LandmarkDetector only accepts cpu/cuda, and STAR measured no faster on mps, so on a Mac the
        # detector + multitask model use the GPU (~10x faster than this torch build's CPU) and STAR stays on CPU.
        lmk_device = "cuda" if device == "cuda" else "cpu"
        # RetinaFace hardcodes torch.load("./weights/mobilenetV1X0.25_pretrain.tar") at construction,
        # so chdir only while loading; callers' relative paths keep working afterwards.
        with contextlib.chdir(WEIGHTS.parent):
            self.det = ArrayFaceDetector(model_path=str(WEIGHTS / "Alignment_RetinaFace.pth"), device=device)
            self.lmk = LandmarkDetector(str(WEIGHTS / "Landmark_98.pkl"), device=lmk_device,
                                        device_ids=[0] if device == "cuda" else [-1])
            self.mtl = MultitaskPredictor(str(WEIGHTS / "MTL_backbone.pth"), device=device)

    def multitask(self, face):
        """Same computation as MTL.forward (openface/model/MTL.py), but also returns the backbone vector."""
        m = self.mtl.model
        with torch.no_grad():
            z = m.base_model(self.mtl.preprocess(face))
            emo = m.emotion_classifier(m.relu(m.fc_emotion(z)))
            gaze = m.gaze_regressor(m.relu(m.fc_gaze(z)))
            aus = m.au_regressor(m.relu(m.fc_au(z)))
        return emo, gaze, aus, z[0].float().cpu().numpy()

    def landmarks(self, frame, box):
        # STAR sees a square crop of side 200*scale around the box center. OpenFace's wrapper uses the box's
        # SHORT side, but RetinaFace boxes are taller than wide, so the crop cut off the chin and STAR drew the
        # jawline up near the lower lip (and pulled the PnP chin point up). The LONG side keeps the whole face in.
        x1, y1, x2, y2 = box
        scale = max(x2 - x1, y2 - y1) / 200 * 1.05
        return self.lmk.alignment.analyze(frame, float(scale), float((x1 + x2) / 2), float((y1 + y2) / 2))

    def process(self, frame):
        h, w = frame.shape[:2]
        f = empty_features()
        dets, _ = self.det.detect_faces(frame)
        dets = dets[dets[:, 4] >= self.min_score] if len(dets) else dets
        f["n_faces"] = float(len(dets))
        if len(dets) == 0:
            return FrameResult(f, None, None)

        # The LARGEST face, not the highest score: a desk/robot camera sees its own user nearest, but classmates
        # behind can score higher (DIPSER lecture: ~10% of frames were on the wrong person with dets[0]).
        det = dets[np.argmax((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))]
        x1, y1, x2, y2 = np.clip(det[:4], 0, [w, h, w, h]).astype(int)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return FrameResult(f, None, None)
        f["face_detected"], f["det_score"] = 1.0, float(det[4])

        emo, gaze, aus, emb = self.multitask(face)
        f.update(zip([f"emo_{e}" for e in EMOTIONS], torch.softmax(emo, 1)[0].tolist()))
        f.update(zip([f"au_{i}" for i in range(8)], aus[0].tolist()))
        f["gaze_yaw"], f["gaze_pitch"] = gaze[0].tolist()

        lm = self.landmarks(frame, (x1, y1, x2, y2))
        pitch, yaw, roll, err, rvec, tvec, K = head_pose(lm, w, h)
        f.update(head_pitch=pitch, head_yaw=yaw, head_roll=roll, pnp_err=err)
        f.update(geometry(lm))
        f.update(face_x=(x1 + x2) / 2 / w, face_y=(y1 + y2) / 2 / h,
                 face_size=np.sqrt((x2 - x1) * (y2 - y1)) / w)
        return FrameResult(f, lm, np.array([x1, y1, x2, y2]), rvec, tvec, K, emb)
