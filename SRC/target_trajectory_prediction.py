"""Predicción de trayectoria de objetivos (Dron Tracking) con YOLO + filtro de Kalman.

Flujo por fotograma:
    YOLO .track()  ->  centroide del BBox  ->  Filtro de Kalman (P, V, A)
    ->  cinemática (rumbo, velocidad angular)  ->  selección de modelo CV / CA / CT
    ->  posición futura + incertidumbre  ->  dibujo en pantalla

Todo se calcula en el plano de la imagen (x, y) en píxeles. Con una sola cámara no se
observa la profundidad (z), por eso el vector de estado es 2D.

Uso:
    python target_trajectory_prediction.py --weights best.pt --source 0
    python target_trajectory_prediction.py --weights best.pt --source video.mp4 \
        --output salida.mp4 --horizon 0.5 --cm-per-pixel 0.12

Teclas: 'q' salir, 'r' reiniciar el seguimiento.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("target_prediction")

# ======================================================================================
# 1. CONFIGURACIÓN
# ======================================================================================


@dataclass(frozen=True)
class KalmanConfig:
    """Parámetros del filtro. Subir `jerk_noise_density` = confía más en la medición."""

    measurement_std_px: float = 4.0        # ruido típico del centroide de YOLO (px)
    jerk_noise_density: float = 5.0e3      # intensidad del ruido de proceso (px²/s⁵)
    initial_velocity_std: float = 150.0    # incertidumbre inicial de velocidad (px/s)
    initial_accel_std: float = 500.0       # incertidumbre inicial de aceleración (px/s²)


@dataclass(frozen=True)
class PredictionConfig:
    """Umbrales de selección de modelo y límites de envolvente física."""

    horizon_s: float = 0.5                 # cuánto hacia el futuro se predice (s)
    path_steps: int = 10                   # puntos de la trayectoria dibujada
    min_speed_px_s: float = 30.0           # por debajo, el rumbo no es confiable
    turn_rate_threshold_rad_s: float = 0.5   # |ω| ≥ umbral -> giro coordinado (CT)
    accel_threshold_px_s2: float = 150.0   # |a| ≥ umbral -> aceleración constante (CA)
    omega_window: int = 10                 # fotogramas usados para Δθ/Δt
    omega_smoothing: float = 0.5           # 0..1, peso de la medición nueva de ω
    model_hold_frames: int = 5             # histéresis: mínimo de cuadros por modelo
    max_speed_px_s: float = 3000.0         # envolvente: velocidad máxima creíble
    max_accel_px_s2: float = 2000.0        # envolvente: aceleración máxima creíble
    uncertainty_sigma: float = 2.0         # tamaño de la elipse de incertidumbre


@dataclass(frozen=True)
class TrackingConfig:
    """Gestión de pistas (tracks)."""

    max_missed_frames: int = 15            # cuadros sin detección antes de borrar el track
    reinit_distance_px: float = 250.0      # salto que obliga a reiniciar el filtro
    trail_length: int = 50                 # puntos del rastro dibujado


@dataclass(frozen=True)
class AppConfig:
    weights: Path
    source: str = "0"
    output: Optional[Path] = None
    confidence: float = 0.5
    device: Optional[str] = None
    show_window: bool = True
    cm_per_pixel: Optional[float] = None
    kalman: KalmanConfig = KalmanConfig()
    prediction: PredictionConfig = PredictionConfig()
    tracking: TrackingConfig = TrackingConfig()


# ======================================================================================
# 2. CARGA DEL MODELO YOLO (.pt)
# ======================================================================================


def validate_weights_path(weights_path: Path) -> Path:
    """Comprueba que el archivo de pesos exista y sea un .pt."""
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {path}")
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Se esperaba un archivo .pt, se recibió: {path.name}")
    return path


def select_device(requested: Optional[str] = None) -> str:
    """Devuelve el dispositivo a usar: el pedido, la GPU si existe, o CPU."""
    if requested:
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_yolo_model(weights_path: Path):
    """Carga el modelo YOLO desde un .pt. Ultralytics se importa aquí para que el
    resto del módulo (Kalman, cinemática) sea utilizable y probable sin él."""
    path = validate_weights_path(weights_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Instala ultralytics: pip install ultralytics") from exc
    model = YOLO(str(path))
    LOGGER.info("Modelo cargado: %s | clases: %s", path.name, model.names)
    return model


# ======================================================================================
# 3. CINEMÁTICA BÁSICA (cálculos de tu investigación)
# ======================================================================================


def bbox_centroid(bbox_xyxy: Sequence[float]) -> np.ndarray:
    """Punto medio del BBox (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def pixel_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """d = sqrt((x2 - x1)² + (y2 - y1)²), en píxeles."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def cm_per_pixel_from_reference(reference_cm: float, reference_px: float) -> float:
    """Escala a partir de un objeto de referencia de tamaño real conocido."""
    if reference_cm <= 0 or reference_px <= 0:
        raise ValueError("La referencia debe ser positiva.")
    return reference_cm / reference_px


def pixels_to_cm(value_px: float, cm_per_pixel: float) -> float:
    return value_px * cm_per_pixel


def heading_angle(velocity: Sequence[float]) -> float:
    """θ = atan2(vy, vx): dirección del movimiento (rad), con y hacia abajo en imagen."""
    return math.atan2(velocity[1], velocity[0])


def wrap_angle(angle: float) -> float:
    """Normaliza a [-π, π) para que un salto 179° -> -179° cuente como 2°."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angular_velocity(theta_prev: float, theta_now: float, dt: float) -> float:
    """ω = Δθ / Δt (rad/s)."""
    return wrap_angle(theta_now - theta_prev) / dt if dt > 0 else 0.0


# ======================================================================================
# 4. FILTRO DE KALMAN (modelo de aceleración constante)
# ======================================================================================


class KalmanFilter2D:
    """Filtro de Kalman lineal con estado [x, vx, ax, y, vy, ay].

    Mide solo la posición (centroide) y estima velocidad y aceleración, es decir, el
    Vector de Estado (P, V, A) de la consideración 1.
    """

    _MEASUREMENT_MATRIX = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=float)

    def __init__(self, config: KalmanConfig) -> None:
        self._cfg = config
        self._R = np.eye(2) * config.measurement_std_px ** 2
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False

    # ---- ciclo de vida -------------------------------------------------------------
    def initialize(self, measurement: Sequence[float]) -> None:
        self.x = np.array([measurement[0], 0, 0, measurement[1], 0, 0], dtype=float)
        block = np.diag(
            [
                self._cfg.measurement_std_px ** 2,
                self._cfg.initial_velocity_std ** 2,
                self._cfg.initial_accel_std ** 2,
            ]
        )
        self.P = np.kron(np.eye(2), block)
        self.initialized = True

    def predict(self, dt: float) -> None:
        F = self._transition_matrix(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)

    def update(self, measurement: Sequence[float]) -> None:
        H = self._MEASUREMENT_MATRIX
        innovation = np.asarray(measurement, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + self._R
        K = np.linalg.solve(S, H @ self.P).T          # ganancia de Kalman
        self.x = self.x + K @ innovation
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self._R @ K.T   # forma de Joseph (estable)

    def project(self, horizon_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Estado y covarianza dentro de `horizon_s` SIN modificar el filtro."""
        F = self._transition_matrix(horizon_s)
        return F @ self.x, F @ self.P @ F.T + self._process_noise(horizon_s)

    # ---- estado --------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[[0, 3]].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[[1, 4]].copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self.x[[2, 5]].copy()

    @staticmethod
    def position_covariance(covariance: np.ndarray) -> np.ndarray:
        return covariance[np.ix_([0, 3], [0, 3])]

    # ---- matrices ------------------------------------------------------------------
    @staticmethod
    def _transition_matrix(dt: float) -> np.ndarray:
        block = np.array([[1, dt, 0.5 * dt ** 2], [0, 1, dt], [0, 0, 1]])
        return np.kron(np.eye(2), block)

    def _process_noise(self, dt: float) -> np.ndarray:
        block = self._cfg.jerk_noise_density * np.array(
            [
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt],
            ]
        )
        return np.kron(np.eye(2), block)


# ======================================================================================
# 5. MODELOS DE PREDICCIÓN (CV, CA, CT)
# ======================================================================================


class MotionModel(str, Enum):
    CV = "CV"  # velocidad constante
    CA = "CA"  # aceleración constante
    CT = "CT"  # giro coordinado


def _as_time_column(t) -> np.ndarray:
    return np.asarray(t, dtype=float)[..., None]


def predict_constant_velocity(p0: np.ndarray, v0: np.ndarray, t) -> np.ndarray:
    """P_f = P_0 + V_0 · Δt"""
    return p0 + v0 * _as_time_column(t)


def predict_constant_acceleration(p0: np.ndarray, v0: np.ndarray, a0: np.ndarray, t) -> np.ndarray:
    """P_f = P_0 + V_0·Δt + ½·A_0·Δt².

    Envolvente: si la aceleración frena el movimiento en un eje, el objetivo se detiene
    en ese eje en lugar de "regresar" (la parábola seguiría hacia atrás).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stop = np.where(v0 * a0 < 0, -v0 / a0, np.inf)
    t_eff = np.minimum(_as_time_column(t), t_stop)
    return p0 + v0 * t_eff + 0.5 * a0 * t_eff ** 2


def predict_coordinated_turn(p0: np.ndarray, v0: np.ndarray, omega: float, t, min_omega: float = 1e-6) -> np.ndarray:
    """Arco de circunferencia: la velocidad rota a razón ω (rad/s)."""
    if abs(omega) < min_omega:
        return predict_constant_velocity(p0, v0, t)
    t = _as_time_column(t)
    sin_wt, cos_wt = np.sin(omega * t), np.cos(omega * t)
    dx = (v0[0] * sin_wt - v0[1] * (1.0 - cos_wt)) / omega
    dy = (v0[0] * (1.0 - cos_wt) + v0[1] * sin_wt) / omega
    return p0 + np.concatenate([dx, dy], axis=-1)


def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= max_norm or norm == 0 else vector * (max_norm / norm)


class MotionModelSelector:
    """Elige CV / CA / CT según lo observado, con histéresis para evitar parpadeo.

    Es una versión ligera de IMM: en vez de mezclar probabilidades, decide con umbrales
    sobre la aceleración y la velocidad angular estimadas.
    """

    def __init__(self, config: PredictionConfig) -> None:
        self._cfg = config
        self._current = MotionModel.CV
        self._frames_in_model = config.model_hold_frames

    def select(self, speed: float, accel_magnitude: float, omega: float) -> MotionModel:
        candidate = self._classify(speed, accel_magnitude, omega)
        self._frames_in_model += 1
        if candidate != self._current and self._frames_in_model >= self._cfg.model_hold_frames:
            self._current = candidate
            self._frames_in_model = 0
        return self._current

    def _classify(self, speed: float, accel_magnitude: float, omega: float) -> MotionModel:
        cfg = self._cfg
        if speed >= cfg.min_speed_px_s and abs(omega) >= cfg.turn_rate_threshold_rad_s:
            return MotionModel.CT
        if accel_magnitude >= cfg.accel_threshold_px_s2:
            return MotionModel.CA
        return MotionModel.CV


class TrajectoryPredictor:
    """Aplica el modelo elegido y la envolvente física (límites y borde del cuadro)."""

    def __init__(self, config: PredictionConfig, frame_size: Tuple[int, int]) -> None:
        self._cfg = config
        self._frame_w, self._frame_h = frame_size

    def predict_path(
        self,
        model: MotionModel,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        omega: float,
        times: np.ndarray,
    ) -> np.ndarray:
        velocity = _limit_norm(velocity, self._cfg.max_speed_px_s)
        if model is MotionModel.CT:
            points = predict_coordinated_turn(position, velocity, omega, times)
        elif model is MotionModel.CA:
            acceleration = _limit_norm(acceleration, self._cfg.max_accel_px_s2)
            points = predict_constant_acceleration(position, velocity, acceleration, times)
        else:
            points = predict_constant_velocity(position, velocity, times)
        return np.clip(points, [0, 0], [self._frame_w - 1, self._frame_h - 1])


# ======================================================================================
# 6. ESTIMACIÓN DE VELOCIDAD ANGULAR
# ======================================================================================


class TurnRateEstimator:
    """ω = Δθ/Δt sobre una ventana de rumbos del objetivo ya filtrado, suavizada."""

    def __init__(self, window: int, smoothing: float) -> None:
        self._history: Deque[Tuple[float, float]] = deque(maxlen=max(window, 2))
        self._alpha = smoothing
        self.omega = 0.0

    def update(self, timestamp: float, heading: float) -> None:
        self._history.append((timestamp, heading))
        if len(self._history) < 2:
            return
        oldest_t, oldest_heading = self._history[0]
        raw = angular_velocity(oldest_heading, heading, timestamp - oldest_t)
        self.omega = self._alpha * raw + (1.0 - self._alpha) * self.omega

    def reset(self) -> None:
        self._history.clear()
        self.omega = 0.0


# ======================================================================================
# 7. PISTAS (TRACKS) DE OBJETIVOS
# ======================================================================================


@dataclass(frozen=True)
class Detection:
    track_id: int
    class_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class TargetSnapshot:
    """Estado completo de un objetivo en un fotograma (lo que se dibuja)."""

    track_id: int
    class_id: int
    bbox: Optional[Tuple[float, float, float, float]]
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    speed: float
    heading: float
    omega: float
    model: MotionModel
    predicted_path: np.ndarray
    predicted_position: np.ndarray
    position_covariance: np.ndarray
    trail: np.ndarray
    missed_frames: int


class TargetTrack:
    """Un objetivo: filtro + cinemática + selección de modelo + predicción."""

    def __init__(self, detection: Detection, kalman_cfg: KalmanConfig, pred_cfg: PredictionConfig,
                 tracking_cfg: TrackingConfig, predictor: TrajectoryPredictor) -> None:
        self.track_id = detection.track_id
        self.class_id = detection.class_id
        self.bbox: Optional[Tuple[float, float, float, float]] = detection.bbox
        self.missed_frames = 0
        self._pred_cfg = pred_cfg
        self._tracking_cfg = tracking_cfg
        self._predictor = predictor
        self._filter = KalmanFilter2D(kalman_cfg)
        self._selector = MotionModelSelector(pred_cfg)
        self._turn_rate = TurnRateEstimator(pred_cfg.omega_window, pred_cfg.omega_smoothing)
        self._trail: Deque[Tuple[float, float]] = deque(maxlen=tracking_cfg.trail_length)
        self._clock = 0.0
        self._model = MotionModel.CV
        self._heading = 0.0

    def update(self, detection: Detection, dt: float) -> None:
        """Fotograma con detección: predecir y corregir con el centroide medido."""
        centroid = bbox_centroid(detection.bbox)
        self.bbox = detection.bbox
        self.missed_frames = 0
        if not self._filter.initialized:
            self._filter.initialize(centroid)
        else:
            self._filter.predict(dt)
            if pixel_distance(self._filter.position, centroid) > self._tracking_cfg.reinit_distance_px:
                self._filter.initialize(centroid)   # salto imposible: reiniciar
                self._turn_rate.reset()
            else:
                self._filter.update(centroid)
        self._refresh_kinematics(dt)

    def coast(self, dt: float) -> None:
        """Fotograma sin detección: solo se predice (el filtro 'navega' a ciegas)."""
        self._filter.predict(dt)
        self.missed_frames += 1
        self._refresh_kinematics(dt)

    def snapshot(self) -> TargetSnapshot:
        position = self._filter.position
        velocity = self._filter.velocity
        acceleration = self._filter.acceleration
        horizon = self._pred_cfg.horizon_s
        times = np.linspace(0.0, horizon, self._pred_cfg.path_steps + 1)[1:]
        path = self._predictor.predict_path(
            self._model, position, velocity, acceleration, self._turn_rate.omega, times
        )
        _, future_cov = self._filter.project(horizon)
        return TargetSnapshot(
            track_id=self.track_id,
            class_id=self.class_id,
            bbox=self.bbox if self.missed_frames == 0 else None,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            speed=float(np.linalg.norm(velocity)),
            heading=self._heading,
            omega=self._turn_rate.omega,
            model=self._model,
            predicted_path=path,
            predicted_position=path[-1],
            position_covariance=KalmanFilter2D.position_covariance(future_cov),
            trail=np.array(self._trail, dtype=np.int32).reshape(-1, 1, 2),
            missed_frames=self.missed_frames,
        )

    def _refresh_kinematics(self, dt: float) -> None:
        self._clock += dt
        velocity = self._filter.velocity
        speed = float(np.linalg.norm(velocity))
        if speed >= self._pred_cfg.min_speed_px_s:
            self._heading = heading_angle(velocity)
            self._turn_rate.update(self._clock, self._heading)
        else:
            self._turn_rate.reset()
        self._model = self._selector.select(
            speed, float(np.linalg.norm(self._filter.acceleration)), self._turn_rate.omega
        )
        self._trail.append(tuple(self._filter.position))


class TrackManager:
    """Mantiene un TargetTrack por ID de YOLO y elimina los que se pierden."""

    def __init__(self, kalman_cfg: KalmanConfig, pred_cfg: PredictionConfig,
                 tracking_cfg: TrackingConfig, frame_size: Tuple[int, int]) -> None:
        self._kalman_cfg = kalman_cfg
        self._pred_cfg = pred_cfg
        self._tracking_cfg = tracking_cfg
        self._predictor = TrajectoryPredictor(pred_cfg, frame_size)
        self._tracks: Dict[int, TargetTrack] = {}

    def update(self, detections: Sequence[Detection], dt: float) -> List[TargetSnapshot]:
        seen = set()
        for detection in detections:
            track = self._tracks.get(detection.track_id)
            if track is None:
                track = TargetTrack(detection, self._kalman_cfg, self._pred_cfg,
                                    self._tracking_cfg, self._predictor)
                self._tracks[detection.track_id] = track
            track.update(detection, dt)
            seen.add(detection.track_id)

        for track_id, track in list(self._tracks.items()):
            if track_id in seen:
                continue
            track.coast(dt)
            if track.missed_frames > self._tracking_cfg.max_missed_frames:
                del self._tracks[track_id]
        return [track.snapshot() for track in self._tracks.values()]

    def reset(self) -> None:
        self._tracks.clear()


# ======================================================================================
# 8. ENTRADA / SALIDA DE VIDEO
# ======================================================================================


class FrameClock:
    """Entrega Δt por fotograma: fijo (1/fps) en archivos, reloj real en cámara."""

    MIN_DT, MAX_DT = 1e-3, 0.25

    def __init__(self, fixed_dt: Optional[float]) -> None:
        self._fixed_dt = fixed_dt
        self._last = time.perf_counter()

    def tick(self) -> float:
        if self._fixed_dt is not None:
            return self._fixed_dt
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        return min(max(dt, self.MIN_DT), self.MAX_DT)


def open_video_source(source: str) -> Tuple[cv2.VideoCapture, bool]:
    """Abre cámara (índice numérico) o archivo. Devuelve (captura, es_en_vivo)."""
    is_live = source.isdigit()
    if not is_live and not Path(source).is_file():
        raise FileNotFoundError(f"No se encontró el video: {source}")
    capture = cv2.VideoCapture(int(source) if is_live else source)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
    return capture, is_live


def create_video_writer(path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de salida: {path}")
    return writer


def extract_detections(result) -> List[Detection]:
    """Convierte un resultado de `model.track()` en Detections (solo las que tienen ID)."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    return [
        Detection(track_id, class_id, tuple(map(float, box)), conf)
        for box, track_id, class_id, conf in zip(xyxy, ids, classes, confidences)
    ]


# ======================================================================================
# 9. DIBUJO
# ======================================================================================

_PALETTE = [(255, 56, 56), (56, 201, 255), (72, 249, 10), (255, 157, 151), (255, 178, 29), (207, 210, 49)]
_WHITE, _GREY = (255, 255, 255), (160, 160, 160)


def _track_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _to_point(array: np.ndarray) -> Tuple[int, int]:
    return int(round(array[0])), int(round(array[1]))


def _format_motion(snapshot: TargetSnapshot, cm_per_pixel: Optional[float]) -> str:
    scale, unit = (cm_per_pixel, "cm") if cm_per_pixel else (1.0, "px")
    accel = float(np.linalg.norm(snapshot.acceleration))
    return (f"v={snapshot.speed * scale:.0f}{unit}/s  a={accel * scale:.0f}{unit}/s2  "
            f"w={math.degrees(snapshot.omega):+.0f}deg/s")


def _draw_bbox(frame: np.ndarray, snapshot: TargetSnapshot, label: str, color) -> None:
    if snapshot.bbox is None:
        return
    x1, y1, x2, y2 = map(int, snapshot.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)


def _draw_uncertainty(frame: np.ndarray, snapshot: TargetSnapshot, sigma: float, color) -> None:
    """Elipse de incertidumbre: zona donde probablemente estará el objetivo."""
    eigenvalues, eigenvectors = np.linalg.eigh(snapshot.position_covariance)
    limit = max(frame.shape[:2]) // 4   # tope visual: al inicio la incertidumbre es enorme
    axes = [int(min(max(sigma * math.sqrt(max(v, 0.0)), 1.0), limit)) for v in eigenvalues[::-1]]
    angle = math.degrees(math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    cv2.ellipse(frame, _to_point(snapshot.predicted_position), tuple(axes), angle, 0, 360, color, 1, cv2.LINE_AA)


def _draw_prediction(frame: np.ndarray, snapshot: TargetSnapshot, sigma: float, color) -> None:
    path = np.vstack([snapshot.position, snapshot.predicted_path]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [path], False, color, 1, cv2.LINE_AA)
    target = _to_point(snapshot.predicted_position)
    cv2.drawMarker(frame, target, color, cv2.MARKER_CROSS, 18, 2)
    _draw_uncertainty(frame, snapshot, sigma, color)


def draw_target(frame: np.ndarray, snapshot: TargetSnapshot, class_names: Dict[int, str],
                pred_cfg: PredictionConfig, cm_per_pixel: Optional[float]) -> None:
    color = _GREY if snapshot.missed_frames else _track_color(snapshot.track_id)
    name = class_names.get(snapshot.class_id, str(snapshot.class_id))
    status = "LOST" if snapshot.missed_frames else snapshot.model.value
    label = f"{name}:{snapshot.track_id} [{status}]"

    _draw_bbox(frame, snapshot, label, color)
    if len(snapshot.trail) > 1:
        cv2.polylines(frame, [snapshot.trail], False, color, 2)
    cv2.circle(frame, _to_point(snapshot.position), 5, color, -1)
    _draw_prediction(frame, snapshot, pred_cfg.uncertainty_sigma, color)

    x, y = _to_point(snapshot.position)
    cv2.putText(frame, _format_motion(snapshot, cm_per_pixel), (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ======================================================================================
# 10. APLICACIÓN
# ======================================================================================


def run(config: AppConfig) -> None:
    model = load_yolo_model(config.weights)
    device = select_device(config.device)
    class_names: Dict[int, str] = dict(model.names)

    capture, is_live = open_video_source(config.source)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = create_video_writer(config.output, fps, frame_size) if config.output else None

    manager = TrackManager(config.kalman, config.prediction, config.tracking, frame_size)
    clock = FrameClock(None if is_live else 1.0 / fps)
    window_name = "Prediccion de trayectoria"
    if config.show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                LOGGER.info("Fin del video o lectura fallida.")
                break

            result = model.track(frame, persist=True, conf=config.confidence, device=device, verbose=False, classes = 67)[0]
            snapshots = manager.update(extract_detections(result), clock.tick())
            for snapshot in snapshots:
                draw_target(frame, snapshot, class_names, config.prediction, config.cm_per_pixel)

            if writer:
                writer.write(frame)
            if config.show_window:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    manager.reset()
                    LOGGER.info("Seguimiento reiniciado.")
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def parse_args(argv: Optional[Sequence[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Predicción de trayectoria de objetivos con YOLO + Kalman.")
    parser.add_argument("--weights", type=Path, required=True, help="Ruta al archivo .pt de YOLO")
    parser.add_argument("--source", default="0", help="Índice de cámara (0, 1...) o ruta de video")
    parser.add_argument("--output", type=Path, default=None, help="Guardar video anotado (.mp4)")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza de YOLO")
    parser.add_argument("--device", default=None, help="'cpu', '0', 'cuda:0'... (auto por defecto)")
    parser.add_argument("--horizon", type=float, default=PredictionConfig.horizon_s, help="Horizonte de predicción (s)")
    parser.add_argument("--cm-per-pixel", type=float, default=None,
                        help="Escala medida con un objeto de referencia; sin ella se muestra en px")
    parser.add_argument("--no-window", action="store_true", help="No abrir ventana (solo guardar video)")
    args = parser.parse_args(argv)
    return AppConfig(
        weights=args.weights,
        source=args.source,
        output=args.output,
        confidence=args.conf,
        device=args.device,
        show_window=not args.no_window,
        cm_per_pixel=args.cm_per_pixel,
        prediction=PredictionConfig(horizon_s=args.horizon),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()