import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

# ============================================================
# 2D Robot + Noisy Sensor + Kalman Filter + (NEW) Features:
# 1/ Slow down near goal (speed scheduling)
# 2/ Sensor noise grows with speed (speed-dependent SENSOR_STD)
# 3/ Adaptive R using a tiny online ML regressor (learns trust)
# 4/ Outlier gating (NIS test) -> inflate R or skip update
# 5/ Pause & re-measure near goal to shrink ellipse
# 6/ Measurements plotted as points that stay (no misleading line)
# ============================================================

# -------------------------
# Config
# -------------------------
DT = 0.05  # keep small for stability (0.01–0.1 is typical)
START = np.array([0.0, 0.0], dtype=float)
GOAL  = np.array([10.0, 7.0], dtype=float)

# Speed schedule
V_MAX = 1.5
V_MIN = 0.05
SLOW_RADIUS = 2.0      # start slowing when within this distance
STOP_RADIUS = 0.15     # "arrived" radius

# Sensor noise (base + speed-dependent)
SENSOR_STD_BASE = 0.40   # noise when nearly stopped
SENSOR_STD_KV   = 0.80   # extra noise per unit speed (tune)

# KF process noise baseline (will scale with DT and speed)
Q_BASE = 0.005

# Outlier gating (Normalized Innovation Squared threshold)
# For 2D Gaussian, ~ 95% quantile is about 5.99, 99% ~ 9.21
NIS_THRESH = 9.21
GATING_MODE = "inflate"   # "inflate" or "skip"
R_INFLATE_FACTOR = 50.0

# Pause & re-measure near goal
PAUSE_DIST = 0.8          # if within this distance, may pause
PAUSE_P_AREA = 0.02       # if ellipse area big, pause for more sensing
PAUSE_FRAMES = 10         # how many frames to pause (and measure)

# Measurement history for ellipse visualization
MEAS_WINDOW = 60

# Random seed (optional)
# np.random.seed(0)

# -------------------------
# Helpers
# -------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def speed_schedule(dist):
    """
    Smooth-ish slowdown: far -> V_MAX, near -> V_MIN.
    Linear schedule inside SLOW_RADIUS.
    """
    if dist <= STOP_RADIUS:
        return 0.0
    if dist >= SLOW_RADIUS:
        return V_MAX
    # linearly interpolate between V_MIN and V_MAX
    t = (dist - STOP_RADIUS) / (SLOW_RADIUS - STOP_RADIUS)
    return V_MIN + t * (V_MAX - V_MIN)

def control_toward_goal(x_hat):
    direction = GOAL - x_hat
    dist = LA.norm(direction)
    vmag = speed_schedule(dist)
    if dist < 1e-9 or vmag == 0.0:
        return np.zeros(2), dist
    return (direction / dist) * vmag, dist

def sensor_std_from_speed(vmag):
    return SENSOR_STD_BASE + SENSOR_STD_KV * vmag

def measure_position(x_true, sensor_std):
    noise = np.random.normal(0.0, sensor_std, size=2)
    return x_true + noise

def draw_covariance_ellipse(ax, mean, cov, n_std=2.0, color="orange", alpha=0.25):
    eigenvals, eigenvecs = LA.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvals, 0.0))
    e = Ellipse(xy=mean, width=width, height=height, angle=angle,
                color=color, alpha=alpha)
    ax.add_patch(e)
    return eigenvals, eigenvecs

def ellipse_area_from_P(P, n_std=2.0):
    # Area of 2D ellipse with semi-axes a,b: area = pi*a*b
    # Here width = 2*n_std*sqrt(l1), height = 2*n_std*sqrt(l2)
    # So semi-axes: a = n_std*sqrt(l1), b = n_std*sqrt(l2)
    l, _ = LA.eigh(P)
    l = np.maximum(l, 0.0)
    a = n_std * np.sqrt(l[1])  # (sorted ascending from eigh)
    b = n_std * np.sqrt(l[0])
    return np.pi * a * b

# -------------------------
# Tiny online "ML" model for adaptive R
# We learn a scalar scale factor for R:
#   R_t = (scale_t) * (sensor_std^2) * I
#
# Model: log(scale_t) = w · feat
# - feats are simple & interpretable
# - trained online using SIMULATION truth (x_true) as supervision
#   (for real robots you'd train in sim/offline or use heuristics)
# -------------------------
class OnlineRScaler:
    def __init__(self, n_feat, lr=0.05):
        self.w = np.zeros(n_feat)
        self.lr = lr

    def features(self, vmag, nis, meas_jump):
        # Simple normalized features (keep magnitudes sane)
        # 1 = bias term
        return np.array([
            1.0,
            clamp(vmag / V_MAX, 0.0, 2.0),
            clamp(nis / NIS_THRESH, 0.0, 5.0),
            clamp(meas_jump / 1.0, 0.0, 5.0),
        ])

    def predict_scale(self, feat):
        # scale = exp(w·feat), clipped for stability
        s = float(np.exp(np.dot(self.w, feat)))
        return clamp(s, 0.2, 20.0)

    def update(self, feat, target_scale):
        # SGD on squared error in log-space:
        # want w·feat ≈ log(target_scale)
        target_log = float(np.log(clamp(target_scale, 0.2, 20.0)))
        pred_log = float(np.dot(self.w, feat))
        err = (pred_log - target_log)
        self.w -= self.lr * err * feat

r_model = OnlineRScaler(n_feat=4, lr=0.03)

# -------------------------
# State (truth, measurements, KF)
# -------------------------
x_true = START.copy()
x_hat  = START.copy()
P = np.eye(2) * 0.01

trajectory_true = [x_true.copy()]
trajectory_kf   = [x_hat.copy()]
trajectory_meas = []  # store ALL measurements (points stay)
last_meas = None

pause_counter = 0

# -------------------------
# Plot setup
# -------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 12)
ax.set_title("True vs Kalman + Measurement Points + Uncertainty Ellipses")

goal_dot, = ax.plot(GOAL[0], GOAL[1], "X", markersize=10, label="goal")
true_dot, = ax.plot([], [], "o", markersize=8, label="true")
kf_dot,   = ax.plot([], [], "o", markersize=8, color="green", label="Kalman estimate")

true_path, = ax.plot([], [], lw=2, label="true path")
kf_path,   = ax.plot([], [], lw=2, color="green", label="Kalman path")

# Measurements as points that stay (NOT a connected line)
meas_scatter = ax.scatter([], [], s=18, alpha=0.35, label="measurements")

# Debug text
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

ax.legend(loc="upper right")

# -------------------------
# Animation update loop
# -------------------------
def update(frame):
    global x_true, x_hat, P, pause_counter, last_meas
    BURN_IN = 20  # frames

    if frame < BURN_IN:
        v, dist_to_goal = control_toward_goal(x_hat)
        v *= frame / BURN_IN   # ramp speed from 0 → normal
    else:
        v, dist_to_goal = control_toward_goal(x_hat)


    # 0) decide control (based on belief)
    v, dist_to_goal = control_toward_goal(x_hat)
    vmag = LA.norm(v)

    # 0.5) pause logic near goal: if close AND uncertainty still big, pause to remeasure
    P_area = ellipse_area_from_P(P, n_std=2.0)
    if pause_counter == 0 and dist_to_goal < PAUSE_DIST and P_area > PAUSE_P_AREA:
        pause_counter = PAUSE_FRAMES

    # 1) update TRUE state (ground truth motion)
    # If pausing, do not move (simulate "stopping to measure")
    if pause_counter > 0:
        v_used = np.zeros(2)
        pause_counter -= 1
    else:
        v_used = v

    x_true = x_true + v_used * DT

    # 2) sensor measurement with speed-dependent std
    sensor_std = sensor_std_from_speed(LA.norm(v_used))
    x_meas = measure_position(x_true, sensor_std)

    # store measurement point permanently
    trajectory_meas.append(x_meas.copy())
    meas_jump = 0.0 if last_meas is None else LA.norm(x_meas - last_meas)
    last_meas = x_meas.copy()

    # 3) Kalman PREDICT (position-only)
    x_pred = x_hat + v_used * DT

    # Make Q scale with DT and speed (simple, stable approximation)
    q_scale = (DT ** 2) * (1.0 + 2.0 * (LA.norm(v_used) / max(V_MAX, 1e-9)))
    Q = np.eye(2) * (Q_BASE * q_scale)
    P_pred = P + Q

    # 4) Compute innovation and NIS for gating/adaptive R features
    y = x_meas - x_pred
    # Base measurement covariance from sensor_std
    R_base = np.eye(2) * (sensor_std ** 2)

    # Innovation covariance using base R (for NIS + features)
    S_base = P_pred + R_base
    try:
        nis = float(y.T @ LA.inv(S_base) @ y)
    except LA.LinAlgError:
        nis = 1e9  # treat as extreme

    # 5) Adaptive R (tiny ML predicts scale factor)
    feat = r_model.features(vmag=LA.norm(v_used), nis=nis, meas_jump=meas_jump)
    r_scale = r_model.predict_scale(feat)
    R = r_scale * R_base

    # 6) Outlier gating
    gated = False
    if nis > NIS_THRESH:
        gated = True
        if GATING_MODE == "skip":
            # Skip update: keep prediction as belief
            x_hat = x_pred
            P = P_pred
        else:
            # Inflate R heavily to reduce measurement influence
            R = (R_INFLATE_FACTOR * r_scale) * R_base

    # 7) Kalman UPDATE (if not skipped)
    if not (gated and GATING_MODE == "skip"):
        S = P_pred + R
        K = P_pred @ LA.inv(S)
        x_hat = x_pred + K @ y
        P = (np.eye(2) - K) @ P_pred

    # 8) store trajectories
    trajectory_true.append(x_true.copy())
    trajectory_kf.append(x_hat.copy())

    # 9) stop near goal (both close AND fairly confident)
    if dist_to_goal < STOP_RADIUS and ellipse_area_from_P(P, n_std=2.0) < 0.01:
        ani.event_source.stop()

    # 10) update plots
    t_true = np.array(trajectory_true)
    t_kf   = np.array(trajectory_kf)

    true_dot.set_data([x_true[0]], [x_true[1]])
    kf_dot.set_data([x_hat[0]], [x_hat[1]])

    true_path.set_data(t_true[:, 0], t_true[:, 1])
    kf_path.set_data(t_kf[:, 0], t_kf[:, 1])

    # measurements stay as points
    meas_scatter.set_offsets(np.array(trajectory_meas))

    # 11) remove old ellipses
    for patch in ax.patches[:]:
        patch.remove()

    # 12) measurement spread ellipse (windowed empirical cov)
    if len(trajectory_meas) >= 10:
        recent = np.array(trajectory_meas[-MEAS_WINDOW:])
        mean_m = np.mean(recent, axis=0)
        cov_m  = np.cov(recent.T)
        draw_covariance_ellipse(ax, mean_m, cov_m, n_std=2.0, color="orange", alpha=0.18)

    # 13) Kalman belief ellipse (P)
    draw_covariance_ellipse(ax, x_hat, P, n_std=2.0, color="green", alpha=0.15)

    # 14) debug text
    info_text.set_text(
        "dist_to_goal: %.3f\n"
        "speed: %.3f\n"
        "sensor_std: %.3f\n"
        "R_scale(ML): %.2f\n"
        "NIS: %.2f%s\n"
        "P_area(2σ): %.4f\n"
        "pause: %d"
        % (
            dist_to_goal,
            LA.norm(v_used),
            sensor_std,
            r_scale,
            nis,
            "  (gated)" if gated else "",
            ellipse_area_from_P(P, n_std=2.0),
            pause_counter,
        )
    )

    return true_dot, kf_dot, true_path, kf_path, meas_scatter, info_text

ani = FuncAnimation(fig, update, frames=np.arange(2000), interval=30, blit=False, repeat=False)
plt.show()
