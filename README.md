# fxdsp_compressor_detector

```python
# fxdsp_compressor_detector.py

from typing import Dict, Tuple, Literal

import jax
import jax.numpy as jnp
from jax import lax, jit
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Bias flags (numeric, JAX-safe)
# ---------------------------------------------------------------------

FORWARD_BIAS = 0
REVERSE_BIAS = 1
FULL_WAVE   = 2


# ---------------------------------------------------------------------
# Diode Rectifier
# ---------------------------------------------------------------------

def diode_rectifier_init(bias: int = FULL_WAVE, threshold: float = 0.3) -> Dict:
    """
    bias: one of FORWARD_BIAS, REVERSE_BIAS, FULL_WAVE (ints)
    """
    bias_flag = int(bias)

    state = {
        "bias": bias_flag,    # numeric flag
        "threshold": 0.0,
        "vt": 0.0,
        "scale": 0.0,
    }
    return diode_rectifier_set_threshold(state, threshold)


def diode_rectifier_set_threshold(state: Dict, threshold: float) -> Dict:
    threshold = jnp.clip(jnp.abs(threshold), 0.01, 0.9)

    scale = 1.0 - threshold
    bias = state["bias"]  # int

    is_reverse = (bias == REVERSE_BIAS)

    scale = jnp.where(is_reverse, -scale, scale)
    threshold = jnp.where(is_reverse, -threshold, threshold)

    vt = -0.1738 * threshold + 0.1735
    scale = scale / jnp.exp(1.0 / vt - 1.0)

    # new immutable state
    return {
        "bias": bias,
        "threshold": threshold,
        "vt": vt,
        "scale": scale,
    }


def diode_rectifier_process_block(x: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
    vt = state["vt"]
    scale = state["scale"]
    bias = state["bias"]

    is_full    = (bias == FULL_WAVE)
    is_reverse = (bias == REVERSE_BIAS)

    x_full = jnp.abs(x)
    x_rev  = -x
    x_fwd  = x

    x_proc = jnp.where(is_full, x_full, jnp.where(is_reverse, x_rev, x_fwd))
    y = jnp.exp((x_proc / vt) - 1.0) * scale

    return y, state


# ---------------------------------------------------------------------
# Dynamic envelope (peak / RMS)
# ---------------------------------------------------------------------

def dynamic_envelope_init(
    attack: float,
    release: float,
    sr: float,
    mode: Literal["peak", "rms"] = "peak",
) -> Dict:
    atk = jnp.exp(-1.0 / (attack * sr))
    rel = jnp.exp(-1.0 / (release * sr))

    mode_flag = jnp.array(1, dtype=jnp.int32) if mode == "rms" else jnp.array(0, dtype=jnp.int32)

    return {
        "env": jnp.array(0.0),
        "attack_coef": atk,
        "release_coef": rel,
        "mode": mode_flag,    # 0 = peak, 1 = rms
    }


def dynamic_envelope_step(x_t, state):
    y_prev = state["env"]
    is_rms = (state["mode"] == 1)

    x_abs = jnp.abs(x_t)
    x_sq  = x_t * x_t

    x_in = jnp.where(is_rms, x_sq, x_abs)

    is_attack = (x_in > y_prev)
    coef = jnp.where(is_attack, state["attack_coef"], state["release_coef"])

    y = (1.0 - coef) * x_in + coef * y_prev
    y = jnp.where(is_rms, jnp.sqrt(y), y)

    new_state = {
        "env": y,
        "attack_coef": state["attack_coef"],
        "release_coef": state["release_coef"],
        "mode": state["mode"],
    }
    return new_state, y


def dynamic_envelope_process(x: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
    def step(carry_state, x_t):
        return dynamic_envelope_step(x_t, carry_state)

    final_state, y_block = lax.scan(step, state, x)
    return y_block, final_state


# ---------------------------------------------------------------------
# Envelope detector (diode + envelope)
# ---------------------------------------------------------------------

def envelope_detector_init(
    bias: int = FULL_WAVE,
    threshold: float = 0.3,
    attack: float = 0.01,
    release: float = 0.1,
    sr: float = 48000,
    mode: Literal["peak", "rms"] = "peak",
) -> Dict:
    diode = diode_rectifier_init(bias, threshold)
    env   = dynamic_envelope_init(attack, release, sr, mode)
    return {"diode": diode, "env": env, "sr": jnp.array(float(sr))}


@jit
def envelope_detector_process_block(x: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
    y_rect, diode_state = diode_rectifier_process_block(x, state["diode"])
    y_env, env_state    = dynamic_envelope_process(y_rect, state["env"])

    # build new state immutably
    new_state = {
        "diode": diode_state,
        "env": env_state,
        "sr": state["sr"],
    }
    return y_env, new_state


# ---------------------------------------------------------------------
# Compressor Gain Computer (JAX-safe, vectorizable)
# ---------------------------------------------------------------------

def compressor_gain_computer(
    level_db,
    threshold_db,
    ratio,
    knee_db,
):
    """
    JAX-compatible static compressor gain curve.
    level_db can be scalar or array.
    """
    level_db      = jnp.asarray(level_db)
    threshold_db  = jnp.asarray(threshold_db)
    ratio         = jnp.asarray(ratio)
    knee_db       = jnp.asarray(knee_db)

    hard_gain = threshold_db + (level_db - threshold_db) / ratio - level_db

    lower = threshold_db - knee_db / 2.0
    upper = threshold_db + knee_db / 2.0
    delta = level_db - lower

    soft_gain = (1.0 / ratio - 1.0) * (delta * delta) / (2.0 * knee_db)

    use_soft = (knee_db > 0.0) & (level_db >= lower) & (level_db <= upper)
    below    = level_db < lower

    gain = jnp.where(
        below,
        0.0,
        jnp.where(use_soft, soft_gain, hard_gain),
    )

    # pure hard-knee if knee_db == 0
    gain = jnp.where(
        knee_db == 0.0,
        jnp.where(level_db < threshold_db, 0.0, hard_gain),
        gain,
    )
    return gain


@jit
def compressor_gain_curve(
    threshold_db=-20.0,
    ratio=4.0,
    knee_db=6.0,
    input_range=(-60.0, 0.0),
):
    inp = jnp.linspace(*input_range, 400)
    gain = compressor_gain_computer(inp, threshold_db, ratio, knee_db)
    out  = inp + gain
    return inp, out


# ---------------------------------------------------------------------
# Full Compressor Detector Block
# ---------------------------------------------------------------------

def compressor_detector_init(
    threshold_db=-20.0,
    ratio=4.0,
    knee_db=6.0,
    attack=0.01,
    release=0.1,
    sr=48000,
    mode="rms",
) -> Dict:
    env_detector = envelope_detector_init(
        bias=FULL_WAVE,
        threshold=0.3,
        attack=attack,
        release=release,
        sr=sr,
        mode=mode,
    )

    return {
        "env_detector": env_detector,
        "threshold_db": jnp.array(float(threshold_db)),
        "ratio": jnp.array(float(ratio)),
        "knee_db": jnp.array(float(knee_db)),
        "sr": jnp.array(float(sr)),
    }


@jit
def compressor_detector_process_block(x: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
    env, env_state = envelope_detector_process_block(x, state["env_detector"])

    new_state = {
        "env_detector": env_state,
        "threshold_db": state["threshold_db"],
        "ratio": state["ratio"],
        "knee_db": state["knee_db"],
        "sr": state["sr"],
    }

    eps = 1e-12
    level_db = 20.0 * jnp.log10(jnp.maximum(env, eps))

    gain_db = compressor_gain_computer(
        level_db,
        new_state["threshold_db"],
        new_state["ratio"],
        new_state["knee_db"],
    )

    gain_lin = jnp.power(10.0, gain_db / 20.0)
    return gain_lin, new_state


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_static_curve():
    x, y = compressor_gain_curve(threshold_db=-18, ratio=4, knee_db=6)
    x = jnp.array(x); y = jnp.array(y)

    plt.figure(figsize=(6, 5))
    plt.plot(x, y, lw=2)
    plt.title("Compressor Static Input/Output Curve")
    plt.xlabel("Input Level (dB)")
    plt.ylabel("Output Level (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_dynamic_response():
    sr = 48000
    t = jnp.linspace(0, 0.5, int(sr * 0.5))

    amp = jnp.concatenate(
        [
            jnp.linspace(0.05, 1.0, int(0.2 * sr)),
            jnp.linspace(1.0, 0.1, int(0.3 * sr)),
        ]
    )
    x = amp * jnp.sin(2 * jnp.pi * 220 * t)

    state = compressor_detector_init(
        threshold_db=-20,
        ratio=4,
        knee_db=6,
        attack=0.005,
        release=0.2,
        sr=sr,
    )

    gain, _ = compressor_detector_process_block(x, state)

    t_np   = jnp.asarray(t)
    amp_np = jnp.asarray(amp)
    gain_np= jnp.asarray(gain)

    plt.figure(figsize=(9, 4))
    plt.plot(t_np, amp_np, color="gray", lw=1, label="Input Amplitude")
    plt.plot(t_np, gain_np, lw=2, color="orange", label="Gain Reduction (linear)")
    plt.title("Compressor Detector Envelope + Gain Reduction")
    plt.xlabel("Time [s]")
    plt.ylabel("Linear Gain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    plot_static_curve()
    plot_dynamic_response()

```
