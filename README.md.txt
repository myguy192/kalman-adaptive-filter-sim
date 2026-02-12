# Adaptive Kalman Filter Simulation (2D)

Prototype simulation of a 2D robot using a Kalman filter under noisy measurements.
Includes adaptive measurement covariance, outlier gating, and speed-dependent noise.

## What this is
- Simulates true robot motion toward a goal
- Adds noisy position measurements
- Applies a Kalman filter with:
  - Speed-dependent sensor noise
  - NIS-based outlier gating
  - Adaptive R scaling
  - Pause-and-remeasure behavior near the goal

## Why I built it
I built this to explore how filtering behavior changes under non-constant noise
and to test robustness strategies before applying similar ideas to real sensors.

## Key assumptions
- Measurement noise approximated as Gaussian
- Linear motion model
- Adaptive R trained in simulation

## Limitations
- Linear Kalman filter
- Adaptive R not deployable as-is
- Prototype / learning-focused

## How to run
```bash
pip install -r requirements.txt
python simulation.py
