"""
Animate a 3D path with Manim.

Put `trajectory.npy` or `trajectory.csv` (shape `(N,3)`) next to this file,
then run:  manim -pql trajectory.py Animate
Or inject points with set_points(...) and run the same command.
"""

import os
import argparse

import numpy as np
from manim import (
	DEGREES,
	BLUE,
	GOLD,
	ORIGIN,
	WHITE,
	UL,
	UR,
	ThreeDAxes,
	ThreeDScene,
	Sphere,
	VMobject,
	Create,
	MoveAlongPath,
	FadeIn,
	Text,
	linear,
)


# Optional module-level data injection point (Nx3 array)
POINTS = None

# Module-level visualization configuration
CONFIG = {
	# Minimal knobs
	"MAX_POINTS": 5000,       # decimate input to at most this many points
	"TARGET_SPAN": 10.0,      # rescale largest data span to this world span (uniform mode)
	"DRAW_TIME": 6.0,         # seconds to draw full path
	"SCALE_MODE": "per-axis", # 'uniform' | 'per-axis' | 'none'

	# Show rescale factor HUD
	"SHOW_SCALE_INFO": True,  # display scale factor used for rescaling
	"UNITS_LABEL": None,      # e.g., "m" for meters; None to omit
	"SCALE_INFO_POSITION": "UL",  # "UL" or "UR"

	# Per-axis target spans (only used if SCALE_MODE == 'per-axis'). If None, each uses TARGET_SPAN.
	"TARGET_SPAN_PER_AXIS": None,  # e.g., (8.0, 2.0, 0.5)
	# If an axis span is <= this absolute tolerance, treat it as exactly constant
	"ZERO_SPAN_TOL": 1e-15,

	# Debug/verification aids
	"DEBUG_PRINT": True,         # print min/max before and after rescale
	"SHOW_AXIS_LABELS": True,    # show 'x', 'y', 'z' labels on axes
	"FORCE_XY_VIEW": False,       # look straight along +z (top-down) to verify XY planarity
}


def set_config(**kwargs):
	"""Update CONFIG keys from kwargs."""
	for k, v in kwargs.items():
		if k in CONFIG:
			CONFIG[k] = v


def set_points(x=None, y=None, z=None, points=None):
	"""Set global POINTS. Either give x,y,z (same length) or an (N,3) array."""
	global POINTS
	POINTS = _standardize_points(x=x, y=y, z=z, points=points)

def _standardize_points(*, x=None, y=None, z=None, points=None, dtype=float):
	"""Return (N,3) float array; filters NaN/Inf and consecutive duplicates."""
	if points is not None:
		arr = np.asarray(points, dtype=dtype)
		if arr.ndim != 2 or arr.shape[1] != 3:
			raise ValueError("`points` must have shape (N,3)")
	else:
		if x is None or y is None or z is None:
			raise ValueError("Provide either `points` or all of `x, y, z` arrays")
		x = np.asarray(x, dtype=dtype).reshape(-1)
		y = np.asarray(y, dtype=dtype).reshape(-1)
		z = np.asarray(z, dtype=dtype).reshape(-1)
		if not (len(x) == len(y) == len(z)):
			raise ValueError("`x, y, z` must have the same length")
		arr = np.stack([x, y, z], axis=1)

	# Remove invalid rows (NaN/Inf)
	mask = np.isfinite(arr).all(axis=1)
	arr = arr[mask]
	if arr.shape[0] < 2:
		raise ValueError("Need at least 2 valid points to form a trajectory")

	# Drop consecutive duplicates to avoid zero-length segments
	diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
	keep = np.concatenate([[True], diffs > 0])
	arr = arr[keep]
	if arr.shape[0] < 2:
		raise ValueError("Trajectory collapses to a single point after filtering")

	return arr


def _nice_step(span):
	"""Pick a reasonable axis step for a span."""
	if span <= 0:
		return 1.0
	raw = span / 5.0
	if raw == 0:
		return 1.0
	exp = int(np.floor(np.log10(raw)))
	base = raw / (10 ** exp)
	if base < 1.5:
		nice = 1.0
	elif base < 3.5:
		nice = 2.0
	elif base < 7.5:
		nice = 5.0
	else:
		nice = 10.0
	return nice * (10 ** exp)


def _rescale_points(P, target_span):
	"""Uniform scale so max span becomes target_span; center to origin."""
	mins = P.min(axis=0)
	maxs = P.max(axis=0)
	center = 0.5 * (mins + maxs)
	spans = maxs - mins
	max_span = float(np.max(spans))
	# Guard against fully-degenerate input (should be caught earlier)
	if max_span <= 0:
		return P * 0.0, 0.0, center
	scale = float(target_span) / max_span
	P2 = (P - center) * scale
	# Clamp axes with zero span to exactly zero after recentering to avoid FP noise
	zero_tol = float(CONFIG.get("ZERO_SPAN_TOL", 0.0))
	zero_axes = np.isclose(spans, 0.0, atol=zero_tol)
	if np.any(zero_axes):
		P2[:, zero_axes] = 0.0
	return P2, scale, center

def _rescale_points_per_axis(P, target_spans):
	"""Scale each axis to its target span; center to origin."""
	mins = P.min(axis=0)
	maxs = P.max(axis=0)
	center = 0.5 * (mins + maxs)
	spans = maxs - mins
	targets = np.asarray(list(target_spans), dtype=float).reshape(3)
	zero_tol = float(CONFIG.get("ZERO_SPAN_TOL", 0.0))
	# Avoid divide-by-zero by using 1.0 where span is zero; we'll clamp after
	denom = np.where(np.isclose(spans, 0.0, atol=zero_tol), 1.0, spans)
	scales = targets / denom
	P2 = (P - center) * scales  # broadcast per-axis
	# Clamp exactly-constant axes to zero after recentering
	zero_axes = np.isclose(spans, 0.0, atol=zero_tol)
	if np.any(zero_axes):
		P2[:, zero_axes] = 0.0
	return P2, scales, center


def _load_points_from_disk():
	"""Load (N,3) from trajectory.npy or trajectory.csv if present."""
	npy = "trajectory.npy"
	csv = "trajectory.csv"
	if os.path.exists(npy):
		arr = np.load(npy)
		return _standardize_points(points=arr)
	if os.path.exists(csv):
		try:
			arr = np.loadtxt(csv, delimiter=",")
		except Exception:
			# Try whitespace-delimited fallback
			arr = np.loadtxt(csv)
		return _standardize_points(points=arr)
	return None


def _demo_helix(n=1200, turns=3.5):
	"""Simple demo helix (N,3)."""
	t = np.linspace(0.0, 2.0 * np.pi * turns, n)
	R = 1.0
	x = R * np.cos(t)
	y = R * np.sin(t)
	z = 0.25 * t
	return _standardize_points(points=np.stack([x, y, z], axis=1))


class Animate(ThreeDScene):
	"""Render the path with axes and a moving sphere."""

	def construct(self):
		# Read configuration from module-level CONFIG
		MAX_POINTS = int(CONFIG.get("MAX_POINTS", 5000))
		DRAW_TIME = float(CONFIG.get("DRAW_TIME", 6.0))
		TARGET_SPAN = float(CONFIG.get("TARGET_SPAN", 8.0))
		# Resolve points
		P = None
		if POINTS is not None:
			P = _standardize_points(points=POINTS)
		else:
			P = _load_points_from_disk()
			if P is None:
				P = _demo_helix()

		# Rescale data to a canonical world span around origin
		scale_mode = str(CONFIG.get("SCALE_MODE", "uniform")).lower()
		per_axis = (scale_mode == "per-axis")
		no_scale = (scale_mode == "none")
		if per_axis:
			# Determine target spans per axis
			targets = CONFIG.get("TARGET_SPAN_PER_AXIS")
			if targets is None:
				targets = np.array([TARGET_SPAN, TARGET_SPAN, TARGET_SPAN], dtype=float)
			else:
				targets = np.asarray(targets, dtype=float).reshape(3)
			# Rescale independently per axis
			P_rescaled, data_scales, data_center = _rescale_points_per_axis(P, targets)
			# Axes ranges centered at origin with +/- target/2 per axis
			half = 0.5 * targets
			x_range = (-half[0], half[0], _nice_step(targets[0]))
			y_range = (-half[1], half[1], _nice_step(targets[1]))
			z_range = (-half[2], half[2], _nice_step(targets[2]))
		elif no_scale:
			# Keep raw data (recentering only) so spans remain physical; choose symmetric ranges
			mins = P.min(axis=0)
			maxs = P.max(axis=0)
			center = 0.5 * (mins + maxs)
			P_rescaled = P - center  # recentre only
			data_scales = np.array([1.0, 1.0, 1.0])
			data_center = center
			spans = maxs - mins
			# Choose nice ranges with small padding (10%)
			pad = 0.05
			mins2 = -0.5 * spans * (1 + pad)
			maxs2 =  0.5 * spans * (1 + pad)
			x_range = (mins2[0], maxs2[0], _nice_step(spans[0]))
			y_range = (mins2[1], maxs2[1], _nice_step(spans[1]))
			z_range = (mins2[2], maxs2[2], _nice_step(spans[2]))
		else:
			# Uniform rescaling to a single world span
			P_rescaled, data_scale, data_center = _rescale_points(P, TARGET_SPAN)
			# Fixed axes centered at origin with +/- TARGET_SPAN/2 per axis
			half = 0.5 * TARGET_SPAN
			x_range = (-half, half, _nice_step(2 * half))
			y_range = (-half, half, _nice_step(2 * half))
			z_range = (-half, half, _nice_step(2 * half))

		axes = ThreeDAxes(
			x_range=x_range,
			y_range=y_range,
			z_range=z_range,
			axis_config={"stroke_color": WHITE, "stroke_width": 2},
		)
		# Simple camera orientation; rely on rescaling/axes for visibility
		if CONFIG.get("FORCE_XY_VIEW", False):
			# Look straight down +z so the XY plane is screen plane
			self.set_camera_orientation(phi=0 * DEGREES, theta=0 * DEGREES)
		else:
			self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

		# Optional axis labels (to confirm orientation)
		if CONFIG.get("SHOW_AXIS_LABELS", False):
			labels = axes.get_axis_labels(Text("x"), Text("y"), Text("z"))
			self.add(labels)

		# Optional HUD: show scale factor used (fixed in frame)
		if CONFIG.get("SHOW_SCALE_INFO", True):
			units = CONFIG.get("UNITS_LABEL")
			if per_axis:
				label = (
					f"scale = (x:{data_scales[0]:.3e}, y:{data_scales[1]:.3e}, z:{data_scales[2]:.3e})"
				)
				if units:
					label += f" {units}/unit"
			elif no_scale:
				label = "scale = 1 (no scaling)"
			else:
				label = f"scale = {data_scale:.3e}"
				if units:
					label += f" {units}/unit"
			scale_text = Text(label, color=WHITE).scale(0.5)
			pos_key = str(CONFIG.get("SCALE_INFO_POSITION", "UL")).upper()
			corner = UL if pos_key == "UL" else UR
			scale_text.to_corner(corner)
			self.add_fixed_in_frame_mobjects(scale_text)
			self.play(FadeIn(scale_text, run_time=0.3))

		# Debug prints of ranges before and after rescale
		if CONFIG.get("DEBUG_PRINT", False):
			mins = P.min(axis=0)
			maxs = P.max(axis=0)
			spans = maxs - mins
			mins2 = P_rescaled.min(axis=0)
			maxs2 = P_rescaled.max(axis=0)
			spans2 = maxs2 - mins2
			print("Original ranges:")
			print(f"  x: [{mins[0]:.3e}, {maxs[0]:.3e}] span={spans[0]:.3e}")
			print(f"  y: [{mins[1]:.3e}, {maxs[1]:.3e}] span={spans[1]:.3e}")
			print(f"  z: [{mins[2]:.3e}, {maxs[2]:.3e}] span={spans[2]:.3e}")
			print("Rescaled ranges:")
			print(f"  x: [{mins2[0]:.3e}, {maxs2[0]:.3e}] span={spans2[0]:.3e}")
			print(f"  y: [{mins2[1]:.3e}, {maxs2[1]:.3e}] span={spans2[1]:.3e}")
			print(f"  z: [{mins2[2]:.3e}, {maxs2[2]:.3e}] span={spans2[2]:.3e}")

		# Build path as a polyline, with optional decimation, then move a sphere along it
		if len(P_rescaled) > MAX_POINTS:
			step = max(1, len(P_rescaled) // MAX_POINTS)
			P_rescaled = P_rescaled[::step]
		manim_pts = [axes.c2p(*xyz) for xyz in P_rescaled]
		path = VMobject()
		path.set_points_as_corners(manim_pts)
		path.set_fill(opacity=0)
		path.set_stroke(color=GOLD, width=5)

		# Particle marker with fixed world radius
		radius = 0.08
		particle = Sphere(radius=radius, color=BLUE)
		particle.move_to(manim_pts[0] if manim_pts else ORIGIN)

		# Add to scene and animate
		self.add(axes, particle)
		draw_time = DRAW_TIME  # seconds for full trace
		self.play(
			Create(path, run_time=draw_time, rate_func=linear),
			MoveAlongPath(particle, path, run_time=draw_time, rate_func=linear),
		)
		self.wait(0.5)


def _parse_cli():
	parser = argparse.ArgumentParser(description="Generate trajectory data and/or render Manim scene.")
	parser.add_argument("--scale-mode", choices=["uniform", "per-axis", "none"], help="Override CONFIG.SCALE_MODE")
	parser.add_argument("--target-span", type=float, help="Override CONFIG.TARGET_SPAN")
	parser.add_argument("--per-axis", type=float, nargs=3, metavar=("SX","SY","SZ"), help="Target spans per axis (activates per-axis mode)")
	parser.add_argument("--no-axis-labels", action="store_true", help="Hide axis labels")
	parser.add_argument("--no-scale-info", action="store_true", help="Hide scale HUD")
	parser.add_argument("--no-debug", action="store_true", help="Disable debug prints")
	parser.add_argument("--points-file", type=str, help="Explicit path to .npy or .csv points file")
	parser.add_argument("--write-demo", action="store_true", help="Just write demo helix to trajectory.csv and exit")
	return parser.parse_args()


def _cli_entry():
	args = _parse_cli()
	if args.scale_mode:
		CONFIG["SCALE_MODE"] = args.scale_mode
	if args.target_span is not None:
		CONFIG["TARGET_SPAN"] = args.target_span
	if args.per_axis:
		CONFIG["SCALE_MODE"] = "per-axis"
		CONFIG["TARGET_SPAN_PER_AXIS"] = tuple(args.per_axis)
	if args.no_axis_labels:
		CONFIG["SHOW_AXIS_LABELS"] = False
	if args.no_scale_info:
		CONFIG["SHOW_SCALE_INFO"] = False
	if args.no_debug:
		CONFIG["DEBUG_PRINT"] = False
	if args.points_file:
		# Load explicit file into POINTS and write standardized CSV for repeatability
		ext = os.path.splitext(args.points_file)[1].lower()
		if ext == ".npy":
			arr = np.load(args.points_file)
		else:
			arr = np.loadtxt(args.points_file, delimiter="," if "," in open(args.points_file, "r").readline() else None)
		set_points(points=arr)
	if args.write_demo:
		demo = _demo_helix()
		np.savetxt("trajectory.csv", demo, delimiter=",")
		print("Wrote demo helix to trajectory.csv")
		return
	print("Configuration:")
	print({k: CONFIG[k] for k in sorted(CONFIG)})
	print("Now run: manim -pql trajectory.py Animate  (or higher quality)" )


if __name__ == "__main__":
	_cli_entry()