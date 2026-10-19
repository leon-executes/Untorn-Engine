"""Untorn Engine — native Windows desktop, local Windows photographic upscaling. One engine, one source.
Source-derived detail is evaluated on a native 4x working grid.
Fractional sizes use adaptive DCCI enlargement or pixel-centred reduction.
"""

from __future__ import annotations
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, concurrent.futures, hashlib, io, json, math, re
import shutil, struct, subprocess, sys, threading, time, traceback, uuid, warnings
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms, ImageOps

# Numerical cache contract: source-detail restoration; prior tile recipes are incompatible.
VERSION = "1.3.0"
APP_VERSION = "2.3.0"
AUTHOR = "Parham (Leon) Faraji"
COPYRIGHT = "Copyright 2026 Parham (Leon) Faraji (@leon-executes)"
# Original portions: PolyForm-Noncommercial-1.0.0. Third-party notices apply.
METHOD = "Untorn Engine"
CACHE_METHOD = "Untorn Definition Gen 1.0"  # Stable identity for existing numerical caches.
ROOT = (
    Path(sys.executable).parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
ASSETS = Path(getattr(sys, "_MEIPASS", ROOT))
BIN = ASSETS / "bin"
FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_PIXELS = 40_000_000
MAX_FILE = 400_000_000
MAX_OUTPUT = 160_000_000
Image.MAX_IMAGE_PIXELS = MAX_OUTPUT
warnings.simplefilter("error", Image.DecompressionBombWarning)
LUMA = np.array([0.2126, 0.7152, 0.0722])
STUDIO_TILE = 128
STUDIO_HALO = 16


# 01 — Input integrity and fitted continuous colour field.
def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temp, "wb") as f:
            f.write(data.encode("utf-8") if isinstance(data, str) else data)
            f.flush()
            os.fsync(f.fileno())
        replace_file(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_json(path, data):
    atomic(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except (OSError, ValueError):
        return default


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode(path):
    path = Path(path)
    if path.suffix.lower() not in FORMATS or path.stat().st_size > MAX_FILE:
        raise ValueError("Use PNG, JPEG, WebP, BMP or TIFF, up to 400 MB.")
    notes = []
    with Image.open(path) as raw:
        if raw.width * raw.height > MAX_PIXELS:
            raise ValueError(
                "Image exceeds the 40-megapixel processing limit. Original remains untouched."
            )
        if getattr(raw, "n_frames", 1) > 1:
            notes.append("First frame/page processed; original multi-frame file retained.")
        png_depth = None
        if raw.format == "PNG":
            with open(path, "rb") as header:
                head = header.read(25)
                png_depth = head[24] if len(head) == 25 else None
        tiff_depth = getattr(raw, "tag_v2", {}).get(258, 8)
        if not isinstance(tiff_depth, (tuple, list)):
            tiff_depth = [tiff_depth]
        if raw.mode in ("I", "F") or "16" in raw.mode or png_depth == 16 or max(tiff_depth) > 8:
            raise ValueError(
                "16-bit/HDR input needs an explicit tone conversion. Convert to 8-bit sRGB first; no silent reduction is applied."
            )
        raw.load()
        im = ImageOps.exif_transpose(raw)
        if raw.info.get("icc_profile"):
            alpha = im.getchannel("A") if "A" in im.getbands() else None
            try:
                profile = ImageCms.ImageCmsProfile(io.BytesIO(raw.info["icc_profile"]))
                rgb = ImageCms.profileToProfile(
                    im.convert("RGB") if im.mode != "CMYK" else im,
                    profile,
                    ImageCms.createProfile("sRGB"),
                    outputMode="RGB",
                )
                im = rgb.convert("RGBA")
                if alpha is not None:
                    im.putalpha(alpha)
                notes.append(
                    "Embedded colour profile converted to sRGB; original profile retained in source file."
                )
            except Exception as e:
                raise ValueError(
                    "Cannot safely convert the embedded colour profile: " + str(e)
                ) from e
        elif im.mode == "CMYK":
            raise ValueError(
                "CMYK without an embedded profile is ambiguous. Export an sRGB image first."
            )
        else:
            im = im.convert("RGBA")
    return im, notes


def fnum(v):
    return ("%.6f" % float(v)).rstrip("0").rstrip(".") or "0"


def colour(rgba):
    v = np.clip(np.rint(rgba), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(v[:3]), fnum(v[3] / 255)


def svg_document(w, h, defs, body):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}" color-interpolation="sRGB">'
        "<title>Untorn Engine restoration</title><desc>Local vector restoration. "
        "Source and verification report retained separately. No embedded raster.</desc>"
        + ("<defs>" + "".join(defs) + "</defs>" if defs else "")
        + "".join(body)
        + "</svg>"
    )


def field_sample(a, points):
    """Bilinear samples in image coordinates (sample centres are x+.5,y+.5)."""
    h, w = a.shape[:2]
    q = np.asarray(points, dtype=np.float32) - 0.5
    x = np.clip(q[..., 0], 0, w - 1)
    y = np.clip(q[..., 1], 0, h - 1)
    ix = x.astype(np.int32)
    iy = y.astype(np.int32)
    fx = x - ix
    fy = y - iy
    if a.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    return ((1 - fx) * a[iy, ix] + fx * a[iy, np.minimum(ix + 1, w - 1)]) * (1 - fy) + (
        a[np.minimum(iy + 1, h - 1), ix] * (1 - fx)
        + a[np.minimum(iy + 1, h - 1), np.minimum(ix + 1, w - 1)] * fx
    ) * fy


def peak_blur(a):
    # Small, fixed analysis kernel; exported colours still come from the source.
    b = np.pad(a, ((1, 1), (0, 0)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    a = (b[:-2] + 2 * b[1:-1] + b[2:]) * 0.25
    b = np.pad(a, ((0, 0), (1, 1)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    return (b[:, :-2] + 2 * b[:, 1:-1] + b[:, 2:]) * 0.25


def contrast_destinations(pixels, detail=0.65, blur_steps=1):
    """Find ridges of RGB contrast magnitude and intensity extrema separately.

    Destinations are subpixel coordinates, not enlarged pixel-cell centres.
    RGB structure tensors retain isoluminant boundaries. Hessian extrema detect
    thin light/dark ridges; their neighbouring edge pair is suppressed locally.
    """
    h, w = pixels.shape[:2]
    rgb = pixels[..., :3].astype(np.float32)
    if min(h, w) < 5:
        return [], {"edge_points": 0, "ridge_points": 0, "threshold": detail}
    a = rgb
    for _ in range(blur_steps):
        a = peak_blur(a)
    gy, gx = np.gradient(a, axis=(0, 1))
    jx = peak_blur(np.mean(gx * gx, axis=2))
    jy = peak_blur(np.mean(gy * gy, axis=2))
    jxy = peak_blur(np.mean(gx * gy, axis=2))
    theta = 0.5 * np.arctan2(2 * jxy, jx - jy)
    normal = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    disc = np.hypot(jx - jy, 2 * jxy)
    coherence = disc / (jx + jy + 1e-6)
    projected = gx * normal[..., 0, None] + gy * normal[..., 1, None]
    dominant = np.take_along_axis(
        projected, np.argmax(np.abs(projected), axis=2)[..., None], axis=2
    )[..., 0]
    normal *= np.where(dominant < 0, -1, 1)[..., None]
    strength = np.sqrt(np.mean(projected**2, axis=2))
    yy, xx = np.mgrid[:h, :w]
    grid = np.stack([xx + 0.5, yy + 0.5], axis=-1).astype(np.float32)
    before = field_sample(strength, grid - normal)
    after = field_sample(strength, grid + normal)
    curvature = before - 2 * strength + after
    edge = (
        (strength >= detail)
        & (strength >= before)
        & (strength > after)
        & (curvature < -0.015 * strength)
        & (coherence > 0.32)
    )
    shift = np.clip(0.5 * (before - after) / np.minimum(curvature, -1e-6), -0.5, 0.5)
    edge_q = grid + shift[..., None] * normal

    # Dark and bright ridges: a local extremum along the strongest Hessian axis.
    lum = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    ly, lx = np.gradient(lum)
    hxy, hxx = np.gradient(lx)
    hyy, hyx = np.gradient(ly)
    hxy = (hxy + hyx) * 0.5
    rad = np.hypot(hxx - hyy, 2 * hxy)
    lp = (hxx + hyy + rad) * 0.5
    lm = (hxx + hyy - rad) * 0.5
    use_minus = np.abs(lm) > np.abs(lp)
    lam = np.where(use_minus, lm, lp)
    other = np.where(use_minus, lp, lm)
    angle = 0.5 * np.arctan2(2 * hxy, hxx - hyy) + np.where(use_minus, np.pi / 2, 0)
    rn = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    deriv = lx * rn[..., 0] + ly * rn[..., 1]
    delta = -deriv / np.where(np.abs(lam) > 1e-5, lam, 1e-5)
    rq = grid + np.clip(delta, -0.6, 0.6)[..., None] * rn
    rmag = np.abs(lam)
    rb = field_sample(rmag, grid - rn)
    ra = field_sample(rmag, grid + rn)
    ridge = (
        (np.abs(delta) <= 0.6)
        & (rmag >= detail)
        & (rmag >= rb)
        & (rmag > ra)
        & (rmag > 1.8 * np.abs(other))
    )
    center = field_sample(lum, rq)
    left = field_sample(lum, rq - 2 * rn)
    right = field_sample(lum, rq + 2 * rn)
    amp = np.minimum(np.abs(center - left), np.abs(center - right))
    ridge &= ((center - left) * (center - right) > 0) & (amp > detail * 1.5)
    margin = 3 + blur_steps
    ridge[:margin] = ridge[-margin:] = False
    ridge[:, :margin] = ridge[:, -margin:] = False
    # Suppress the two edge flanks only where a verified ridge destination exists.
    suppress = np.zeros((h, w), bool)
    ry, rx = np.nonzero(ridge)
    for sign in [-1, 0, 1]:
        p = np.rint(rq[ry, rx] + sign * rn[ry, rx] * 1.2 - 0.5).astype(int)
        suppress[np.clip(p[:, 1], 0, h - 1), np.clip(p[:, 0], 0, w - 1)] = True
    edge &= ~suppress
    edge[:margin] = edge[-margin:] = False
    edge[:, :margin] = edge[:, -margin:] = False
    records = []
    for kind, mask, coords, normals, mag in [
        ("boundary", edge, edge_q, normal, strength),
        ("ridge", ridge, rq, rn, amp),
    ]:
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            n = normals[y, x]
            q = coords[y, x]
            confidence = float(
                np.clip(
                    (
                        coherence[y, x]
                        if kind == "boundary"
                        else 1 - abs(other[y, x]) / (abs(lam[y, x]) + 1e-6)
                    ),
                    0.2,
                    1,
                )
            )
            records.append(
                {
                    "kind": kind,
                    "cell": [x, y],
                    "point": q.tolist(),
                    "normal": n.tolist(),
                    "contrast": float(mag[y, x]),
                    "confidence": confidence,
                }
            )
    return records, {
        "edge_points": int(edge.sum()),
        "ridge_points": int(ridge.sum()),
        "threshold": detail,
    }


def saturated_peak_colours(colours, neighbours, boost=1.0):
    """Hue-preserving gamut endpoints, with signed local contrast polarity.

    Chromatic maxima reach HSV S=V=1; minima reach S=1 and 35% of their
    original V. Near-neutral samples (<=4/255 chroma) reach white or black.
    This is explicit enhancement, not an estimate of lost ground truth.
    """
    c = np.asarray(colours, dtype=np.float32) / 255
    ref = np.asarray(neighbours, dtype=np.float32) / 255
    delta = c - ref
    lum = delta @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dominant = np.take_along_axis(delta, np.argmax(np.abs(delta), axis=-1)[..., None], axis=-1)[
        ..., 0
    ]
    positive = np.where(np.abs(lum) > 1 / 255, lum > 0, dominant >= 0)
    hi = c.max(axis=-1)
    lo = c.min(axis=-1)
    chroma = hi - lo
    hue = (c - lo[..., None]) / np.maximum(chroma[..., None], 1e-8)
    value = np.where(positive, 1, np.maximum(hi * 0.35, 1 / 255))
    target = hue * value[..., None]
    target = np.where(
        (chroma <= 4 / 255)[..., None], positive[..., None].astype(np.float32), target
    )
    return np.clip((c + (target - c) * boost) * 255, 0, 255)


def profile_rgba(a):
    a = np.uint8(np.clip(np.rint(a), 0, 255))
    return np.dstack([a, np.full(a.shape[:2], 255, dtype="uint8")]) if a.shape[-1] == 3 else a


def profile_response(d, width, kind):
    if kind == "ridge":
        return np.exp(-0.5 * (d / width) ** 2)
    return 1 / (1 + np.exp(np.clip(-d / width, -50, 50)))


def sample_peak_field(
    a, model, scale, width_factor=0.6, boost=0.0, prior=0.22, fine=0.0, core=0.0, return_guard=False
):
    """Blend analytic local fields with smooth compact support, never draw ink.

    Overlapping models use a partition of unity. The result remains defined
    between source sample locations. Narrowing and endpoint boost are explicit
    restoration priors. A finite SVG compilation is evaluated separately.
    """
    h, w = a.shape[:2]
    oh, ow = h * scale, w * scale
    baseline = np.asarray(Image.fromarray(a).resize((ow, oh), Image.Resampling.BICUBIC))[
        ..., :3
    ].astype(float)
    numerator = baseline * prior
    denom = np.full((oh, ow), prior)
    if return_guard:
        polarity = np.zeros((oh, ow))
        coverage = np.zeros((oh, ow))
    for p in model["profiles"]:
        if p["confidence"] < 0.015:
            continue
        q = np.array(p["q"])
        n = np.array(p["normal"])
        t = np.array([-n[1], n[0]])
        c = np.array(p["coeff"])
        radius = 3.5
        x0 = max(0, int((q[0] - radius) * scale))
        x1 = min(ow, int((q[0] + radius) * scale) + 1)
        y0 = max(0, int((q[1] - radius) * scale))
        y1 = min(oh, int((q[1] + radius) * scale) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = (xx + 0.5) / scale - q[0]
        dy = (yy + 0.5) / scale - q[1]
        d = dx * n[0] + dy * n[1]
        s = dx * t[0] + dy * t[1]
        radius2 = (d / 2.7) ** 2 + (s / 2.0) ** 2
        weight = np.maximum(1 - radius2, 0) ** 3 * p["confidence"]
        low = np.clip(c[0], 0, 255)
        high = np.clip(c[0] + c[2], 0, 255)
        if boost:
            l, hc = low.copy(), high.copy()
            low = saturated_peak_colours(l, hc, boost)
            high = saturated_peak_colours(hc, l, boost)
        width = max(0.025, p["width"] * width_factor)
        f = (
            fine_profile_response(d, width, p["kind"], fine, core)
            if fine or core
            else profile_response(d, width, p["kind"])
        )
        field = low + (high - low) * f[..., None] + c[1] * s[..., None]
        numerator[y0:y1, x0:x1] += np.clip(field, 0, 255) * weight[..., None]
        denom[y0:y1, x0:x1] += weight
        if return_guard and p["kind"] == "ridge":
            sign = np.sign(float((high - low) @ np.array([0.2126, 0.7152, 0.0722])))
            support = weight * profile_response(d, max(width, 0.35) * 1.8, "ridge")
            polarity[y0:y1, x0:x1] += sign * support
            coverage[y0:y1, x0:x1] += support
    result = profile_rgba(numerator / denom[..., None])
    return (result, polarity / (prior + coverage)) if return_guard else result


def fine_profile_response(d, width, kind, strength=0.7, core=0.0):
    if kind != "ridge":
        return profile_response(d, width, kind)
    z = d / max(width, 0.025)
    if core:
        # Compact monotone coordinate warp: steeper only inside 0.6 widths.
        # C2 join leaves the gentle outer shoulder exactly unchanged.
        taper = np.maximum(1 - (z / 0.6) ** 2, 0) ** 3
        z = z * (1 + core * taper)
    # Smooth cusp: exact centre unchanged, every nonzero shoulder reduced.
    # Multiplying by the Gaussian envelope prevents a long tail joining neighbours.
    return np.exp(-0.5 * z * z - strength * (np.sqrt(z * z + 0.02**2) - 0.02))


def fit_field_batch(a, points, batch=96):
    """The original two-pass profile objective, evaluated in bounded batches."""
    rgb = a[..., :3].astype(float)
    h, w = rgb.shape[:2]
    models = []
    points = [p for p in points if 3 <= p["cell"][0] < w - 3 and 3 <= p["cell"][1] < h - 3]
    offsets = np.array([(x, y) for y in [-1 / 3, 0, 1 / 3] for x in [-1 / 3, 0, 1 / 3]])
    grid = np.array([(x + 0.5, y + 0.5) for y in range(-3, 4) for x in range(-3, 4)])
    ss, ww = np.meshgrid([-0.6, -0.3, 0, 0.3, 0.6], [0.06, 0.12, 0.24, 0.42, 0.7, 1.05])
    ss = ss.ravel()
    ww = ww.ravel()

    def solve(q, n, t, cells, shift, width, ridges):
        coords = cells[:, None, :] + grid
        d = np.einsum("bni,bi->bn", coords - q[:, None, :], n)
        along = np.einsum("bni,bi->bn", coords - q[:, None, :], t)
        weight = np.exp(-0.5 * ((d / 1.7) ** 2 + (along / 1.7) ** 2))
        ix = (coords[..., 0] - 0.5).astype(int)
        iy = (coords[..., 1] - 0.5).astype(int)
        target = rgb[iy, ix]
        f = np.zeros((len(q), shift.shape[1], 49))
        for off in offsets:
            dist = d[:, None, :] + (n @ off)[:, None, None] - shift[:, :, None]
            z = dist / width[:, :, None]
            f += (
                np.where(
                    ridges[:, None, None],
                    np.exp(-0.5 * z * z),
                    1 / (1 + np.exp(np.clip(-z, -50, 50))),
                )
                / 9
            )
        # Sufficient statistics avoid constructing and multiplying millions of
        # repeated [1, tangent, response] design rows. Same regularized solve.
        wf = f * weight[:, None, :]
        wa = weight * along
        gram = np.empty((len(q), shift.shape[1], 3, 3))
        rhs = np.empty((len(q), shift.shape[1], 3, 3))
        gram[:, :, 0, 0] = weight.sum(1)[:, None]
        gram[:, :, 0, 1] = gram[:, :, 1, 0] = wa.sum(1)[:, None]
        gram[:, :, 1, 1] = (wa * along).sum(1)[:, None]
        gram[:, :, 0, 2] = gram[:, :, 2, 0] = wf.sum(2)
        gram[:, :, 1, 2] = gram[:, :, 2, 1] = (wf * along[:, None, :]).sum(2)
        gram[:, :, 2, 2] = (wf * f).sum(2)
        rhs[:, :, 0] = np.einsum("bn,bnc->bc", weight, target)[:, None]
        rhs[:, :, 1] = np.einsum("bn,bnc->bc", wa, target)[:, None]
        rhs[:, :, 2] = np.einsum("bmn,bnc->bmc", wf, target)
        coeff = np.linalg.solve(gram + np.eye(3) * 1e-7, rhs)
        # Evaluate residuals directly to retain the original near-tie behaviour.
        pred = (
            coeff[:, :, 0, None, :]
            + along[:, None, :, None] * coeff[:, :, 1, None, :]
            + f[:, :, :, None] * coeff[:, :, 2, None, :]
        )
        error = np.sqrt(
            np.sum((pred - target[:, None]) ** 2 * weight[:, None, :, None], axis=(2, 3))
            / (3 * weight.sum(axis=1)[:, None])
        )
        lo = coeff[:, :, 0]
        hi = lo + coeff[:, :, 2]
        excursion = np.maximum(-np.minimum(lo, hi), 0) + np.maximum(np.maximum(lo, hi) - 255, 0)
        return coeff, error, error + 0.15 * np.mean(excursion, axis=2)

    for start in range(0, len(points), batch):
        pp = points[start : start + batch]
        q = np.array([p["point"] for p in pp])
        n = np.array([p["normal"] for p in pp])
        t = np.stack([-n[:, 1], n[:, 0]], 1)
        cells = np.array([p["cell"] for p in pp])
        ridges = np.array([p["kind"] == "ridge" for p in pp])
        c, e, o = solve(
            q,
            n,
            t,
            cells,
            np.broadcast_to(ss, (len(pp), len(ss))),
            np.broadcast_to(ww, (len(pp), len(ww))),
            ridges,
        )
        best = o.argmin(axis=1)
        sr = (
            ss[best, None, None]
            + np.linspace(-0.15, 0.15, 7)[None, None, :]
            + np.zeros((len(pp), 3, 1))
        ).reshape(len(pp), 21)
        wr = np.broadcast_to(
            np.clip(ww[best, None, None] * np.array([0.75, 1, 1.25])[None, :, None], 0.025, 1.3),
            (len(pp), 3, 7),
        ).reshape(len(pp), 21)
        c, e, o = solve(q, n, t, cells, sr, wr, ridges)
        best = o.argmin(axis=1)
        for k, p in enumerate(pp):
            j = best[k]
            coef = c[k, j]
            err = e[k, j]
            contrast = float(np.sqrt(np.mean(coef[2] ** 2)))
            models.append(
                {
                    "kind": p["kind"],
                    "q": (q[k] + n[k] * sr[k, j]).tolist(),
                    "normal": n[k].tolist(),
                    "width": float(wr[k, j]),
                    "coeff": coef.tolist(),
                    "confidence": float(np.clip(1 - err / max(contrast * 0.3, 2), 0, 1))
                    * p["confidence"],
                    "fit_rmse": float(err),
                    "contrast": contrast,
                }
            )
    return {"profiles": models}


def studio_tile(a, box, scale):
    h, w = a.shape[:2]
    x0, y0, x1, y1 = box
    lx = max(0, x0 - STUDIO_HALO)
    ly = max(0, y0 - STUDIO_HALO)
    rx = min(w, x1 + STUDIO_HALO)
    ry = min(h, y1 + STUDIO_HALO)
    crop = np.asarray(a[ly:ry, lx:rx])
    points, _ = contrast_destinations(crop, 2.0, 1)
    # Profiles influence at most 3.5 pixels around their fitted centres.
    points = [
        p
        for p in points
        if x0 - lx - 5 <= p["cell"][0] < x1 - lx + 5 and y0 - ly - 5 <= p["cell"][1] < y1 - ly + 5
    ]
    model = fit_field_batch(crop, points)
    result = sample_peak_field(crop, model, scale, 1.0)
    return result[
        (y0 - ly) * scale : (y1 - ly) * scale, (x0 - lx) * scale : (x1 - lx) * scale
    ], len(model["profiles"])


def write_stream_png(path, rows, width, height):
    """RGB PNG, row-streamed and atomic; no second full-size output allocation."""
    import struct, zlib

    path = Path(path)
    temp = path.with_suffix(".png.part")

    def chunk(f, kind, data):
        f.write(
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    with open(temp, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        chunk(f, b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        chunk(f, b"sRGB", b"\x00")
        z = zlib.compressobj(3)
        pending = bytearray()
        for row in rows:
            row = np.ascontiguousarray(row[..., :3], dtype=np.uint8)
            filtered = row.copy()
            filtered[1:] = row[1:] - row[:-1]
            pending.extend(z.compress(b"\x01" + filtered.tobytes()))
            if len(pending) > 1024 * 1024:
                chunk(f, b"IDAT", bytes(pending))
                pending.clear()
        pending.extend(z.flush())
        chunk(f, b"IDAT", bytes(pending))
        chunk(f, b"IEND", b"")
        f.flush()
        os.fsync(f.fileno())
    replace_file(temp, path)


def compact_surface_svg(pixels):
    """Exact same bilinear field as smooth_svg; local IDs and deduplicated stops.

    Short gradient spans retain the original Qt colour-table precision guard.
    Whole constant fields bypass all gradients/masks. No lossy stop pruning.
    """
    h, w = pixels.shape[:2]
    if np.all(pixels == pixels[0, 0]):
        return svg_document(
            w, h, [], [f'<path d="M0 0H{w}V{h}H0Z" fill="{colour(pixels[0, 0])[0]}"/>']
        )
    defs = [
        '<linearGradient id="f" x2="0" y2="1"><stop offset=".25" stop-color="#fff" stop-opacity="0"/><stop offset=".5" stop-color="#fff"/></linearGradient>',
        '<mask id="m" maskContentUnits="objectBoundingBox" x="0" y="0" width="1" height="1"><path d="M0 0H1V1H0Z" fill="url(#f)"/></mask>',
    ]
    body = []
    pool = {}
    for y, row in enumerate(pixels):
        ypos = 0 if y == 0 else y - 1.5
        rh = h if y == 0 else 4
        paint = []
        for x in range(0, w, 8):
            lo = max(0, x - 1)
            hi = min(w, x + 10)
            span = hi - lo
            v = row[lo:hi, :3]
            keep = np.ones(span, bool)
            if span > 2:
                d = np.diff(v.astype(np.int16), axis=0)
                keep[1:-1] = np.any(d[:-1] != d[1:], axis=1)
            ix = np.flatnonzero(keep)
            key = (lo, hi, ix.tobytes(), v[ix].tobytes())
            gid = pool.get(key)
            if np.all(v == v[0]):
                fill = colour([*v[0], 255])[0]
            else:
                if gid is None:
                    gid = "a" + str(len(pool))
                    pool[key] = gid
                    stops = "".join(
                        f'<stop offset="{(i + 0.5) / span:.9f}" stop-color="#{v[i, 0]:02x}{v[i, 1]:02x}{v[i, 2]:02x}"/>'
                        for i in ix
                    )
                    defs.append(
                        f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{lo}" x2="{hi}">{stops}</linearGradient>'
                    )
                # Keep the original rectangle and halo geometry exactly.
                fill = f"url(#{gid})"
            paint.append(
                f'<path d="M{x} {fnum(ypos)}h{min(9, w - x)}v{rh}h-{min(9, w - x)}Z" fill="{fill}"/>'
            )
        body.append(('<g mask="url(#m)">' if y else "<g>") + "".join(paint) + "</g>")
    return svg_document(w, h, defs, body)


# 02 — DCCI structure, Local Laplacian middle band and compact grain.
def arr(im):
    return np.asarray(im.convert("RGB"))


def image(a):
    return Image.fromarray(np.uint8(np.clip(np.rint(a), 0, 255)))


def blur(a, r):
    # Floating point separable Gaussian avoids repeatedly quantizing residuals.
    x = np.arange(-int(np.ceil(3 * r)), int(np.ceil(3 * r)) + 1)
    k = np.exp(-x * x / (2 * r * r))
    k /= k.sum()
    out = np.asarray(a, dtype=np.float32)
    for axis in (0, 1):
        n = len(k) // 2
        pads = [(0, 0)] * out.ndim
        pads[axis] = (n, n)
        p = np.pad(out, pads, mode="reflect")
        v = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            v += p[tuple(sl)] * w
        out = v
    return out


def dcci_twice(a):
    """DCC.m k=5,T=1.15, per RGB channel in [0,1], edge-extended boundaries.

    Four source pixels of context make the otherwise-unfilled reference border
    usable. All interior stencils, direction tests and cubic weights are retained.
    """
    a = np.asarray(a, dtype=np.float64)
    h, w, c = a.shape
    src = np.pad(a, ((4, 4), (4, 4), (0, 0)), mode="edge")
    m, n = 2 * src.shape[0] - 1, 2 * src.shape[1] - 1
    grid = np.zeros((m, n, c), dtype=np.float64)
    grid[::2, ::2] = src
    for phase, ys, xs in [
        (0, np.arange(3, m - 3, 2), np.arange(3, n - 3, 2)),
        (1, np.arange(4, m - 4, 2), np.arange(3, n - 3, 2)),
        (1, np.arange(3, m - 3, 2), np.arange(4, n - 4, 2)),
    ]:
        yy, xx = ys[:, None], xs[None, :]

        def at(y, x):
            return grid[yy + y, xx + x]

        if phase == 0:
            d1 = sum(abs(at(y, x) - at(y - 2, x + 2)) for y in [-1, 1, 3] for x in [-3, -1, 1])
            d2 = sum(abs(at(y, x) - at(y + 2, x + 2)) for y in [-3, -1, 1] for x in [-3, -1, 1])
            p1 = (-at(3, -3) + 9 * at(1, -1) + 9 * at(-1, 1) - at(-3, 3)) / 16
            p2 = (-at(-3, -3) + 9 * at(-1, -1) + 9 * at(1, 1) - at(3, 3)) / 16
        else:
            d1 = sum(abs(at(y, -1) - at(y, 1)) for y in [-2, 0, 2]) + sum(
                abs(at(y, x) - at(y, x + 2)) for y in [-1, 1] for x in [-2, 0]
            )
            d2 = sum(abs(at(-1, x) - at(1, x)) for x in [-2, 0, 2]) + sum(
                abs(at(y, x) - at(y + 2, x)) for x in [-1, 1] for y in [-2, 0]
            )
            p1 = (-at(0, -3) + 9 * at(0, -1) + 9 * at(0, 1) - at(0, 3)) / 16
            p2 = (-at(-3, 0) + 9 * at(-1, 0) + 9 * at(1, 0) - at(3, 0)) / 16
        w1 = 1 / (1 + d1**5)
        w2 = 1 / (1 + d2**5)
        grid[yy, xx] = np.where(
            (1 + d1) / (1 + d2) > 1.15,
            p2,
            np.where((1 + d2) / (1 + d1) > 1.15, p1, (w1 * p1 + w2 * p2) / (w1 + w2)),
        )
    return grid[8 : 8 + 2 * h - 1, 8 : 8 + 2 * w - 1].copy()


def bands(a, scale):
    small = blur(a, max(0.6, 0.35 * scale))
    broad = blur(a, 1.2 * scale)
    return broad, small - broad, a - small


def gate(d, a, db, ab, scale):
    dy, dx = np.gradient(blur(d, 0.6 * scale))
    ay, ax = np.gradient(blur(a, 0.6 * scale))
    radius = max(0.6, 0.5 * scale)
    de = blur(dx * dx + dy * dy, radius)
    ae = blur(ax * ax + ay * ay, radius)
    agreement = np.clip(blur(dx * ax + dy * ay, radius) / np.sqrt(de * ae + 1e-8), 0, 1)
    support = np.sqrt(de) / (np.sqrt(de) + 2 / scale)
    return blur(
        (0.35 + 0.65 * agreement) * (0.5 + 0.5 * support) * np.exp(-abs(db - ab) / 20), radius
    )


def structure_mask(a, scale):
    """Coherent local gradients indicate outlines/ridges rather than grain."""
    dy, dx = np.gradient(blur(a, max(0.6, 0.35 * scale)))
    radius = max(1, 0.75 * scale)
    xx, yy, xy = [blur(v, radius) for v in [dx * dx, dy * dy, dx * dy]]
    energy = xx + yy
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (energy + 1e-8)
    # Only confidently directional structure should suppress grain. A raw
    # coherence weight would erase most texture in already enlarged photos.
    coherence = np.clip((coherence - 0.65) / 0.30, 0, 1)
    coherence = coherence * coherence * (3 - 2 * coherence)
    strength = np.sqrt(energy) / (np.sqrt(energy) + 1 / scale)
    mask = np.clip(coherence * strength, 0, 1)
    # Protect adjacent shoulders too, rather than outlining an excluded core.
    r = max(1, int(np.ceil(0.4 * scale)))
    p = np.pad(mask, r, mode="edge")
    return np.maximum.reduce(
        [
            p[y : y + a.shape[0], x : x + a.shape[1]]
            for y in range(2 * r + 1)
            for x in range(2 * r + 1)
        ]
    )


def hessian(a, sigma):
    smooth = blur(a, sigma)
    dy, dx = np.gradient(smooth)
    yy, yx = np.gradient(dy)
    xy, xx = np.gradient(dx)
    xy = (xy + yx) * 0.5
    trace = xx + yy
    disc = np.sqrt((xx - yy) ** 2 + 4 * xy * xy)
    l1 = (trace + disc) * 0.5
    l2 = (trace - disc) * 0.5
    compact = np.where(
        l1 * l2 > 0, np.minimum(abs(l1), abs(l2)) / (np.maximum(abs(l1), abs(l2)) + 1e-8), 0
    )
    return abs(trace) * sigma * sigma, compact


def speckles(a, scale):
    factor = scale / 3
    sizes = [max(0.45, s * factor) for s in [0.55, 0.85, 1.4, 2.3, 3.6]]
    measures = [hessian(a, s) for s in sizes]
    response = np.stack([m[0] for m in measures])
    winner = response.argmax(axis=0)
    small = np.maximum(response[0], response[1])
    shape = np.where(response[0] >= response[1], measures[0][1], measures[1][1])
    # Hard size and shape veto, with a soft weight within the accepted sites.
    accepted = (winner < 2) & (shape > 0.30) & (small > 0.12)
    margin = max(3, int(np.ceil(scale)))
    accepted[:margin] = False
    accepted[-margin:] = False
    accepted[:, :margin] = False
    accepted[:, -margin:] = False
    mask = accepted * np.clip((shape - 0.30) / 0.35, 0, 1) * np.clip((small - 0.12) / 0.5, 0, 1)
    # No spreading outside the size/shape accepted region: long contours stay cut.
    detail = a - blur(a, max(0.7, 0.38 * scale))
    return (
        detail * mask,
        mask,
        {
            "accepted_fraction": float(accepted.mean()),
            "mean_mask": float(mask.mean()),
            "sigmas": sizes,
        },
    )


def resize_float(a, size):
    return np.asarray(
        Image.fromarray(np.asarray(a, np.float32)).resize(size, Image.Resampling.BILINEAR),
        dtype=np.float64,
    )


def conv5(a):
    out = np.asarray(a, dtype=np.float64)
    for axis in (0, 1):
        pads = [(0, 0)] * out.ndim
        pads[axis] = (2, 2)
        p = np.pad(out, pads, mode="constant")
        v = np.zeros_like(out)
        for i, w in enumerate([0.05, 0.25, 0.4, 0.25, 0.05]):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            v += w * p[tuple(sl)]
        out = v
    return out


def down(a):
    return (conv5(a) / conv5(np.ones_like(a)))[::2, ::2]


def up(a, shape):
    z = np.zeros(shape)
    w = np.zeros(shape)
    z[::2, ::2] = a
    w[::2, ::2] = 1
    return conv5(z) / conv5(w)


def gaussian(a, levels=6):
    g = [a]
    while len(g) < levels and min(g[-1].shape) > 2:
        g.append(down(g[-1]))
    return g


def laplacian(a, levels):
    g = gaussian(a, levels)
    return [g[i] - up(g[i + 1], g[i].shape) for i in range(len(g) - 1)] + [g[-1]]


def collapse(l):
    out = l[-1]
    for b in l[-2::-1]:
        out = up(out, b.shape) + b
    return out


def remap(a, ref, sigma=0.075, alpha=0.65):
    d = abs(a - ref)
    normalized = d / sigma
    # Same noise-aware power remapping as the supplied Matlab reference.
    t = np.clip((d - 0.01) / 0.01, 0, 1)
    t = t * t * (t - 2) ** 2
    mapped = t * normalized**alpha + (1 - t) * normalized
    return ref + np.sign(a - ref) * np.where(d < sigma, sigma * mapped, d)


def local_laplacian(a, bins=64, alpha=0.65, sigma=0.075):
    """Reference pyramid/remapping with sampled reference intensities (approximation)."""
    a = np.asarray(a, dtype=np.float64) / 255
    g = gaussian(a)
    out = [np.zeros_like(x) for x in g]
    for ref in np.linspace(0, 1, bins):
        l = laplacian(remap(a, ref, sigma, alpha), len(g))
        for k in range(len(g) - 1):
            out[k] += l[k] * np.maximum(0, 1 - abs(g[k] - ref) * (bins - 1))
    out[-1] = g[-1]
    return collapse(out) * 255


def periodic(a):
    h, w = a.shape
    v = np.zeros_like(a)
    v[0, :] = a[-1, :] - a[0, :]
    v[-1, :] = -v[0, :]
    v[:, 0] += a[:, -1] - a[:, 0]
    v[:, -1] -= a[:, -1] - a[:, 0]
    denom = (
        2 * np.cos(2 * np.pi * np.arange(h) / h)[:, None]
        + 2 * np.cos(2 * np.pi * np.arange(w) / w)[None, :]
        - 4
    )
    denom[0, 0] = 1
    f = np.fft.fft2(v) / denom
    f[0, 0] = 0
    return a - np.fft.ifft2(f).real


def write_raw(path, a):
    with path.open("wb") as f:
        f.write(struct.pack("<II", a.shape[1], a.shape[0]))
        f.write(np.asarray(a, dtype="<f4").tobytes())


def read_raw(path):
    with path.open("rb") as f:
        w, h = struct.unpack("<II", f.read(8))
        a = np.frombuffer(f.read(), "<f4").reshape(h, w).copy()
    assert np.isfinite(a).all()
    return a


def luminance_add(rgb, delta):
    b = np.asarray(rgb, float)
    delta = np.clip(delta, -b.min(-1), 255 - b.max(-1))
    return image(b + delta[..., None])


# 03 — Portable application, jobs and interface.


def replace_file(source, destination):
    """Windows readers/antivirus can briefly hold a destination without delete sharing."""
    for attempt in range(16):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 15:
                raise
            time.sleep(min(0.02 * 2**attempt, 0.4))


_progress_lock = threading.RLock()


def status_file(folder, stage, progress=0, **extra):
    path = Path(folder) / "progress.json"
    with _progress_lock:
        previous = read_json(path, {})
        progress = max(float(previous.get("progress", 0)), min(1.0, max(0.0, float(progress))))
        save_json(path, dict(stage=stage, progress=progress, updated=time.time(), **extra))


FITTED_MAX_PIXELS = 512 * 512


def use_fitted_for(size):
    return int(size[0]) * int(size[1]) <= FITTED_MAX_PIXELS


class RestorationProgress:
    """Per-tile milestones aggregated independently of worker completion order."""

    def __init__(self, folder, count, fitted):
        self.folder = folder
        self.values = [0.0] * count
        self.lock = threading.RLock()
        self.stages = (
            {
                "Restoring edges": 0.01,
                "Fitting colour and shading": 0.07,
                "Resolving fine contours": 0.60,
                "Refining middle detail": 0.66,
                "Restoring surface detail": 0.80,
                "Blending Jinc4 detail": 0.64,
                "X-Ray clarity and highlights": 0.90,
                "Matching source detail": 0.86,
                "Protecting source texture": 0.85,
                "Refining contour sampling": 0.91,
            }
            if fitted
            else {
                "Restoring edges": 0.01,
                "Resolving fine contours": 0.13,
                "Refining middle detail": 0.26,
                "Restoring surface detail": 0.62,
                "Blending Jinc4 detail": 0.24,
                "X-Ray clarity and highlights": 0.90,
                "Matching source detail": 0.86,
                "Protecting source texture": 0.85,
                "Refining contour sampling": 0.91,
            }
        )

    def update(self, index, stage, value=None):
        with self.lock:
            self.values[index] = max(
                self.values[index], self.stages.get(stage, 0.0) if value is None else value
            )
            done = sum(v >= 1 for v in self.values)
            label = stage if len(self.values) == 1 else "Restoring image"
            status_file(
                self.folder,
                label,
                0.94 * sum(self.values) / len(self.values),
                tile=done,
                tiles=len(self.values),
                active_stage=stage,
            )


def dcci_four(low):
    pad = 8
    w, h = low.size
    grid = np.pad(arr(low).astype(float) / 255, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    for _ in range(2):
        grid = dcci_twice(grid)
    box = (
        4 * (pad - 0.5) + 0.5,
        4 * (pad - 0.5) + 0.5,
        4 * (pad + w - 0.5) + 0.5,
        4 * (pad + h - 0.5) + 0.5,
    )
    return image(
        np.stack(
            [
                np.asarray(
                    Image.fromarray((grid[..., c] * 255).astype("float32")).resize(
                        (w * 4, h * 4), Image.Resampling.LANCZOS, box=box
                    )
                )
                for c in range(3)
            ],
            -1,
        )
    )


def fitted_four(low):
    a = np.asarray(low.convert("RGBA"))
    h, w = a.shape[:2]
    out = np.empty((h * 4, w * 4, 4), np.uint8)
    for y in range(0, h, 128):
        for x in range(0, w, 128):
            r, b = min(w, x + 128), min(h, y + 128)
            out[y * 4 : b * 4, x * 4 : r * 4] = studio_tile(a, (x, y, r, b), 4)[0]
    return Image.fromarray(out).convert("RGB")


def restore_grain(base, donors, texture_gate, unsharp_amount=2.0, strength=1.0):
    """Equal source-detail ensemble: fine-band unsharp, fixed compact support."""
    gains = (2.5, 2.5, 1.75, 1.75) if len(donors) == 4 else (2.5, 1.75, 1.75)
    if len(donors) not in (3, 4):
        raise ValueError("Expected three or four detail contributors")
    by = base @ LUMA
    own = by - blur(by, 1.52)
    total = np.zeros_like(by)
    coverage = np.zeros_like(by)
    for donor, gain in zip(donors, gains):
        _, mask, _ = speckles(donor, 4)
        fine = donor - blur(donor, 1.52)
        sharp = fine + unsharp_amount * (fine - blur(fine, 0.8))
        total += (gain * sharp - own) * mask * texture_gate
        coverage = np.maximum(coverage, mask)
    grain = strength * 12 * np.tanh((6 * (total / len(donors))) / 12)
    return grain, dict(
        donor_gains=list(gains),
        weights=[1 / len(donors)] * len(donors),
        unsharp_amount=unsharp_amount,
        strength=strength,
        unsharp_radius=0.8,
        grain_gain=6 * strength,
        grain_soft_cap=12 * strength,
        accepted_fraction=float(np.mean(coverage * texture_gate > 0)),
    )


# 02c — X-Ray. Shared Local Laplacian field and bounded grain refinement.
def xray_grain_cells(a):
    """View a native 4x field as cells aligned to source pixels."""
    h, w = a.shape[:2]
    if h % 4 or w % 4:
        raise ValueError("X-Ray grain requires a source-aligned 4x field")
    return a.reshape(h // 4, 4, w // 4, 4, *a.shape[2:])


def xray_grain_project(delta, support):
    """Remove cell DC without spreading outside accepted grain sites."""
    delta = np.asarray(delta, float)
    weights = xray_grain_cells(support)
    cells = xray_grain_cells(delta.copy())
    mass = weights.sum((1, 3), keepdims=True)
    sums = cells.sum((1, 3), keepdims=True)
    cells -= weights[..., None] * np.divide(
        sums, mass[..., None], out=np.zeros_like(sums), where=mass[..., None] > 1e-12
    )
    return cells.reshape(delta.shape)


def xray_grain_contract():
    """Stable public description of the grain refinement parameters."""
    return dict(
        scope="Local Laplacian grain only",
        laplacian_references=80,
        shared_laplacian=True,
        unsharp_amount=2.6,
        unsharp_radius=0.8,
        contrast_amplitude=0.90,
        whites_peak_lift=0.04,
        whites_scope="Positive fine peaks in upper source tones; projected to zero cell mean",
        chroma_strength=0.12,
        chroma_source="DCCI fine chroma; muted weighted and luminance-free",
        veil_strength=0.18,
        minimum_transmission=0.82,
        veil_scope="Fine grain only; source-guided neutral-veil estimate, not depth recovery",
        support="Existing Laplacian size/shape mask and texture gate; no spreading",
        projection="4x4 source-cell zero mean before quantization",
        rgb_delta_limit=3.0,
        gamut="Common scale per source cell",
        overshoot_guard="3x3 pre-correction envelope; headroom min(1, 2% local range)",
        global_colour_or_tone=False,
    )


def xray_gcs_grain_guard(sharp, reference):
    """Attenuate unsupported two-axis oscillation without filtering the image.

    Opposing samples test native-grid periods of 2, 4 and 8 pixels. Only
    oscillation along both axes is penalized; source-agreeing detail is retained.
    A continuous overlapping confidence field avoids introducing block borders.
    """
    energy = blur(sharp * sharp, 1.2)
    periodicity = np.zeros_like(sharp)
    h, w = sharp.shape
    for lag in (1, 2, 4):
        p = np.pad(sharp, lag, mode="edge")
        horizontal = 0.5 * (p[lag : lag + h, :w] + p[lag : lag + h, 2 * lag : 2 * lag + w])
        vertical = 0.5 * (p[:h, lag : lag + w] + p[2 * lag : 2 * lag + h, lag : lag + w])
        # Both directions must alternate; a single coherent line is not a grid.
        opposing = np.minimum(-sharp * horizontal, -sharp * vertical)
        score = np.clip(blur(opposing, 1.2) / (energy + 0.25), 0, 1)
        periodicity = np.maximum(periodicity, score)
    agreement = np.clip(
        2 * sharp * reference / (sharp * sharp + reference * reference + 1e-8), 0, 1
    )
    disagreement = np.clip(blur(1 - agreement, 1.2), 0, 1)
    weight = 0.85 * periodicity * disagreement
    return sharp + weight * (reference - sharp)


def xray_restore_grain(source_rgb, base, donors, texture_gate):
    """Shared-80 ensemble plus a bounded, grain-only RGB correction.

    GCS fine detail receives a source-guided periodic-pattern guard. The Laplacian donor's
    residual is refined; source chroma supplies its colour-bearing fine band.
    The correction is applied after existing tone composition so gamut bounds
    preserve the source-cell mean and exact accepted support.
    """
    if len(donors) not in (3, 4):
        raise ValueError("Expected three or four detail contributors")
    gains = (2.5, 2.5, 1.75, 1.75) if len(donors) == 4 else (2.5, 1.75, 1.75)
    own = base @ LUMA
    own = own - blur(own, 1.52)
    total = np.zeros_like(own)
    coverage = np.zeros_like(own)
    ll_index = len(donors) - 2
    for index, (donor, gain) in enumerate(zip(donors, gains)):
        _, mask, _ = speckles(donor, 4)
        fine = donor - blur(donor, 1.52)
        core = fine - blur(fine, 0.8)
        sharp = fine + 2.0 * core
        if index == 0:
            reference_sharp = sharp
        elif index == len(donors) - 1:
            sharp = xray_gcs_grain_guard(sharp, reference_sharp)
        support = mask * texture_gate
        term = (gain * sharp - own) * support
        total += term
        coverage = np.maximum(coverage, mask)
        if index == ll_index:
            ll_fine, ll_core, ll_support, ll_gain, ll_term = fine, core, support, gain, term
    grain = 12 * np.tanh((6 * total / len(donors)) / 12)
    smooth = blur(source_rgb, 4.8)
    saturation = (smooth.max(-1) - smooth.min(-1)) / np.maximum(smooth.max(-1), 1.0)
    veil = np.clip(smooth.min(-1) / 255, 0, 1) * (1 - saturation) ** 2
    transmission = 1 - 0.18 * veil
    sharp = (ll_fine + 2.6 * ll_core) / transmission
    sharp = xray_grain_whites(sharp, smooth @ LUMA)
    refined_term = (ll_gain * 0.90 * sharp - own) * ll_support
    refined_grain = 12 * np.tanh((6 * (total - ll_term + refined_term) / len(donors)) / 12)
    luma = xray_grain_project(np.repeat((refined_grain - grain)[..., None], 3, axis=2), ll_support)
    y = source_rgb @ LUMA
    chroma = source_rgb - y[..., None]
    fine_chroma = chroma - blur(chroma, 1.52)
    fine_chroma = fine_chroma + 2.6 * (fine_chroma - blur(fine_chroma, 0.8))
    fine_chroma -= (fine_chroma @ LUMA)[..., None]
    colour = (
        fine_chroma
        * (
            0.12
            * 0.90
            * ll_gain
            * 6
            / len(donors)
            * (1 - saturation) ** 2
            / transmission
            * ll_support
        )[..., None]
    )
    colour = xray_grain_project(colour, ll_support)
    delta = luma + colour
    cells = xray_grain_cells(delta)
    peak = np.max(np.abs(cells), axis=(1, 3, 4), keepdims=True)
    delta = (cells * np.minimum(1, 3 / np.maximum(peak, 1e-12))).reshape(delta.shape)
    meta = dict(
        donor_gains=list(gains),
        weights=[1 / len(donors)] * len(donors),
        unsharp_amount=2.0,
        strength=1.0,
        unsharp_radius=0.8,
        grain_gain=6.0,
        grain_soft_cap=12.0,
        accepted_fraction=float(np.mean(coverage * texture_gate > 0)),
        laplacian_refinement=xray_grain_contract(),
        gcs_grain_guard="Source-guided two-axis periodicity; periods 2/4/8; maximum blend 0.85",
    )
    return grain, delta, meta


def xray_local_envelope(rgb):
    """Three-by-three extrema with at most one code value of detail headroom."""
    a = np.asarray(rgb, float)
    p = np.pad(a, ((0, 0), (1, 1), (0, 0)), mode="edge")
    low = np.minimum(np.minimum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
    high = np.maximum(np.maximum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
    p = np.pad(low, ((1, 1), (0, 0), (0, 0)), mode="edge")
    low = np.minimum(np.minimum(p[:-2], p[1:-1]), p[2:])
    p = np.pad(high, ((1, 1), (0, 0), (0, 0)), mode="edge")
    high = np.maximum(np.maximum(p[:-2], p[1:-1]), p[2:])
    allowance = np.minimum(1.0, 0.02 * (high - low))
    return np.maximum(0, low - allowance), np.minimum(255, high + allowance)


def xray_guard_gcs(donor, envelope):
    """Light suppression of contour excursions; keep in-envelope detail exact."""
    low, high = envelope
    return donor + 0.25 * (np.clip(donor, low, high) - donor)


def xray_apply_grain(rgb, delta):
    """Apply only the supported grain correction, preserving cell means in gamut."""
    base = np.clip(np.asarray(rgb, float), 0, 255)
    low, high = xray_local_envelope(base)
    room = np.where(
        delta > 0,
        (high - base) / np.maximum(delta, 1e-12),
        (low - base) / np.minimum(delta, -1e-12),
    )
    factor = np.clip(np.min(xray_grain_cells(room), axis=(1, 3, 4), keepdims=True), 0, 1)
    bounded = (xray_grain_cells(delta) * factor).reshape(delta.shape)
    return base + bounded


XRAY_RECIPE = "xray-v9"
XRAY_PLUS_RECIPE = "xray-shared80-spatial-blend-v8"
XRAY_FINE_STRENGTH = 1.0
XRAY_LAPLACIAN_BINS = 80
XRAY_DETAIL_STRENGTH = 0.60
XRAY_LIGHTING_STRENGTH = 2.75
XRAY_JINC_CONTRAST = 0.50


def xray_luma(rgb, delta):
    rgb = np.asarray(rgb, float)
    delta = np.clip(delta, -rgb.min(-1), 255 - rgb.max(-1))
    return rgb + delta[..., None]


def xray_middle(a):
    """Smooth middle-band isolation on the native 4x grid, not a hard frequency cut."""
    return blur(a, 1.4) - blur(a, 4.8)


def xray_jinc(x):
    z = np.asarray(x, dtype=np.float64) * np.pi
    t = np.ones_like(z)
    v = t.copy()
    for k in range(1, 60):
        t *= -(z * z / 4) / (k * (k + 1))
        v += t
        if np.max(np.abs(t)) < 1e-15:
            break
    return v


def xray_jinc4(low):
    """Circular Jinc4 Sharp, pixel-centred phases and 90% local anti-ringing."""
    a = arr(low).astype(float)
    h, w = a.shape[:2]
    out = np.empty((h * 4, w * 4, 3), float)
    radius = 4.24106286379607
    sharp = 0.8845120932605005
    for py in range(4):
        cy = (py + 0.5) / 4 - 0.5
        for px in range(4):
            cx = (px + 0.5) / 4 - 0.5
            value = np.zeros_like(a)
            norm = 0.0
            lo = np.full_like(a, np.inf)
            hi = -lo.copy()
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    r = math.hypot(dx - cx, dy - cy)
                    if r > radius:
                        continue
                    weight = float(
                        xray_jinc(r / sharp) * xray_jinc(r / radius * 1.2196698912665045)
                    )
                    v = a[
                        np.clip(np.arange(h) + dy, 0, h - 1)[:, None],
                        np.clip(np.arange(w) + dx, 0, w - 1)[None, :],
                    ]
                    value += weight * v
                    norm += weight
                    if r <= 1.25:
                        lo = np.minimum(lo, v)
                        hi = np.maximum(hi, v)
            value /= norm
            value = value + 0.9 * (np.clip(value, lo, hi) - value)
            allowance = np.minimum(1.0, 0.02 * (hi - lo))
            out[py::4, px::4] = np.clip(value, lo - allowance, hi + allowance)
    return np.clip(out, 0, 255)


def xray_mix(dcci, gcs, jinc):
    return dcci + XRAY_DETAIL_STRENGTH * xray_middle(
        0.12 * (gcs - dcci) + 0.06 * XRAY_JINC_CONTRAST * (jinc - dcci)
    )


def xray_directional_deblur(y):
    """Short-kernel inverse correction using local edge directions, not a camera trajectory."""
    gy, gx = np.gradient(blur(y, 1.0))
    xx = blur(gx * gx, 2.0)
    yy = blur(gy * gy, 2.0)
    xy = blur(gx * gy, 2.0)
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
    angle = 0.5 * np.arctan2(2 * xy, xx - yy)
    total = np.zeros_like(y)
    norm = np.zeros_like(y)
    h, w = y.shape
    for dy, dx, theta in [(0, 1, 0), (1, 0, np.pi / 2), (1, 1, np.pi / 4), (1, -1, -np.pi / 4)]:
        weight = np.maximum(0, np.cos(angle - theta) ** 2 - 0.5) ** 2

        def forward(a):
            p = np.pad(a, 1, mode="edge")
            return 0.6 * a + 0.2 * (
                p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
                + p[1 - dy : 1 - dy + h, 1 - dx : 1 - dx + w]
            )

        estimate = y.copy()
        for _ in range(3):
            residual = y - forward(estimate)
            estimate += 0.65 * forward(residual)
            estimate = np.clip(estimate, y - 8, y + 8)
        total += weight * (estimate - y)
        norm += weight
    correction = (
        6
        * np.tanh((total / np.maximum(norm, 1e-8)) / 6)
        * coherence
        * np.clip((xx + yy) / 12, 0, 1)
    )
    return xray_middle(correction)


def xray_highlights(y):
    """Monotone Hermite shoulder: lower highlights, retain a modest white lift."""
    x = np.array([0.0, 0.25, 0.50, 0.70, 0.85, 0.94, 0.985, 1.0])
    v = np.array([0.0, 0.25, 0.50, 0.65, 0.78, 0.915, 0.996, 1.0])
    d = np.diff(v) / np.diff(x)
    m = np.empty_like(x)
    m[0] = d[0]
    m[-1] = d[-1]
    m[1:-1] = 2 * d[:-1] * d[1:] / (d[:-1] + d[1:])
    m[2] = 0.75  # Preserve the preceding release's lower-midpoint tangent.
    z = np.clip(np.asarray(y, float) / 255, 0, 1)
    i = np.minimum(len(x) - 2, np.searchsorted(x, z, side="right") - 1)
    h = x[i + 1] - x[i]
    t = (z - x[i]) / h
    return 255 * (
        (2 * t**3 - 3 * t * t + 1) * v[i]
        + (t**3 - 2 * t * t + t) * h * m[i]
        + (-2 * t**3 + 3 * t * t) * v[i + 1]
        + (t**3 - t * t) * h * m[i + 1]
    )


def xray_contrast(y, strength=1.0):
    # Gentle midtone separation; fixed black and white endpoints.
    return y + 0.05 * strength * (y - 127.5) * np.sin(np.pi * np.clip(y, 0, 255) / 255) ** 2


def xray_shadow_visibility(y):
    # Small smooth lift in dark tones without raising the black endpoint.
    t = np.clip(y / 150.0, 0, 1)
    return y + 2.0 * 16 * t * t * (1 - t) * (1 - t)


def xray_grain_whites(fine, source_luma):
    """Slight white-peak emphasis within the supported fine band only."""
    t = np.clip((source_luma / 255 - 0.5) / 0.4, 0, 1)
    weight = t * t * (3 - 2 * t)
    return fine + 0.04 * weight * np.maximum(fine, 0)


def xray_whites(y, strength=0.28):
    """Gentle upper-tone lift, with a smooth onset and preserved white endpoint."""
    t = np.clip((y / 255 - 0.5) / 0.4, 0, 1)
    weight = t * t * (3 - 2 * t)
    return y + strength * weight * (255 - y)


def xray_donor_tone(rgb, contrast_strength=1.0, whites_strength=0.28):
    """Floating-point lighting treatment for the two small donor contributions only."""
    from PIL import ImageFilter

    y = rgb @ LUMA
    # Fixed local parameters avoid tile-wise histogram/exposure jumps.
    dark = np.asarray(
        Image.fromarray(np.uint8(np.clip(rgb.min(-1), 0, 255))).filter(ImageFilter.MinFilter(9)),
        float,
    )
    transmission = np.maximum(0.89, 1 - 0.11 * blur(dark, 3.0) / 255)
    clear = 255 + (y - 255) / transmission
    rgb = xray_luma(rgb, 10 * np.tanh((clear - y) / 10))
    y = rgb @ LUMA
    tone = xray_whites(
        xray_shadow_visibility(xray_contrast(xray_highlights(y), contrast_strength)),
        strength=whites_strength,
    )
    return xray_luma(rgb, tone - y)


def xray_layer_lighting(gcs, jinc):
    # No tone operation on DCCI, fitted colour, Local Laplacian or the final image.
    correction = XRAY_LIGHTING_STRENGTH * (
        0.12 * (xray_donor_tone(gcs) - gcs)
        + 0.06 * (xray_donor_tone(jinc, contrast_strength=0.25, whites_strength=0.31) - jinc)
    )
    # Smooth the correction, never the image: avoid modulating Max's fine grain.
    return blur(correction, 2.0)


def xray_detail_patch(
    low, folder, notify=lambda x: None, acceleration=None, use_fitted=True, self_examples=False
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    timings = {}
    phase = [None, time.perf_counter()]
    pixels = arr(low)
    if np.all(pixels == pixels[0, 0]):
        base = np.asarray(low.resize((low.width * 4, low.height * 4)), float)
        lighting = xray_layer_lighting(base, base)
        if self_examples:
            base = xray_dcci_vibrance(base, base)
        base = base + lighting
        out = image(base)
        return out, dict(
            recipe=XRAY_PLUS_RECIPE if self_examples else XRAY_RECIPE,
            flat=True,
            fitted_enabled=use_fitted,
            timings={},
        )
    if min(low.size) < 2:
        padded = Image.fromarray(
            np.pad(
                pixels,
                ((0, max(0, 2 - low.height)), (0, max(0, 2 - low.width)), (0, 0)),
                mode="edge",
            )
        )
        out, meta = xray_detail_patch(
            padded, folder, notify, acceleration, use_fitted, self_examples
        )
        return out.crop((0, 0, low.width * 4, low.height * 4)), meta

    def stage(name):
        now = time.perf_counter()
        if phase[0] is not None:
            timings[phase[0]] = now - phase[1]
        phase[:] = [name, now]
        if name:
            notify(name)

    stage("Restoring edges")
    d = arr(dcci_four(low)).astype(float)
    fitted = None
    if use_fitted:
        stage("Fitting colour and shading")
        fitted = arr(fitted_four(low)).astype(float)
    stage("Resolving fine contours")
    src = folder / "source.bmp"
    low.save(src)

    def contours(name, psf, passes):
        dst = folder / (name + ".bmp")
        with (folder / (name + ".log")).open("w") as log:
            subprocess.run(
                [
                    str(BIN / "sinterp.exe"),
                    "-x",
                    "4",
                    "-p",
                    psf,
                    "-r",
                    str(passes),
                    str(src),
                    str(dst),
                ],
                check=True,
                stdout=log,
                stderr=log,
                timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        with Image.open(dst) as im:
            result = arr(im.convert("RGB")).astype(float)
        if result.shape != d.shape:
            raise ValueError("Contour renderer returned incorrect dimensions")
        return result

    # Lightly guard the middle donor; retain grain detection on the ordinary GCS field.
    c = contours("contours", ".35", 3)
    strong = contours("xray-contours", ".60", 5)
    contour_envelope = xray_local_envelope(d)
    strong = xray_guard_gcs(strong, contour_envelope)
    stage("Blending Jinc4 detail")
    j = xray_jinc4(low)
    middle = xray_mix(d, strong, j) - d
    lighting = xray_layer_lighting(strong, j)
    del strong, j
    base = np.clip(d - blur(d, 4.8) + blur(fitted, 4.8), 0, 255) if fitted is not None else d.copy()
    stage("Refining middle detail")
    dl, cl = d @ LUMA, c @ LUMA
    ll = (
        acceleration.laplacian(dl, XRAY_LAPLACIAN_BINS, folder)
        if acceleration
        else local_laplacian(dl, bins=XRAY_LAPLACIAN_BINS)
    )
    # Reuse the middle field; refinement is confined to the grain contribution.
    grain_ll = ll
    db, dm, _ = bands(dl, 4)
    _, lm, _ = bands(ll, 4)
    structure = 0.65 * gate(dl, dl, db, db, 4) * (lm - dm)
    stage("Restoring surface detail")
    yl = arr(low).astype(float) @ LUMA
    gy, gx = np.gradient(blur(yl, 0.65))
    edge = np.hypot(gx, gy)
    texture_gate = (
        resize_float(np.exp(-((edge / 9) ** 2)), (d.shape[1], d.shape[0]))
        * (1 - np.maximum(structure_mask(dl, 4), structure_mask(cl, 4))) ** 2
    )
    texture_gate[:4] = 0
    texture_gate[-4:] = 0
    texture_gate[:, :4] = 0
    texture_gate[:, -4:] = 0
    grain, grain_delta, grain_meta = xray_restore_grain(
        d,
        base,
        (dl, fitted @ LUMA, grain_ll, cl) if fitted is not None else (dl, grain_ll, cl),
        texture_gate,
    )
    stage("Protecting source texture")
    grain = spatial_protect_grain(grain, dl)
    self_meta = None
    if self_examples:
        stage("Matching source detail")
        extra_grain, self_meta = xray_self_grain(low, texture_gate)
        grain = grain + extra_grain
        base = xray_dcci_vibrance(base, d)
    rgb = xray_luma(base, 14 * np.tanh(structure / 14) + grain)
    stage("X-Ray clarity and highlights")
    # Retain positive donor lift; reduce negative donor tone to avoid dimming.
    lighting = np.where(lighting < 0, 0.5 * lighting, lighting)
    rgb = rgb + middle + lighting
    rgb = xray_apply_grain(rgb, grain_delta)
    carrier = xray_luma(base, 14 * np.tanh(structure / 14)) + middle + lighting
    stage("Refining contour sampling")
    rgb, spatial_meta = spatial_cleanup(rgb, carrier, "xray")
    output = image(rgb)
    stage(None)
    return output, dict(
        recipe=XRAY_PLUS_RECIPE if self_examples else XRAY_RECIPE,
        fitted_enabled=use_fitted,
        grain=grain_meta,
        spatial=spatial_meta,
        self_examples=self_meta,
        timings=timings,
        xray=dict(
            blend=dict(dcci=0.82, gcs=0.12, jinc4=0.06),
            blend_scope="middle_band",
            fine_strength=XRAY_FINE_STRENGTH,
            gcs_psf=0.60,
            gcs_refinement=5,
            laplacian_bins=XRAY_LAPLACIAN_BINS,
            grain_recipe="Shared 80-reference field with grain-only sharpening, contrast, chroma and veil refinement",
            grain_laplacian_bins=80,
            grain_refinement=xray_grain_contract(),
            grain_gcs_psf=0.35,
            grain_gcs_refinement=3,
            extra_donor_unsharp=False,
            jinc_anti_ringing=0.9,
            jinc_overshoot_limit="min(1 code value, 2% local source range)",
            gcs_overshoot_suppression=0.25,
            gcs_overshoot_scope="middle donor only; grain detection unchanged",
            gcs_grain_guard="Source-guided periodicity suppression; unchanged support and weights",
            contrast_gain=0.05,
            jinc_detail_contrast=XRAY_JINC_CONTRAST,
            jinc_tone_contrast=0.25,
            jinc_whites_lift=0.31,
            lighting_smoothing=2.0,
            shadow_lift=2.0,
            detail_strength=XRAY_DETAIL_STRENGTH,
            lighting_strength=XRAY_LIGHTING_STRENGTH,
            lighting_scope="GCS 12% and Jinc4 6% only",
            global_tone=False,
            deblur="No additional directional deblur",
            highlights="Donor-only shoulder and Whites lift; cannot recover clipped information",
        ),
    )


# 02d — Bounded internal examples. Original implementation, not a paper port.
# Inspired by Glasner et al. (ICCV 2009) and Huang et al. (CVPR 2015).
def xray_self_sample(a, y, x):
    """Bilinear sampling with explicit edge extension and float32 intermediates."""
    y = np.clip(y, 0, a.shape[0] - 1)
    x = np.clip(x, 0, a.shape[1] - 1)
    iy = y.astype(np.int32)
    ix = x.astype(np.int32)
    fy = y - iy
    fx = x - ix
    jy = np.minimum(iy + 1, a.shape[0] - 1)
    jx = np.minimum(ix + 1, a.shape[1] - 1)
    return (
        (1 - fy) * ((1 - fx) * a[iy, ix] + fx * a[iy, jx])
        + fy * ((1 - fx) * a[jy, ix] + fx * a[jy, jx])
    ).astype(np.float32)


def xray_self_details(low):
    """Two bounded cross-scale residuals; source-only donors, no learned weights.

    Translation branch searches nine local examples; transformed branch shares
    that search and adds +/-20 degree rotations. 3x3 query patches are matched
    to integrated 12x12 source patches. Only accepted detail is transferred.
    This intentionally omits global nearest-neighbour search, plane estimation,
    and the papers' iterative optimizers. No claim of precise strand recovery.
    """
    y = (arr(low).astype(np.float32) @ LUMA).astype(np.float32)
    h, w = y.shape
    outputs = [np.zeros((h * 4, w * 4), np.float32) for _ in range(2)]
    if min(h, w) < 3 or float(np.ptp(y)) < 1:
        return outputs, dict(queries=0, accepted=[0, 0], candidate_count=27)
    # Descriptor integration on the source grid, shared by both searches.
    smooth = blur(y, 1.1)
    gy, gx = np.gradient(blur(y, 0.65))
    xx = blur(gx * gx, 1)
    yy = blur(gy * gy, 1)
    xy = blur(gx * gy, 1)
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-5)
    cy, cx = np.mgrid[0:h:2, 0:w:2]
    cy = cy.ravel()
    cx = cx.ravel()
    py, px = np.mgrid[-1:2, -1:2]
    py = py.ravel()
    px = px.ravel()
    query = xray_self_sample(y, cy[:, None] + py, cx[:, None] + px)
    centered = query - query.mean(1, keepdims=True)
    energy = np.mean(centered**2, 1)
    active = (energy > 9) & (coherence[cy, cx] > 0.20)
    cy = cy[active]
    cx = cx[active]
    query = query[active]
    centered = centered[active]
    energy = energy[active]
    norm = np.zeros((h * 4, w * 4), np.float32)
    accepted = [0, 0]
    win = np.outer(np.hanning(14)[1:-1], np.hanning(14)[1:-1]).astype(np.float32)
    sy, sx = np.mgrid[:12, :12]
    sy = sy.astype(np.float32) - 5.5
    sx = sx.astype(np.float32) - 5.5
    # Small batches cap temporary arrays independently of image dimensions.
    for start in range(0, len(cy), 128):
        by = cy[start : start + 128]
        bx = cx[start : start + 128]
        q = centered[start : start + 128]
        qe = energy[start : start + 128]
        n = len(by)
        best = np.full(n, np.inf)
        chosen = np.zeros((n, 5), np.float32)
        plain = None
        for angle in (0.0, -20.0, 20.0):
            c = np.float32(math.cos(math.radians(angle)))
            s = np.float32(math.sin(math.radians(angle)))
            for dy in (-4, 0, 4):
                for dx in (-4, 0, 4):
                    desc = xray_self_sample(
                        smooth,
                        by[:, None] + dy + 4 * (c * py + s * px),
                        bx[:, None] + dx + 4 * (c * px - s * py),
                    )
                    desc -= desc.mean(1, keepdims=True)
                    de = np.mean(desc**2, 1)
                    gain = np.clip(np.mean(q * desc, 1) / (de + 1e-5), 0.5, 2.0)
                    error = np.mean((q - desc * gain[:, None]) ** 2, 1) / (qe + 4)
                    # Reject out-of-image donors instead of accepting repeated borders.
                    valid = (by + dy >= 8) & (by + dy < h - 8) & (bx + dx >= 8) & (bx + dx < w - 8)
                    error = np.where(valid, error, np.inf)
                    better = error < best
                    best[better] = error[better]
                    chosen[better] = np.stack(
                        [np.full(n, dy), np.full(n, dx), np.full(n, c), np.full(n, s), gain], 1
                    )[better]
            if angle == 0:
                plain = (best.copy(), chosen.copy())
        for branch, (cost, params) in enumerate((plain, (best, chosen))):
            confidence = np.clip((0.45 - cost) / 0.35, 0, 1) * np.clip(
                (coherence[by, bx] - 0.20) / 0.55, 0, 1
            )
            keep = confidence > 0.01
            accepted[branch] += int(keep.sum())
            dy, dx, c, s, gain = params.T
            high = xray_self_sample(
                y,
                by[:, None, None]
                + dy[:, None, None]
                + c[:, None, None] * sy
                + s[:, None, None] * sx,
                bx[:, None, None]
                + dx[:, None, None]
                + c[:, None, None] * sx
                - s[:, None, None] * sy,
            )
            # Fit the donor's coarse samples, then transfer its subpixel residual.
            coarse = high.reshape(n, 3, 4, 3, 4).mean((2, 4))
            ty, tx = np.mgrid[:12, :12]
            ty = (ty + 0.5) / 4 - 0.5
            tx = (tx + 0.5) / 4 - 0.5
            residual = np.empty_like(high)
            for k in range(n):
                residual[k] = high[k] - xray_self_sample(coarse[k], ty, tx)
            residual *= gain[:, None, None]
            residual = 10 * np.tanh(residual / 10)
            residual *= confidence[:, None, None]
            for k in np.flatnonzero(keep):
                top = int(by[k]) * 4 - 4
                left = int(bx[k]) * 4 - 4
                t = max(0, top)
                l = max(0, left)
                b = min(h * 4, top + 12)
                r = min(w * 4, left + 12)
                weight = win[t - top : b - top, l - left : r - left]
                outputs[branch][t:b, l:r] += (
                    residual[k, t - top : b - top, l - left : r - left] * weight
                )
        # All active query windows contribute normalization, including rejected
        # matches: a doubtful patch fades to zero rather than amplifying a neighbour.
        for ay, ax in zip(by, bx):
            top = int(ay) * 4 - 4
            left = int(ax) * 4 - 4
            t = max(0, top)
            l = max(0, left)
            b = min(h * 4, top + 12)
            r = min(w * 4, left + 12)
            norm[t:b, l:r] += win[t - top : b - top, l - left : r - left]
    for out in outputs:
        out /= np.maximum(norm, 1e-8)
        # One exact box-model consistency projection. The added field has zero
        # average in every source-pixel cell; this is not a true-camera guarantee.
        cells = out.reshape(h, 4, w, 4)
        cells -= cells.mean((1, 3), keepdims=True)
        np.clip(out, -10, 10, out=out)
        cells -= cells.mean((1, 3), keepdims=True)
    return outputs, dict(
        queries=len(cy),
        accepted=accepted,
        candidate_count=27,
        search_radius=4,
        rotation_degrees=[-20, 0, 20],
        batch=128,
        source_only=True,
        projection="4x4 box-cell zero mean",
    )


def xray_self_vibrance(rgb):
    """Muted-weighted DCCI chroma lift, bounded to two percent."""
    y = rgb @ LUMA
    chroma = rgb - y[..., None]
    saturation = (rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1.0)
    return chroma * (0.02 * (1 - saturation) ** 2)[..., None]


def xray_dcci_vibrance(base, dcci):
    """DCCI-derived colour correction, with shared RGB gamut protection."""
    base = np.clip(np.asarray(base, float), 0, 255)
    colour = xray_self_vibrance(dcci)
    room = np.where(
        colour > 0, (255 - base) / np.maximum(colour, 1e-8), -base / np.minimum(colour, -1e-8)
    )
    return base + colour * np.minimum(1, room.min(-1))[..., None]


def xray_self_grain(low, texture_gate):
    """Only compact fine-band residuals, gated identically to the grain layer."""
    details, meta = xray_self_details(low)
    fine = sum(a - blur(a, 0.8) for a in details) / 6
    delta = fine * np.clip(texture_gate, 0, 1)
    h, w = low.height, low.width
    # Preserve local tone under the box model; all gating happens before this.
    cells = delta.reshape(h, 4, w, 4)
    cells -= cells.mean((1, 3), keepdims=True)
    meta.update(
        weights=[2 / 3, 1 / 6, 1 / 6],
        recipe="bounded-self-grain-v3",
        full_paper_implementation=False,
        scope="fine_grain_only",
        grain_highpass_radius=0.8,
        existing_texture_gate=True,
        vibrance=dict(layer="dcci", strength=0.02, muted_weighted=True),
    )
    return delta, meta


def xray_self_average(low, rgb, texture_gate=None):
    """Diagnostic grain-only blend. Production integrates before tone/lighting."""
    if texture_gate is None:
        texture_gate = np.ones((low.height * 4, low.width * 4), np.float32)
    delta, meta = xray_self_grain(low, texture_gate)
    # Shared RGB headroom prevents saturation without independent channel clips.
    # A per-source-cell gain retains the zero-mean correction in the box model.
    rgb = np.clip(np.asarray(rgb, float), 0, 255)
    lo = -rgb.min(-1)
    hi = 255 - rgb.max(-1)
    gain = np.minimum(
        1, np.where(delta > 0, hi / np.maximum(delta, 1e-8), lo / np.minimum(delta, -1e-8))
    )
    h, w = low.height, low.width
    gain = gain.reshape(h, 4, w, 4).min((1, 3))
    delta *= np.repeat(np.repeat(gain, 4, 0), 4, 1)
    rgb = rgb + delta[..., None]
    return rgb, meta


# 03a — Source-supported grain protection and static contour sampling.


def spatial_smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def spatial_confidence(dl):
    energy = blur(abs(dl - blur(dl, 1.52)), 1)
    h, w = energy.shape
    p = np.pad(energy, ((0, 0), (4, 4)), mode="edge")
    peak = np.maximum.reduce([p[:, x : x + w] for x in range(9)])
    p = np.pad(peak, ((4, 4), (0, 0)), mode="edge")
    peak = np.maximum.reduce([p[y : y + h] for y in range(9)])
    return 1 - 0.98 * blur(spatial_smoothstep(1, 2.5, peak), 1)


def spatial_sample(field, x, y):
    h, w = field.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    if field.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (field[y0, x0] * (1 - fx) + field[y0, x1] * fx) * (1 - fy) + (
        field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    ) * fy


def spatial_edge_flow(carrier, radius=2.4, amount=0.85, guide=None, ridge=False):
    """Positive-weight integration along coherent contours, excluding grain fields.

    The structure tensor controls direction and spatial_confidence. Midpoint integration
    follows curves. No cross-normal sharpening or negative filter lobes are used.
    A range weight rejects samples that cross a neighbouring strand.
    """
    y = carrier @ LUMA
    guidance = y if guide is None else guide @ LUMA
    gy, gx = np.gradient(blur(guidance, 0.7))
    xx, yy, xy = (blur(v, 2) for v in (gx * gx, gy * gy, gx * gy))
    gap = np.sqrt((xx - yy) ** 2 + 4 * xy * xy)
    coherence = gap / (xx + yy + 1e-8)
    theta = 0.5 * np.arctan2(2 * xy, xx - yy)
    tx, ty = -np.sin(theta), np.cos(theta)
    strength = np.sqrt(xx + yy)
    mask = (
        spatial_smoothstep(0.7, 0.95, coherence) * spatial_smoothstep(2.5, 5, strength)
        if ridge
        else spatial_smoothstep(0.55, 0.88, coherence) * spatial_smoothstep(1.8, 4.2, strength)
    )
    h, w = y.shape
    py, px = np.mgrid[:h, :w].astype(np.float32)
    total = carrier.copy()
    weights = np.ones_like(y)
    for distance in [radius * 0.5, radius]:
        for sign in [-1, 1]:
            mx, my = px + sign * 0.5 * distance * tx, py + sign * 0.5 * distance * ty
            ux, uy = spatial_sample(tx, mx, my), spatial_sample(ty, mx, my)
            direction = np.where(ux * tx + uy * ty >= 0, 1, -1)
            norm = np.maximum(np.hypot(ux, uy), 1e-6)
            sx, sy = (
                px + sign * distance * ux * direction / norm,
                py + sign * distance * uy * direction / norm,
            )
            neighbour = spatial_sample(carrier, sx, sy)
            difference = neighbour @ LUMA - y
            weight = np.exp(-0.5 * (distance / radius) ** 2) / (
                1 + (difference / (12 if ridge else 8)) ** 4
            )
            total += neighbour * weight[..., None]
            weights += weight
    delta = total / weights[..., None] - carrier
    # Limit excursions without clipping individual RGB channels differently.
    cap = np.minimum(1, (10 if ridge else 5) / np.maximum(np.max(abs(delta), axis=2), 1e-8))
    delta *= (amount * mask * cap)[..., None]
    delta[:4] = delta[-4:] = 0
    delta[:, :4] = delta[:, -4:] = 0
    return delta, mask


def spatial_trajectory_kernel(radius):
    x = np.arange(-radius, radius + 1, dtype=float)
    design = np.stack([np.ones_like(x), x, x * x], axis=1)
    return np.linalg.pinv(design)[0]


def spatial_track_axis(luma, radius=10, displacement=1.1):
    """Trace ridge/valley and edge centres; regularize positions, not colours."""
    guide = blur(luma, 0.7)
    gy, gx = np.gradient(guide)
    xx, yy, xy = (blur(v, 2) for v in (gx * gx, gy * gy, gx * gy))
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-8)
    support = spatial_smoothstep(0.7, 0.96, coherence) * spatial_smoothstep(2, 5, np.sqrt(xx + yy))
    support *= spatial_smoothstep(0.42, 0.65, xx / (xx + yy + 1e-8))
    tracks = []
    active = {}
    h, w = guide.shape
    # Distinct feature types cannot attach to each other.
    for row in range(3, h - 3):
        features = []
        for kind, field in enumerate([guide[row], gx[row]]):
            centre = field[2:-2]
            left, right = field[1:-3], field[3:-1]
            maximum = (centre > left) & (centre >= right)
            minimum = (centre < left) & (centre <= right)
            if kind == 1:
                maximum &= centre > 3
                minimum &= centre < -3
            for polarity, valid in enumerate([maximum, minimum]):
                for index in np.flatnonzero(valid) + 2:
                    if support[row, index] < 0.25:
                        continue
                    curvature = field[index - 1] - 2 * field[index] + field[index + 1]
                    offset = min(
                        0.5,
                        max(
                            -0.5, 0.5 * (field[index - 1] - field[index + 1]) / (curvature + 1e-15)
                        ),
                    )
                    features.append((kind * 2 + polarity, index + offset, support[row, index]))
        buckets = {}
        for kind, indices in active.items():
            for ti in indices:
                buckets.setdefault((kind, int(tracks[ti][-1][1])), []).append(ti)
        candidates = []
        for fi, (kind, x, score) in enumerate(features):
            nearby = (
                ti
                for column in range(int(x) - 2, int(x) + 3)
                for ti in buckets.get((kind, column), ())
            )
            for ti in nearby:
                path = tracks[ti]
                prev = path[-1]
                if prev[0] != row - 1:
                    continue
                velocity = min(1.3, max(-1.3, prev[1] - path[-2][1])) if len(path) > 1 else 0
                error = abs(x - (prev[1] + velocity))
                if error < 1.4 and abs(x - prev[1]) < 1.8:
                    candidates.append((error, fi, ti))
        used_f, used_t = set(), set()
        updated = {}
        for _, fi, ti in sorted(candidates):
            if fi in used_f or ti in used_t:
                continue
            kind, x, score = features[fi]
            tracks[ti].append((row, x, score))
            updated.setdefault(kind, []).append(ti)
            used_f.add(fi)
            used_t.add(ti)
        for fi, (kind, x, score) in enumerate(features):
            if fi not in used_f:
                tracks.append([(row, x, score)])
                updated.setdefault(kind, []).append(len(tracks) - 1)
        active = updated
    accum = np.zeros_like(guide)
    weight = np.zeros_like(guide)
    kernel = spatial_trajectory_kernel(radius)
    accepted = 0
    for path in tracks:
        if len(path) < max(18, 2 * radius + 1):
            continue
        values = np.asarray(path)
        original = values[:, 1]
        padded = np.pad(original, (radius, radius), mode="reflect")
        fitted = np.convolve(padded, kernel, mode="valid")
        taper = spatial_smoothstep(
            0, 6, np.minimum(np.arange(len(path)), np.arange(len(path))[::-1])
        )
        shifts = np.clip(original - fitted, -displacement, displacement) * taper
        if np.max(abs(shifts)) < 0.02:
            continue
        accepted += 1
        targets = values[:, 1] - shifts
        rows = values[:, 0].astype(np.intp)
        starts = targets.astype(np.intp)
        for offset in range(-5, 7):
            columns = starts + offset
            valid = (columns >= 0) & (columns < w)
            rr, cc = rows[valid], columns[valid]
            weights = (
                np.exp(-0.5 * ((cc - targets[valid]) / 2.4) ** 2) * values[valid, 2] * taper[valid]
            )
            accum[rr, cc] += shifts[valid] * weights
            weight[rr, cc] += weights
    # Zero outside accepted coherent traces; no global filtering of texture.
    field = accum / np.maximum(weight, 1e-8)
    field *= spatial_smoothstep(0.03, 0.35, weight) * support
    field[:4] = field[-4:] = 0
    field[:, :4] = field[:, -4:] = 0
    return field, accepted


def spatial_cubic_axis(rgb, shift, axis=1):
    """Cubic coordinate sampling bounded by the four local samples."""
    a = rgb if axis == 1 else rgb.swapaxes(0, 1)
    d = shift if axis == 1 else shift.T
    h, w = a.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    position = np.clip(xx + d, 0, w - 1)
    i = position.astype(int)
    t = (position - i)[..., None]
    p0, p1, p2, p3 = (a[yy, np.clip(i + j, 0, w - 1)] for j in [-1, 0, 1, 2])
    out = p1 + 0.5 * t * (
        p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0))
    )
    out = np.clip(out, np.minimum.reduce([p0, p1, p2, p3]), np.maximum.reduce([p0, p1, p2, p3]))
    return out if axis == 1 else out.swapaxes(0, 1)


def spatial_realign(carrier, radius=10, displacement=1.1):
    y = carrier @ LUMA
    sx, nx = spatial_track_axis(y, radius, displacement)
    sy, ny = spatial_track_axis(y.T, radius, displacement)
    sy = sy.T
    # Select the dominant tracking axis per pixel instead of double-warping it.
    gy, gx = np.gradient(blur(y, 1))
    xx, yy = (blur(v, 2) for v in [gx * gx, gy * gy])
    horizontal = spatial_smoothstep(0.35, 0.65, xx / (xx + yy + 1e-8))
    outx = spatial_cubic_axis(carrier, sx)
    outy = spatial_cubic_axis(carrier, sy, axis=0)
    out = horizontal[..., None] * outx + (1 - horizontal[..., None]) * outy
    return out, {
        "x_tracks": nx,
        "y_tracks": ny,
        "max_x_displacement": float(abs(sx).max()),
        "max_y_displacement": float(abs(sy).max()),
    }


def spatial_normal_definition(structure, amount):
    y = structure @ LUMA
    gy, gx = np.gradient(blur(y, 0.7))
    xx, yy, xy = (blur(v, 2) for v in [gx * gx, gy * gy, gx * gy])
    theta = 0.5 * np.arctan2(2 * xy, xx - yy)
    nx, ny = np.cos(theta), np.sin(theta)
    py, px = np.mgrid[: y.shape[0], : y.shape[1]]
    across = 0.5 * (
        spatial_sample(y, px + nx * 1.2, py + ny * 1.2)
        + spatial_sample(y, px - nx * 1.2, py - ny * 1.2)
    )
    detail = amount * np.clip(y - across, -6, 6) * spatial_edge_mask(structure)
    h, w = y.shape
    p = np.pad(y, 1, mode="edge")
    neighbours = [p[j : j + h, i : i + w] for j in range(3) for i in range(3)]
    bounded = np.clip(y + detail, np.minimum.reduce(neighbours), np.maximum.reduce(neighbours))
    return xray_luma(structure, bounded - y)


def spatial_edge_mask(carrier):
    y = carrier @ LUMA
    gy, gx = np.gradient(blur(y, 0.7))
    xx, yy, xy = (blur(v, 2) for v in (gx * gx, gy * gy, gx * gy))
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-8)
    return spatial_smoothstep(0.7, 0.96, coherence) * spatial_smoothstep(2, 5, np.sqrt(xx + yy))


def spatial_resize(rgb, size):
    return np.stack(
        [
            np.asarray(
                Image.fromarray(rgb[..., c].astype(np.float32)).resize(
                    size, Image.Resampling.BICUBIC
                )
            )
            for c in range(3)
        ],
        axis=2,
    )


def spatial_protect_grain(grain, source_luma):
    """Avoid re-emphasizing compact DCCI detail; round grain with bounded energy recovery."""
    g = grain * spatial_confidence(source_luma)
    smoothed = blur(g, 0.45)
    gain = np.minimum(
        1.35, np.sqrt(blur(g * g, 3) / np.maximum(blur(smoothed * smoothed, 3), 1e-12))
    )
    return 0.35 * g + 0.65 * smoothed * gain


def spatial_fxaa(rgb, subpixel=0.25, threshold=0.063):
    """FXAA 3.11 PC Quality, preset 39, in pixel coordinates on the CPU.

    Copyright (c) 2014 NVIDIA CORPORATION. BSD-3-Clause; see licenses/FXAA-BSD.txt.
    Only contrast-qualified pixels enter the endpoint search. No GPU context or
    additional runtime is required. The last search step advances without a fetch,
    matching the reference shader's bounded endpoint search.
    """
    y = np.asarray(rgb @ LUMA, np.float64)
    h, w = y.shape
    p = np.pad(y, 1, mode="edge")
    n, south, west, east = p[:h, 1 : w + 1], p[2:, 1 : w + 1], p[1 : h + 1, :w], p[1 : h + 1, 2:]
    hi = np.maximum.reduce([y, n, south, west, east])
    lo = np.minimum.reduce([y, n, south, west, east])
    rows, cols = np.nonzero(hi - lo >= np.maximum(255 * 0.015625, hi * threshold))
    if not len(rows):
        return rgb.copy()
    pick = (rows, cols)
    m, n, south, west, east = (v[pick] for v in (y, n, south, west, east))
    nw, ne, sw, se = (v[pick] for v in (p[:h, :w], p[:h, 2:], p[2:, :w], p[2:, 2:]))
    edge_h = 2 * abs(-2 * m + n + south) + abs(-2 * east + ne + se) + abs(-2 * west + nw + sw)
    edge_v = 2 * abs(-2 * m + west + east) + abs(-2 * n + nw + ne) + abs(-2 * south + sw + se)
    horizontal = edge_h >= edge_v
    average = (2 * (n + south + west + east) + nw + ne + sw + se) / 12
    c = np.clip(abs(average - m) / (hi - lo)[pick], 0, 1)
    sub = (c * c * (3 - 2 * c)) ** 2 * subpixel
    neg = np.where(horizontal, n, west)
    pos = np.where(horizontal, south, east)
    gn, gp = neg - m, pos - m
    pair_n = abs(gn) >= abs(gp)
    sign = np.where(pair_n, -1.0, 1.0)
    pair = 0.5 * (m + np.where(pair_n, neg, pos))
    gradient = 0.25 * np.maximum(abs(gn), abs(gp))
    bx = cols + np.where(horizontal, 0, 0.5 * sign)
    by = rows + np.where(horizontal, 0.5 * sign, 0)
    ox, oy = horizontal.astype(float), (~horizontal).astype(float)
    dn, dp = np.ones(len(rows)), np.ones(len(rows))
    en = spatial_sample(y, bx - ox, by - oy) - pair
    ep = spatial_sample(y, bx + ox, by + oy) - pair
    done_n, done_p = abs(en) >= gradient, abs(ep) >= gradient
    steps = (1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0)
    for i, step in enumerate(steps):
        dn += (~done_n) * step
        dp += (~done_p) * step
        if i == len(steps) - 1 or (done_n.all() and done_p.all()):
            break
        active = ~done_n
        en[active] = (
            spatial_sample(y, (bx - ox * dn)[active], (by - oy * dn)[active]) - pair[active]
        )
        active = ~done_p
        ep[active] = (
            spatial_sample(y, (bx + ox * dp)[active], (by + oy * dp)[active]) - pair[active]
        )
        done_n, done_p = abs(en) >= gradient, abs(ep) >= gradient
    good = np.where(dn < dp, en < 0, ep < 0) != (m - pair < 0)
    offset = np.maximum(np.where(good, 0.5 - np.minimum(dn, dp) / (dn + dp), 0), sub) * sign
    result = rgb.copy()
    result[pick] = spatial_sample(
        rgb, cols + np.where(horizontal, 0, offset), rows + np.where(horizontal, offset, 0)
    )
    return result


SPATIAL_PROFILES = {
    "fast": dict(radius=10, displacement=0.9, flow=0.65, flow_radius=3.2, definition=0.25),
    "max": dict(radius=16, displacement=1.1, flow=0.75, flow_radius=4.0, definition=0.25),
    "xray": dict(radius=16, displacement=1.6, flow=0.9, flow_radius=4.5, definition=0.5),
}


def spatial_envelope(carrier):
    """Compute one shared local colour envelope for both structural branches."""
    low, high = carrier.copy(), carrier.copy()
    for axis in (0, 1):
        pads = [(0, 0), (0, 0), (0, 0)]
        pads[axis] = (2, 2)
        pl, ph = np.pad(low, pads, mode="edge"), np.pad(high, pads, mode="edge")
        lows, highs = [], []
        for offset in range(5):
            sl = [slice(None)] * 3
            sl[axis] = slice(offset, offset + carrier.shape[axis])
            lows.append(pl[tuple(sl)])
            highs.append(ph[tuple(sl)])
        low, high = np.minimum.reduce(lows), np.maximum.reduce(highs)
    return low, high


def spatial_branch(rgb, carrier, mode, aligned, fxaa, mask, envelope):
    """Evaluate one bounded branch without recomputing contributors."""
    profile = SPATIAL_PROFILES[mode]
    if not np.any(mask > 0.01):
        return rgb, dict(
            recipe="spatial-77-23-v1", profile=mode, skipped="No coherent edge support"
        )
    if aligned:
        structure, meta = spatial_realign(carrier, profile["radius"], profile["displacement"])
    else:
        structure, meta = carrier.copy(), {}
    meta.update(realignment=aligned, fxaa=fxaa)
    correction, _ = spatial_edge_flow(
        structure, profile["flow_radius"], profile["flow"], guide=structure, ridge=True
    )
    structure += correction
    if fxaa:
        h, w = carrier.shape[:2]
        half = spatial_resize(structure, (max(1, w // 2), max(1, h // 2)))
        delta = spatial_resize(spatial_fxaa(half) - half, (w, h))
        structure += mask[..., None] * 0.75 * delta
    structure = spatial_normal_definition(structure, profile["definition"])
    # Intermediate bicubic correction can ring even when FXAA itself is convex.
    # Bound the structural delta to a five-pixel source neighbourhood, with one
    # shared RGB factor. Texture is restored after this bound and is not filtered.
    low, high = envelope
    change = structure - carrier
    room = np.where(
        change > 0,
        (high - carrier) / np.maximum(change, 1e-12),
        (low - carrier) / np.minimum(change, -1e-12),
    )
    factor = np.clip(room.min(axis=2), 0, 1)
    result = rgb + change * factor[..., None]
    meta.update(
        recipe="spatial-77-23-v1",
        profile=mode,
        parameters=dict(profile),
        texture_residual_retained=True,
    )
    return result, meta


def spatial_cleanup(rgb, carrier, mode):
    """Mix 77% unaligned FXAA with 23% aligned structure without FXAA.

    Contributor fields and neighbourhood bounds are shared. Each branch retains
    the same texture residual; floating-point composition avoids intermediate
    export quantization. Destination sampling and PNG rounding remain downstream.
    """
    mask = spatial_edge_mask(carrier)
    if not np.any(mask > 0.01):
        return rgb, dict(
            recipe="spatial-77-23-v1", profile=mode, skipped="No coherent edge support"
        )
    envelope = spatial_envelope(carrier)
    unaligned, _ = spatial_branch(rgb, carrier, mode, False, True, mask, envelope)
    aligned, aligned_meta = spatial_branch(rgb, carrier, mode, True, False, mask, envelope)
    result = 0.77 * unaligned + 0.23 * aligned
    return result, dict(
        recipe="spatial-77-23-v1",
        profile=mode,
        branches=[
            dict(weight=0.77, realignment=False, fxaa=True),
            dict(weight=0.23, realignment=True, fxaa=False),
        ],
        max_x_displacement=aligned_meta.get("max_x_displacement", 0),
        max_y_displacement=aligned_meta.get("max_y_displacement", 0),
        parameters=dict(SPATIAL_PROFILES[mode]),
        texture_residual_retained=True,
        shared_contributors=True,
        shared_envelope=True,
    )


def normalize_mode(mode):
    """Resolve the 1.x command alias without exposing a duplicate quality mode."""
    mode = "xray" if mode == "xray-plus" else mode
    if mode not in ("fast", "max", "xray"):
        raise ValueError("Mode must be fast, max or xray")
    return mode


def detail_patch(
    low,
    folder,
    notify=lambda x: None,
    texture_box=None,
    mode="max",
    acceleration=None,
    use_fitted=True,
):
    """DCCI/fitted base, Local Laplacian mid, four-way fine-detail restoration."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mode = normalize_mode(mode)
    if mode == "xray":
        return xray_detail_patch(low, folder, notify, acceleration, use_fitted, self_examples=True)
    timings = {}
    phase = [None, time.perf_counter()]

    def stage(name):
        now = time.perf_counter()
        if phase[0] is not None:
            timings[phase[0]] = now - phase[1]
        phase[:] = [name, now]
        if name is not None:
            notify(name)

    pixels = arr(low)
    if np.all(pixels == pixels[0, 0]):
        return low.resize((low.width * 4, low.height * 4)), {"flat": True, "texture_box": None}
    stage("Restoring edges")
    dcci = dcci_four(low)
    fitted = None
    if use_fitted:
        stage("Fitting colour and shading")
        fitted = fitted_four(low)
    stage("Resolving fine contours")
    src = folder / "source.bmp"
    dst = folder / "contours.bmp"
    low.save(src)
    with (folder / "contours.log").open("w") as log:
        subprocess.run(
            [str(BIN / "sinterp.exe"), "-x", "4", "-p", ".35", "-r", "3", str(src), str(dst)],
            check=True,
            stdout=log,
            stderr=log,
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    with Image.open(dst) as im:
        contours = im.convert("RGB")
    if contours.size != dcci.size:
        raise ValueError("Contour renderer returned incorrect dimensions")
    d, c = [arr(im).astype(float) for im in [dcci, contours]]
    f = arr(fitted).astype(float) if fitted is not None else None
    dl, cl = d @ LUMA, c @ LUMA
    base = np.clip(d - blur(d, 4.8) + blur(f, 4.8), 0, 255) if use_fitted else d.copy()
    stage("Refining middle detail")
    bins = 32 if mode == "fast" else 64
    ll = (
        acceleration.laplacian(dl, bins, folder) if acceleration else local_laplacian(dl, bins=bins)
    )
    db, dm, _ = bands(dl, 4)
    _, lm, _ = bands(ll, 4)
    structure = 0.65 * gate(dl, dl, db, db, 4) * (lm - dm)
    stage("Restoring surface detail")
    yl = arr(low).astype(float) @ LUMA
    dy, dx = np.gradient(blur(yl, 0.65))
    edge = np.hypot(dx, dy)
    texture_gate = (
        resize_float(np.exp(-((edge / 9) ** 2)), dcci.size)
        * (1 - np.maximum(structure_mask(dl, 4), structure_mask(cl, 4))) ** 2
    )
    texture_gate[:4] = 0
    texture_gate[-4:] = 0
    texture_gate[:, :4] = 0
    texture_gate[:, -4:] = 0
    grain, grain_meta = restore_grain(
        base, (dl, f @ LUMA, ll, cl) if use_fitted else (dl, ll, cl), texture_gate
    )
    stage("Protecting source texture")
    grain = spatial_protect_grain(grain, dl)
    correction = 14 * np.tanh(structure / 14)
    carrier = xray_luma(base, correction)
    rgb = xray_luma(base, correction + grain)
    stage("Refining contour sampling")
    rgb, spatial_meta = spatial_cleanup(rgb, carrier, mode)
    output = image(rgb)
    stage(None)
    return output, dict(
        recipe="source-grain-spatial-blend-v3"
        if use_fitted
        else "source-grain-spatial-blend-no-fitted-v3",
        fitted_enabled=use_fitted,
        grain=grain_meta,
        spatial=spatial_meta,
        timings=timings,
    )


def dimensions(size, scale):
    if not math.isfinite(scale) or not 1 <= scale <= 16:
        raise ValueError("Choose a scale from 1 to 16, including decimals.")
    w, h = (max(1, int(math.floor(v * scale + 0.5))) for v in size)
    if w * h > MAX_OUTPUT:
        raise ValueError("This size exceeds 160 megapixels. Choose a smaller scale.")
    return w, h


def vector_export(folder):
    """Stream exact bilinear colour fields as paths, gradients and vector masks.

    All coordinates are SVG geometry. The field is built from restored PNG
    samples; converting its representation cannot recover additional information.
    """
    folder = Path(folder)
    png = folder / "result.png"
    target = folder / "result.svg"
    temp = folder / "result.svg.part"

    def report(stage, progress):
        save_json(
            folder / "svg-progress.json", dict(stage=stage, progress=progress, updated=time.time())
        )

    with Image.open(png) as im:
        w, h = im.size
        if w * h > 2_000_000:
            raise ValueError(
                "True-vector export is limited to 2 megapixels to bound file size and renderer cost. Your full-resolution PNG is ready and unchanged. Export a smaller job for SVG."
            )
        pixels = np.asarray(im.convert("RGBA"))
    tiles = [
        (x, y, min(w, x + 128), min(h, y + 128)) for y in range(0, h, 128) for x in range(0, w, 128)
    ]
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="{w}" height="{h}" viewBox="0 0 {w} {h}" color-interpolation="sRGB"><title>Untorn Engine — vector field</title><desc>Continuous bilinear field of restored colour samples. Vector paths, gradients and masks only; no embedded raster. Fine geometry does not prove recovery of missing photographic detail.</desc>'
            )
            for i, (x, y, r, b) in enumerate(tiles):
                report("Compiling vector field", (i + 1) / len(tiles))
                lx, ly, rx, by = max(0, x - 2), max(0, y - 2), min(w, r + 2), min(h, b + 2)
                svg = compact_surface_svg(pixels[ly:by, lx:rx])
                content = svg[svg.index(">") + 1 : -6]
                content = re.sub(r'id="([^"]+)"', lambda m: f'id="t{i}_{m[1]}"', content)
                content = re.sub(r"url\(#([^\)]+)\)", lambda m: f"url(#t{i}_{m[1]})", content)
                # A one-pixel shared boundary guards antialiasing cracks in readers.
                cx, cy, cr, cb = max(0, x - 1), max(0, y - 1), min(w, r + 1), min(h, b + 1)
                stream.write(
                    f'<defs><clipPath id="clip{i}" clipPathUnits="userSpaceOnUse"><path d="M{cx} {cy}H{cr}V{cb}H{cx}Z"/></clipPath></defs><g clip-path="url(#clip{i})"><g transform="translate({lx},{ly})">'
                    + content
                    + "</g></g>"
                )
                if stream.tell() > 250_000_000:
                    raise ValueError(
                        "SVG reached its 250 MB limit. PNG remains ready; no vector detail was discarded to force completion."
                    )
            stream.write("</svg>")
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temp, target)
        report("Vector SVG ready", 1)
        save_json(
            folder / "svg.json",
            dict(
                format="SVG 1.1",
                raster_elements=0,
                filters=0,
                geometry="paths / linear gradients / vector masks",
                size=[w, h],
                bytes=target.stat().st_size,
                sha256=sha(target),
                sampling="Continuous bilinear field of final restoration samples; no infinite photographic detail claim",
            ),
        )
    finally:
        temp.unlink(missing_ok=True)


def bicubic_export(folder):
    folder = Path(folder)
    request = read_json(folder / "request.json")
    meta = read_json(folder / "result.json")
    source, _ = decode(Path(request["input"]))
    source = source.convert("RGB")
    ow, oh = meta["output_size"]
    # Full Pillow sampling avoids strip-boundary coefficient roundoff. This
    # isolated operation has no floating-point colour planes and stays within
    # the worker budget at the validated 160 MP output cap. Only a preview is saved.
    im = source.resize((ow, oh), Image.Resampling.BICUBIC)
    im.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    im.save(folder / "bicubic-preview.png", compress_level=3)
    save_json(
        folder / "bicubic.json",
        dict(
            size=[ow, oh],
            method="Pillow BICUBIC; same decoded source and exact destination dimensions",
            preview_sha256=sha(folder / "bicubic-preview.png"),
        ),
    )


# 02a — Hardware selection. Optional GPU work is always isolated and bounded.


def memory_budget():
    available = 4 * 1024**3
    if os.name == "nt":
        import ctypes as ct

        class Memory(ct.Structure):
            _fields_ = [("length", ct.c_uint32), ("load", ct.c_uint32)] + [
                (k, ct.c_uint64)
                for k in [
                    "total",
                    "available",
                    "page",
                    "available_page",
                    "virtual",
                    "available_virtual",
                    "extended",
                ]
            ]

        m = Memory()
        m.length = ct.sizeof(m)
        if ct.windll.kernel32.GlobalMemoryStatusEx(ct.byref(m)):
            available = m.available
    # Preserve memory for the desktop and other applications. The hard job limit
    # is an upper bound, not a preallocation or a promise that memory stays free.
    budget = max(1024**3, min(8 * 1024**3, int(available * 0.55)))
    return available, budget


def worker_count(tiles, requested=0):
    if os.environ.get("UNTORN_SAFE_WORKER") == "1":
        return 1
    _, budget = memory_budget()
    automatic = max(1, min(6, max(1, (os.cpu_count() or 2) // 2), budget // (1536 * 1024**2)))
    return max(1, min(tiles, automatic, requested if requested else automatic))


class Acceleration:
    """GPU dispatch for the original FP64 Local Laplacian equations.

    Ranking prefers discrete devices. A deterministic native calibration verifies
    arithmetic and measures the crossover. Driver failure/timeout/invalid data
    discards the GPU result and immediately uses the CPU implementation.
    """

    def __init__(self, folder, device="auto"):
        self.folder = Path(folder)
        self.device = device
        self.lock = threading.Lock()
        self.plan = None
        self.calls = 0
        self.fallbacks = []
        self.used = set()
        self.disabled = False

    def execute(self, a, bins, device, folder, timeout=15):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "gpu-input.f64"
        target = folder / "gpu-output.f64"
        h, w = a.shape
        try:
            source.write_bytes(struct.pack("<II", w, h) + np.asarray(a, dtype="<f8").tobytes())
            target.unlink(missing_ok=True)
            p = subprocess.run(
                [
                    str(BIN / "untorn_gpu.exe"),
                    str(source),
                    str(target),
                    str(bins),
                    str(device),
                    "1",
                ],
                capture_output=True,
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if p.returncode:
                raise RuntimeError(
                    p.stderr.decode(errors="replace")[:500] or f"GPU executor exit {p.returncode}"
                )
            raw = target.read_bytes()
            if len(raw) != 8 + w * h * 8 or struct.unpack("<II", raw[:8]) != (w, h):
                raise ValueError("Incomplete GPU output")
            result = np.frombuffer(raw[8:], "<f8").reshape(h, w).copy()
            if not np.isfinite(result).all() or abs(result).max() > 1e5:
                raise ValueError("Invalid GPU output")
            return result
        finally:
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)

    def prepare(self):
        if self.plan is not None:
            return
        self.plan = {"backend": "cpu", "reason": "CPU requested"}
        if self.device == "cpu" or os.environ.get("UNTORN_SAFE_WORKER") == "1":
            return
        exe = BIN / "untorn_gpu.exe"
        if os.name != "nt" or not exe.exists():
            self.plan["reason"] = "Optional GPU executor unavailable"
            return
        try:
            p = subprocess.run(
                [str(exe), "--probe"],
                capture_output=True,
                text=True,
                timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if p.returncode:
                raise ValueError(p.stderr[:500] or "GPU enumeration failed")
            devices = json.loads(p.stdout)
            if not devices:
                self.plan["reason"] = "No compatible FP64 OpenCL GPU"
                return
            fingerprint = hashlib.sha256(
                (sha(exe) + json.dumps(devices, sort_keys=True)).encode()
            ).hexdigest()
            cache = self.folder.parent / ".hardware" / f"{fingerprint}.json"
            saved = read_json(cache)
            if (
                saved
                and saved.get("contract") == 1
                and time.time() - saved.get("created", 0) < 7 * 86400
            ):
                self.plan = saved
                return
            self.plan = {
                "backend": "cpu",
                "reason": "No GPU passed validation",
                "contract": 1,
                "created": time.time(),
                "devices": devices,
                "rejected": [],
            }
            # Calibrate at a useful workload instead of assuming that a GPU is
            # faster than a CPU for tiny arrays. No source photograph is used.
            yy, xx = np.mgrid[:384, :384]
            sample = (
                127
                + 60 * np.sin(xx * 0.073)
                + 40 * np.cos(yy * 0.119)
                + 15 * np.sin((xx + yy) * 0.31)
            ).astype(float)
            start = time.perf_counter()
            expected = local_laplacian(sample, bins=64)
            cpu_seconds = time.perf_counter() - start
            for dev in devices[:3]:
                try:
                    start = time.perf_counter()
                    actual = self.execute(sample, 64, dev["index"], self.folder / "gpu-validation")
                    gpu_seconds = time.perf_counter() - start
                    error = float(abs(expected - actual).max())
                    if error > 1e-7:
                        raise ValueError(f"GPU arithmetic mismatch: {error:g}")
                    threshold = int(
                        max(96 * 96, 384 * 384 * gpu_seconds / max(cpu_seconds, 0.001) * 1.25)
                    )
                    self.plan.update(
                        backend="opencl-fp64",
                        device=dev,
                        minimum_work=threshold,
                        cpu_seconds=cpu_seconds,
                        gpu_seconds=gpu_seconds,
                        maximum_error=error,
                        reason="Validated GPU with measured workload crossover",
                    )
                    break
                except Exception as exc:
                    self.plan["rejected"].append({"device": dev["name"], "reason": str(exc)[:500]})
            save_json(cache, self.plan)
        except Exception as exc:
            self.plan = {"backend": "cpu", "reason": str(exc)[:500]}
        save_json(self.folder / "hardware.json", self.plan)

    def laplacian(self, a, bins, folder):
        # Small stages stay on CPU. Calibration happens only when a stage is
        # large enough to plausibly repay driver/context startup.
        if self.device == "cpu" or self.disabled or a.size * bins < 128 * 1024 * 64:
            return local_laplacian(a, bins=bins)
        with self.lock:
            self.prepare()
            if (
                self.plan.get("backend") == "opencl-fp64"
                and a.size * bins >= self.plan["minimum_work"] * 64
                and not self.disabled
            ):
                try:
                    out = self.execute(a, bins, self.plan["device"]["index"], folder)
                    self.calls += 1
                    self.used.add(self.plan["device"]["name"])
                    return out
                except Exception as exc:
                    self.disabled = True
                    self.fallbacks.append(str(exc)[:500])
                    save_json(
                        self.folder / "gpu-fallback.json",
                        dict(
                            reason=self.fallbacks[-1],
                            action="CPU recomputation; GPU output discarded",
                        ),
                    )
        return local_laplacian(a, bins=bins)

    def report(self):
        return {
            "selection": self.plan or {"backend": "cpu", "reason": "Small workload"},
            "gpu_stages": self.calls,
            "gpu_devices": sorted(self.used),
            "fallbacks": list(self.fallbacks),
        }


def reconstruct_tile(low, folder, bounds, index, count, mode, acceleration=None, progress=None):
    x, y, r, b = bounds
    w, h = low.size
    cache = folder / "tiles" / f"{x}-{y}"
    cache.mkdir(parents=True, exist_ok=True)
    l, t, rr, bb = max(0, x - 16), max(0, y - 16), min(w, r + 16), min(h, b + 16)
    pl, pt, pr, pb = max(0, l - 32), max(0, t - 32), min(w, rr + 32), min(h, bb + 32)
    tile = cache / "detail.png"
    receipt = cache / "receipt.json"
    expected = ((rr - l) * 4, (bb - t) * 4)
    good = False
    try:
        good = tile.exists() and read_json(receipt, {}).get("sha256") == sha(tile)
        if good:
            with Image.open(tile) as check:
                good = check.size == expected
                check.verify()
    except Exception:
        good = False
    if not good:

        def notify(stage):
            if progress:
                progress.update(index, stage)
            else:
                status_file(folder, stage)

        result, meta = detail_patch(
            low.crop((pl, pt, pr, pb)),
            cache,
            notify,
            mode=mode,
            acceleration=acceleration,
            use_fitted=use_fitted_for(low.size),
        )
        result = result.crop(((l - pl) * 4, (t - pt) * 4, (rr - pl) * 4, (bb - pt) * 4))
        temp = cache / "detail.part"
        result.save(temp, format="PNG")
        replace_file(temp, tile)
        save_json(receipt, dict(sha256=sha(tile), **meta))
        for scratch in cache.iterdir():
            if scratch.suffix in {".bmp", ".raw"}:
                scratch.unlink()
    if progress:
        progress.update(index, "Detail ready", 0.92)
    return l, t, rr, bb, tile, None if good else meta.get("timings", {})


# 02b — DCCI output enlargement. Bounded patches share a global sample grid.
def dcci_midpoints(a):
    """Exact diagonal phase of dcci_twice, without its unused lattice nodes."""
    h, w, _ = a.shape
    src = np.pad(a, ((2, 2), (2, 2), (0, 0)), mode="edge")

    def at(y, x):
        return src[2 + y : 2 + y + h - 1, 2 + x : 2 + x + w - 1]

    d1 = sum(abs(at(y, x) - at(y - 1, x + 1)) for y in [0, 1, 2] for x in [-1, 0, 1])
    d2 = sum(abs(at(y, x) - at(y + 1, x + 1)) for y in [-1, 0, 1] for x in [-1, 0, 1])
    p1 = (-at(2, -1) + 9 * at(1, 0) + 9 * at(0, 1) - at(-1, 2)) / 16
    p2 = (-at(-1, -1) + 9 * at(0, 0) + 9 * at(1, 1) - at(2, 2)) / 16
    w1 = 1 / (1 + d1**5)
    w2 = 1 / (1 + d2**5)
    return np.where(
        (1 + d1) / (1 + d2) > 1.15,
        p2,
        np.where((1 + d2) / (1 + d1) > 1.15, p1, (w1 * p1 + w2 * p2) / (w1 + w2)),
    )


def dcci_output(im, size, box, notify=lambda: None, block=160):
    """Enlarge a restored tile with DCCI, without Lanczos enlargement.

    DCCI inserts nodes between existing sample centres. One extra half-step
    grid makes the pixel centres of 2x/4x outputs exact DCCI nodes, rather than
    shifting the image or resampling those nodes with a generic kernel.
    Fractional scales blend adjacent DCCI levels continuously; cubic sampling
    registers each level to the exact requested grid, without Lanczos enlargement.
    Context patches bound temporary arrays independently of output dimensions.
    """
    ow, oh = size
    left, top, right, bottom = box
    sx = (right - left) / ow
    sy = (bottom - top) / oh
    ratio = max(1 / sx, 1 / sy)
    if not 1 < ratio <= 4.000001:
        raise ValueError("DCCI output enlargement must be greater than 1x and at most 4x")
    factor = 2 ** int(math.ceil(math.log2(ratio) - 1e-10))
    # The extra grid resolves centre registration without another detail pass.
    steps = int(math.log2(factor))
    halo = 12
    fraction = np.clip((ratio - factor / 2) / (factor / 2), 0, 1)
    mix = fraction * fraction * (3 - 2 * fraction)
    block = max(16, min(int(block), int(max(16, (512 / factor - 2 * halo) * min(1 / sx, 1 / sy)))))
    result = np.empty((oh, ow, 3), np.float32)
    w, h = im.size
    for y in range(0, oh, block):
        y1 = min(oh, y + block)
        for x in range(0, ow, block):
            x1 = min(ow, x + block)
            notify()
            a = left + x * sx
            b = top + y * sy
            c = left + x1 * sx
            d = top + y1 * sy
            l = math.floor(a) - halo
            t = math.floor(b) - halo
            r = math.ceil(c) + halo
            bt = math.ceil(d) + halo
            cl = max(0, l)
            ct = max(0, t)
            cr = min(w, r)
            cb = min(h, bt)
            pixels = np.asarray(im.crop((cl, ct, cr, cb)), np.float64) / 255
            grid = np.pad(pixels, ((ct - t, bt - cb), (cl - l, r - cr), (0, 0)), mode="edge")
            lower = None
            for step in range(steps):
                if mix < 1 - 1e-10 and step == steps - 1:
                    lower = grid if step == 0 else dcci_midpoints(grid)
                grid = dcci_twice(grid)
            # Only the diagonal phase of the final half-step is needed.
            grid = dcci_midpoints(grid)
            xx = ((np.arange(x, x1) + 0.5) * sx + left - l - 0.5) * factor - 0.5
            yy = ((np.arange(y, y1) + 0.5) * sy + top - t - 0.5) * factor - 0.5
            if np.max(abs(xx - np.rint(xx))) < 1e-7 and np.max(abs(yy - np.rint(yy))) < 1e-7:
                # At exact 8x/16x, every destination is a DCCI-computed node.
                patch = (
                    grid[np.rint(yy).astype(int)[:, None], np.rint(xx).astype(int)[None, :]] * 255
                )
            else:
                bounds = (
                    (a - l - 0.5) * factor,
                    (b - t - 0.5) * factor,
                    (c - l - 0.5) * factor,
                    (d - t - 0.5) * factor,
                )
                patch = np.stack(
                    [
                        np.asarray(
                            Image.fromarray((grid[..., channel] * 255).astype(np.float32)).resize(
                                (x1 - x, y1 - y), Image.Resampling.BICUBIC, box=bounds
                            )
                        )
                        for channel in range(3)
                    ],
                    -1,
                )
            if lower is not None:
                lf = factor / 2
                phase = 0 if steps == 1 else 0.5
                bounds = (
                    (a - l - 0.5) * lf - phase + 0.5,
                    (b - t - 0.5) * lf - phase + 0.5,
                    (c - l - 0.5) * lf - phase + 0.5,
                    (d - t - 0.5) * lf - phase + 0.5,
                )
                previous = np.stack(
                    [
                        np.asarray(
                            Image.fromarray((lower[..., channel] * 255).astype(np.float32)).resize(
                                (x1 - x, y1 - y), Image.Resampling.BICUBIC, box=bounds
                            )
                        )
                        for channel in range(3)
                    ],
                    -1,
                )
                # Adjacent scale levels agree at the octave boundary. A smooth
                # transition prevents tiny slider changes from switching the
                # whole image to a different edge/grain restoration.
                patch = previous + (patch - previous) * mix
            result[y:y1, x:x1] = np.clip(patch, 0, 255)
    return result


def process_request(request):
    folder = Path(request["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    status_file(folder, "Reading image")
    source = Path(request["input"])
    low, notes = decode(source)
    if np.asarray(low.getchannel("A")).min() != 255:
        raise ValueError(
            "This photographic engine currently requires an opaque image. Transparency was not flattened."
        )
    low = low.convert("RGB")
    w, h = low.size
    if min(w, h) < 16:
        raise ValueError("Use an image at least 16 pixels wide and high for detail analysis.")
    ow, oh = dimensions(low.size, float(request["scale"]))
    mode = request.get("mode", "max")
    mode = normalize_mode(mode)
    signature = dict(
        source=sha(source),
        version=VERSION,
        size=[ow, oh],
        svg=bool(request.get("svg")),
        method=CACHE_METHOD,
    )
    fitted_enabled = use_fitted_for((w, h))
    if not fitted_enabled:
        signature["fitted_enabled"] = False
    if mode != "max":
        signature["mode"] = mode
    if mode == "xray":
        signature["xray_recipe"] = XRAY_PLUS_RECIPE
    if max(ow / w, oh / h) > 4:
        signature["enlargement"] = "dcci-centred-v1"
    stamp = folder / "signature.json"
    if stamp.exists():
        previous = read_json(stamp, {})
        if previous.get("mode") == "xray-plus" and previous.get("xray_recipe") == XRAY_PLUS_RECIPE:
            migrated = dict(previous, mode="xray")
            if migrated == signature:
                save_json(stamp, migrated)
                previous = migrated
        # Only compatible restoration tiles can be reused across output scales.
        if (
            "enlargement" in signature
            and "enlargement" not in previous
            and previous == {k: v for k, v in signature.items() if k != "enlargement"}
        ):
            save_json(stamp, signature)
    if stamp.exists() and read_json(stamp) != signature:
        raise ValueError("Saved work belongs to a different input or setting. Start a new job.")
    save_json(stamp, signature)
    required = ow * oh * 17 + w * h * 70 + 100_000_000
    if shutil.disk_usage(folder).free < required:
        raise ValueError(
            f"Not enough working space: approximately {required / 1e9:.1f} GB required."
        )
    # A fixed native 4x restoration preserves the established recipe. Output
    # coordinates use the exact rounded destination dimensions, including decimals.
    tiles = (
        [(0, 0, w, h)]
        if max(w, h) <= 192
        else [
            (x, y, min(w, x + 128), min(h, y + 128))
            for y in range(0, h, 128)
            for x in range(0, w, 128)
        ]
    )
    progress = RestorationProgress(folder, len(tiles), fitted_enabled)
    acc_path = folder / "pixels.accum"
    weight_path = folder / "pixels.weights"
    total = np.memmap(acc_path, dtype=np.float32, mode="w+", shape=(oh, ow, 3))
    weights = np.memmap(weight_path, dtype=np.float32, mode="w+", shape=(oh, ow))
    total[:] = 0
    weights[:] = 0
    start = time.monotonic()
    pool = None
    timing_totals = {}
    timed_tiles = 0
    cached_tiles = 0
    workers = worker_count(len(tiles), int(request.get("workers", 0)))
    acceleration = Acceleration(folder, request.get("device", "auto"))
    save_json(
        folder / "execution.json",
        dict(
            workers=workers, device=request.get("device", "auto"), memory_limit=memory_budget()[1]
        ),
    )
    try:
        futures = None
        if workers > 1:
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="reconstruct"
            )
            futures = [
                pool.submit(
                    reconstruct_tile, low, folder, box, i, len(tiles), mode, acceleration, progress
                )
                for i, box in enumerate(tiles)
            ]
        for index, (x, y, r, b) in enumerate(tiles):
            try:
                l, t, rr, bb, tile, tile_timings = (
                    futures[index].result()
                    if futures
                    else reconstruct_tile(
                        low, folder, (x, y, r, b), index, len(tiles), mode, acceleration, progress
                    )
                )
            except Exception as exc:
                if not futures:
                    raise
                # Drain in-flight tasks, preserve every validated tile, then
                # retry serially. No shared output accumulation occurs in workers.
                pool.shutdown(wait=True, cancel_futures=True)
                pool = None
                futures = None
                acceleration.disabled = True
                save_json(
                    folder / "parallel-fallback.json",
                    dict(reason=str(exc), action="Serial CPU retry; completed tiles preserved"),
                )
                l, t, rr, bb, tile, tile_timings = reconstruct_tile(
                    low, folder, (x, y, r, b), index, len(tiles), mode, None, progress
                )
            if tile_timings is None:
                cached_tiles += 1
            else:
                timed_tiles += 1
                for label, seconds in tile_timings.items():
                    timing_totals[label] = timing_totals.get(label, 0) + seconds
            sampling_started = time.perf_counter()
            x0 = max(0, math.ceil(l * ow / w))
            x1 = min(ow, math.floor(rr * ow / w))
            y0 = max(0, math.ceil(t * oh / h))
            y1 = min(oh, math.floor(bb * oh / h))
            # Rounding is global, not independently restarted at each tile.
            box = (
                (x0 * w / ow - l) * 4,
                (y0 * h / oh - t) * 4,
                (x1 * w / ow - l) * 4,
                (y1 * h / oh - t) * 4,
            )
            with Image.open(tile) as im:
                if max(ow / w, oh / h) > 4:
                    out = dcci_output(
                        im,
                        (x1 - x0, y1 - y0),
                        box,
                        lambda: progress.update(index, "DCCI output enlargement", 0.94),
                    )
                else:
                    out = np.asarray(
                        im.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS, box=box), np.float32
                    )
            timing_totals["Output sampling"] = (
                timing_totals.get("Output sampling", 0) + time.perf_counter() - sampling_started
            )

            def ramp(coords, start, end, limit):
                v = np.ones_like(coords)
                if start > 0:
                    v *= np.clip((coords - (start - 16)) / 32, 0, 1)
                if end < limit:
                    v *= np.clip(((end + 16) - coords) / 32, 0, 1)
                return v

            wx = ramp((np.arange(x0, x1) + 0.5) * w / ow, x, r, w)
            wy = ramp((np.arange(y0, y1) + 0.5) * h / oh, y, b, h)
            blend = wy[:, None] * wx[None, :]
            total[y0:y1, x0:x1] += out * blend[..., None]
            weights[y0:y1, x0:x1] += blend
            progress.update(index, "Tile saved", 1.0)
        status_file(folder, "Writing PNG", 0.95)
        encoding_started = time.perf_counter()

        def rows(total, weights):
            for row in range(oh):
                if not np.all(weights[row] > 0):
                    raise ArithmeticError("Incomplete restoration coverage")
                yield np.uint8(np.clip(np.rint(total[row] / weights[row, :, None]), 0, 255))

        png = folder / "result.png"
        write_stream_png(png, rows(total, weights), ow, oh)
        with Image.open(png) as im:
            im.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
            im.save(folder / "preview.png", compress_level=3)
        low.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        low.save(folder / "original.png", compress_level=3)
        timing_totals["PNG and previews"] = time.perf_counter() - encoding_started
        meta = dict(
            signature,
            timings=dict(
                stage_seconds=timing_totals,
                timed_tiles=timed_tiles,
                cached_tiles=cached_tiles,
                accounting="Sum of measured tile-stage seconds; parallel stages overlap, so totals are not wall time",
            ),
            mode=mode,
            app_version=APP_VERSION,
            input_size=[w, h],
            output_size=[ow, oh],
            scale=request["scale"],
            native_scale=4,
            tiles=len(tiles),
            workers=workers,
            acceleration=acceleration.report(),
            seconds=time.monotonic() - start,
            notes=notes,
            output_sha256=sha(png),
            fitted_policy=dict(
                enabled=fitted_enabled, source_pixels=w * h, maximum_pixels=FITTED_MAX_PIXELS
            ),
            recipe=(
                "DCCI / fitted shading / Local Laplacian mid / equal source-detail grain with Unsharp 50%"
                if fitted_enabled
                else "DCCI / Local Laplacian mid / three-way source-detail grain with Unsharp 50%"
            ),
            recipe_id=(
                "source-grain-spatial-blend-v3"
                if fitted_enabled
                else "source-grain-spatial-blend-no-fitted-v3"
            ),
            sampling=(
                "4x restoration; adaptive DCCI enlargement with exact centre nodes at 8x/16x; continuous adjacent-level blending and cubic sampling for fractional output"
                if max(ow / w, oh / h) > 4
                else "4x native restoration; pixel-centred Lanczos reduction to requested dimensions"
            ),
            texture="Compact source-detail restoration; fixed masks, equal contributors and bounded luminance contrast",
            svg="Optional separate true-vector field export",
            spatial=dict(
                recipe="spatial-77-23-v1",
                profile=mode,
                parameters=dict(SPATIAL_PROFILES[mode]),
                grain_guard="Avoid repeated compact-detail emphasis; energy-bounded grain rounding",
                anti_aliasing="77% unaligned FXAA + 23% aligned without FXAA; shared contributor fields",
                branches=[
                    dict(weight=0.77, realignment=False, fxaa=True),
                    dict(weight=0.23, realignment=True, fxaa=False),
                ],
                texture_residual_retained=True,
            ),
        )
        if mode == "xray":
            meta.update(
                recipe_id=XRAY_PLUS_RECIPE,
                recipe="X-Ray / unsharpened DCCI, middle-band strong GCS 12% and Jinc4 Sharp 6%, one shared 80-reference Local Laplacian field with bounded grain-only refinement, reduced middle-detail strength and donor-only lighting",
                xray=dict(
                    dcci=0.82,
                    gcs=0.12,
                    jinc4=0.06,
                    gcs_psf=0.60,
                    laplacian_bins=XRAY_LAPLACIAN_BINS,
                    fine_strength=XRAY_FINE_STRENGTH,
                    blend_scope="middle_band",
                    grain_gcs_psf=0.35,
                    extra_donor_unsharp=False,
                    jinc_anti_ringing=0.9,
                    jinc_overshoot_limit="min(1 code value, 2% local source range)",
                    gcs_overshoot_suppression=0.25,
                    gcs_overshoot_scope="middle donor only; grain detection unchanged",
                    gcs_grain_guard="Source-guided periodicity suppression; unchanged support and weights",
                    contrast_gain=0.05,
                    jinc_detail_contrast=XRAY_JINC_CONTRAST,
                    jinc_tone_contrast=0.25,
                    jinc_whites_lift=0.31,
                    shadow_lift=2.0,
                    lighting_smoothing=2.0,
                    grain_laplacian_bins=80,
                    grain_refinement=xray_grain_contract(),
                    near_white=0.996,
                    whites_lift=0.28,
                    detail_strength=XRAY_DETAIL_STRENGTH,
                    lighting_strength=XRAY_LIGHTING_STRENGTH,
                    lighting_scope="GCS 12% and Jinc4 6% only",
                    global_tone=False,
                ),
            )
        if mode == "xray":
            meta.update(
                self_examples=dict(
                    scope="fine_grain_only",
                    grain_highpass_radius=0.8,
                    existing_texture_gate=True,
                    vibrance=dict(layer="dcci", strength=0.02),
                    weights=[2 / 3, 1 / 6, 1 / 6],
                    recipe="bounded-self-grain-v3",
                    full_paper_implementation=False,
                    source_only=True,
                ),
                recipe="X-Ray / shared 80-reference Local Laplacian; bounded grain sharpness, contrast, source-chroma vibrance and veil compensation; source matching; donor-weighted tone",
            )
        save_json(folder / "result.json", meta)
        status_file(folder, "Complete", 1, **meta)
        return meta
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        del total, weights
        for p in [acc_path, weight_path]:
            p.unlink(missing_ok=True)


# 03a — JPEG export. Isolated from restoration; publish only a complete file.
def jpeg_export(request):
    source = Path(request["source"]).resolve()
    destination = Path(request["destination"]).resolve()
    quality = request["quality"]
    size = request["size"]
    protected = {source, *[Path(p).resolve() for p in request.get("protected", [])]}
    if destination in protected:
        raise ValueError("Choose a new JPEG destination; original and working files are preserved.")
    if destination.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("Choose a .jpg or .jpeg destination.")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")
    if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError("JPEG dimensions must be positive whole pixels.")
    token = request.get("token", uuid.uuid4().hex)
    if not re.fullmatch("[0-9a-f]{32}", token):
        raise ValueError("Invalid export identifier")
    temp = destination.with_name("." + destination.name + "." + token + ".tmp")
    try:
        with Image.open(source) as original:
            w, h = original.size
            ow, oh = size
            if ow > w or oh > h:
                raise ValueError(
                    "JPEG export can reduce dimensions. Increase the restoration scale to make a larger image."
                )
            if ow * oh > MAX_OUTPUT:
                raise ValueError("JPEG output exceeds the pixel limit.")
            if abs(oh - ow * h / w) > 0.51 and abs(ow - oh * w / h) > 0.51:
                raise ValueError("Keep the original aspect ratio.")
            im = original if original.mode == "RGB" else original.convert("RGB")
            if im.size != tuple(size):
                im = im.resize(tuple(size), Image.Resampling.LANCZOS, reducing_gap=3)
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            im.save(
                temp,
                format="JPEG",
                quality=quality,
                subsampling=0,
                optimize=True,
                progressive=True,
                icc_profile=profile,
            )
            if im is not original:
                im.close()
        # Decode before replacing an existing destination. A failed encoder
        # cannot leave a partial file or destroy the previous export.
        with Image.open(temp) as check:
            if check.format != "JPEG" or check.size != tuple(size):
                raise ValueError("JPEG verification failed")
            check.load()
        replace_file(temp, destination)
        return dict(
            path=str(destination),
            size=size,
            quality=quality,
            bytes=destination.stat().st_size,
            subsampling="4:4:4",
        )
    finally:
        temp.unlink(missing_ok=True)


def command(*args):
    return (
        [sys.executable, *args]
        if getattr(sys, "frozen", False)
        else [sys.executable, str(Path(__file__).resolve()), *args]
    )


def terminate_tree(proc):
    if proc and proc.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=15,
            )
        else:
            proc.terminate()


def contain_worker(proc):
    """Windows job object: all native descendants die if the supervisor exits."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes as wt

    class Basic(ctypes.Structure):
        _fields_ = [
            ("ProcessTime", ctypes.c_int64),
            ("JobTime", ctypes.c_int64),
            ("Flags", wt.DWORD),
            ("Min", ctypes.c_size_t),
            ("Max", ctypes.c_size_t),
            ("Count", wt.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("Priority", wt.DWORD),
            ("Scheduling", wt.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in ["ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes"]
        ]

    class Extended(ctypes.Structure):
        _fields_ = [
            ("Basic", Basic),
            ("IO", IO),
            ("ProcessMemory", ctypes.c_size_t),
            ("JobMemory", ctypes.c_size_t),
            ("PeakProcess", ctypes.c_size_t),
            ("PeakJob", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    kernel.CreateJobObjectW.restype = wt.HANDLE
    kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.Basic.Flags = 0x2000 | 0x200
    info.JobMemory = memory_budget()[1]
    if (
        not handle
        or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info))
        or not kernel.AssignProcessToJobObject(handle, int(proc._handle))
    ):
        err = ctypes.get_last_error()
        if handle:
            kernel.CloseHandle(handle)
        terminate_tree(proc)
        raise OSError(err, "Could not isolate the image worker")
    return lambda: kernel.CloseHandle(handle)


# 04a — Processing time and comparison navigation.
def duration_text(seconds):
    seconds = max(0, int(float(seconds or 0)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class ProcessingClock:
    """Monotonic active time. Checkpoints survive restart without counting downtime."""

    def __init__(self):
        self.started = {}

    def start(self, job):
        self.started.setdefault(job["id"], time.monotonic())

    def elapsed(self, job):
        start = self.started.get(job["id"])
        return float(job.get("elapsed_seconds", 0)) + (
            max(0.0, time.monotonic() - start) if start is not None else 0.0
        )

    def checkpoint(self, job, stop=False):
        now = time.monotonic()
        start = self.started.get(job["id"])
        if start is None:
            return False
        job["elapsed_seconds"] = float(job.get("elapsed_seconds", 0)) + max(0.0, now - start)
        if stop:
            self.started.pop(job["id"], None)
        else:
            self.started[job["id"]] = now
        return True


def comparison_view(image_size, viewport, zoom=1.0, centre=(0.5, 0.5)):
    """Return destination bounds and matching normalized source rectangle."""
    w, h = image_size
    vw, vh = (max(1, int(v)) for v in viewport)
    scale = min(vw / w, vh / h) * max(1.0, float(zoom))
    dw, dh = w * scale, h * scale
    iw, ih = min(vw, max(1, round(dw))), min(vh, max(1, round(dh)))
    rw, rh = min(1.0, vw / dw), min(1.0, vh / dh)
    left = max(0.0, min(1.0 - rw, float(centre[0]) - rw / 2))
    top = max(0.0, min(1.0 - rh, float(centre[1]) - rh / 2))
    return ((vw - iw) // 2, (vh - ih) // 2, iw, ih), (left, top, left + rw, top + rh)


def comparison_crop(im, size, rect, resample=Image.Resampling.LANCZOS):
    w, h = im.size
    box = tuple(v * d for v, d in zip(rect, (w, h, w, h)))
    return im.resize(tuple(size), resample, box=box)


def comparison_region(request):
    """Decode large images outside the UI; return only the visible image area."""
    folder = Path(request["folder"])
    dest = Path(request["destination"])
    size = request["size"]
    rect = request["rect"]
    if (
        len(size) != 2
        or any(type(v) is not int or v < 1 for v in size)
        or size[0] * size[1] > 8_000_000
    ):
        raise ValueError("Comparison viewport is too large")
    if (
        len(rect) != 4
        or not all(math.isfinite(v) and 0 <= v <= 1 for v in rect)
        or rect[0] >= rect[2]
        or rect[1] >= rect[3]
    ):
        raise ValueError("Invalid comparison region")
    dest.mkdir(parents=True, exist_ok=True)
    with Image.open(folder / "result.png") as im:
        output_size = im.size
        if im.width * im.height > MAX_OUTPUT:
            raise ValueError("Image exceeds comparison limit")
        comparison_crop(im, size, rect).convert("RGB").save(dest / "after.png", compress_level=1)
    source, _ = decode(Path(read_json(folder / "request.json")["input"]))
    source = source.convert("RGB")
    if request.get("bicubic"):
        source = source.resize(output_size, Image.Resampling.BICUBIC)
    comparison_crop(
        source,
        size,
        rect,
        Image.Resampling.LANCZOS if request.get("bicubic") else Image.Resampling.BICUBIC,
    ).save(dest / "before.png", compress_level=1)
    save_json(dest / "ready.json", dict(size=size, rect=rect))


class Studio:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.proc = None
        self.active = None
        self.stop = False
        self.jobs = []
        self.clock = ProcessingClock()
        for p in sorted(root.glob("*/job.json")):
            job = read_json(p)
            if not isinstance(job, dict):
                continue
            if "elapsed_seconds" not in job:
                job["elapsed_seconds"] = read_json(p.parent / "result.json", {}).get("seconds", 0)
            if job.get("state") in ["running", "queued"]:
                job["state"] = "paused"
                job["message"] = "Session interrupted. Resume saved tiles."
            if job.get("export_state") in ["running", "queued"]:
                job["export_state"] = "paused"
            if job.get("comparison_state") in ["running", "queued"]:
                job["comparison_state"] = "paused"
            self.jobs.append(job)
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def persist(self, job):
        save_json(self.root / job["id"] / "job.json", job)

    def snapshot(self):
        with self.lock:
            result = []
            for job in self.jobs:
                j = dict(job)
                j["elapsed_seconds"] = self.clock.elapsed(job)
                j["progress"] = read_json(self.root / j["id"] / "progress.json", {})
                result.append(j)
            return result

    def add(self, path, scale, svg, name=None, mode="max", device="auto", workers=0):
        if device not in ("auto", "cpu") or not isinstance(workers, int) or not 0 <= workers <= 6:
            raise ValueError("Device must be auto/cpu and workers from 0 to 6")
        mode = normalize_mode(mode)
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() not in FORMATS:
            raise ValueError("Choose a supported image file.")
        if path.stat().st_size > MAX_FILE:
            raise ValueError("Image exceeds 400 MB.")
        if not math.isfinite(scale) or not 1 <= scale <= 16:
            raise ValueError("Scale must be between 1 and 16.")
        ident = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        folder = self.root / ident
        folder.mkdir()
        destination = folder / ("input" + path.suffix.lower())
        shutil.copyfile(path, destination)
        job = dict(
            id=ident,
            name=name or path.name,
            scale=scale,
            svg=svg,
            mode=mode,
            device=device,
            workers=workers,
            state="queued",
            created=time.time(),
            elapsed_seconds=0.0,
            message="Waiting",
        )
        save_json(
            folder / "request.json",
            dict(
                input=str(destination),
                folder=str(folder),
                scale=scale,
                svg=svg,
                mode=mode,
                device=device,
                workers=workers,
            ),
        )
        with self.lock:
            self.jobs.append(job)
            self.persist(job)
        return job

    def act(self, ident, action):
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == ident), None)
            if job is None:
                raise ValueError("Job not found")
            if action == "cancel":
                self.clock.checkpoint(job, stop=True)
                job["state"] = "paused"
                job["message"] = "Paused. Completed tiles are saved."
                if self.active == ident:
                    terminate_tree(self.proc)
            elif action == "resume" and job["state"] in ["paused", "failed"]:
                job["state"] = "queued"
                job["message"] = "Resuming saved work"
            elif (
                action == "export-svg"
                and job["state"] == "complete"
                and job.get("export_state") != "running"
            ):
                job["export_state"] = "queued"
                job["export_message"] = "Waiting for vector export"
            elif (
                action == "compare-bicubic"
                and job["state"] == "complete"
                and job.get("comparison_state") not in ["queued", "running", "complete"]
            ):
                job["comparison_state"] = "queued"
            else:
                raise ValueError("This action is not available for this job.")
            self.persist(job)

    def work(self):
        while not self.stop:
            with self.lock:
                job = next((j for j in self.jobs if j["state"] == "queued"), None)
                if job:
                    job["state"] = "running"
                    self.active = job["id"]
                    self.clock.start(job)
                    self.persist(job)
            if not job:
                with self.lock:
                    export = next((j for j in self.jobs if j.get("export_state") == "queued"), None)
                if export:
                    self.export(export)
                else:
                    with self.lock:
                        comparison = next(
                            (j for j in self.jobs if j.get("comparison_state") == "queued"), None
                        )
                    if comparison:
                        self.comparison(comparison)
                    else:
                        time.sleep(0.3)
                continue
            folder = self.root / job["id"]
            error = None
            release = None
            started = time.time()
            try:
                (folder / "error.json").unlink(missing_ok=True)
                with (folder / "worker.log").open("a", encoding="utf-8") as log:
                    with self.lock:
                        if job["state"] != "running":
                            continue
                        self.proc = subprocess.Popen(
                            command("--worker", str(folder / "request.json")),
                            stdout=log,
                            stderr=log,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        release = contain_worker(self.proc)
                    while self.proc.poll() is None:
                        time.sleep(0.5)
                        with self.lock:
                            if self.clock.checkpoint(job):
                                self.persist(job)
                        progress = read_json(folder / "progress.json", {})
                        # This watchdog also handles an unresponsive Python stage.
                        if time.time() - max(started, progress.get("updated", started)) > 1200:
                            terminate_tree(self.proc)
                            error = "Processing stopped after 20 minutes without stage progress. Saved tiles can be resumed."
                            break
                    if self.proc.returncode != 0 and not error:
                        failure = read_json(folder / "error.json", {})
                        error = failure.get(
                            "error", "The worker stopped unexpectedly. Your saved tiles are intact."
                        )
                        recover = not failure or failure.get("kind") in (
                            "MemoryError",
                            "TimeoutExpired",
                        )
                        with self.lock:
                            recover = recover and job["state"] == "running" and not self.stop
                        if recover:
                            if release:
                                release()
                                release = None
                            save_json(
                                folder / "recovery.json",
                                dict(reason=error, action="One serial CPU retry"),
                            )
                            status_file(folder, "Retrying on CPU; saved tiles preserved")
                            (folder / "error.json").unlink(missing_ok=True)
                            safe_env = dict(os.environ, UNTORN_SAFE_WORKER="1")
                            with self.lock:
                                if job["state"] != "running" or self.stop:
                                    continue
                                self.proc = subprocess.Popen(
                                    command("--worker", str(folder / "request.json")),
                                    stdout=log,
                                    stderr=log,
                                    env=safe_env,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                )
                                release = contain_worker(self.proc)
                            retry_started = time.time()
                            while self.proc.poll() is None:
                                time.sleep(0.5)
                                with self.lock:
                                    if self.clock.checkpoint(job):
                                        self.persist(job)
                                progress = read_json(folder / "progress.json", {})
                                if (
                                    time.time()
                                    - max(retry_started, progress.get("updated", retry_started))
                                    > 1200
                                ):
                                    terminate_tree(self.proc)
                                    break
                            error = (
                                None
                                if self.proc.returncode == 0
                                else read_json(folder / "error.json", {}).get(
                                    "error", "CPU recovery stopped. Saved tiles remain available."
                                )
                            )
            except Exception as exc:
                error = str(exc)
            finally:
                if release:
                    release()
            with self.lock:
                self.clock.checkpoint(job, stop=True)
                if job["state"] == "running":
                    job["state"] = "failed" if error else "complete"
                    job["message"] = error or "Ready to export"
                if job["state"] == "complete" and job.get("svg"):
                    job["export_state"] = "queued"
                self.persist(job)
                self.proc = None
                self.active = None

    def export(self, job):
        folder = self.root / job["id"]
        release = None
        error = None
        with self.lock:
            job["export_state"] = "running"
            self.active = job["id"]
            self.persist(job)
        try:
            (folder / "svg-error.json").unlink(missing_ok=True)
            with (folder / "svg-worker.log").open("a") as log:
                self.proc = subprocess.Popen(
                    command("--svg-worker", str(folder)),
                    stdout=log,
                    stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                release = contain_worker(self.proc)
                try:
                    code = self.proc.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    terminate_tree(self.proc)
                    raise ValueError("Vector export timed out; PNG is unchanged.")
                if code:
                    error = read_json(folder / "svg-error.json", {}).get(
                        "error", "Vector export stopped. PNG is unchanged."
                    )
        except Exception as exc:
            error = str(exc)
        finally:
            if release:
                release()
        with self.lock:
            job["export_state"] = "failed" if error else "complete"
            job["export_message"] = error or "Vector SVG ready"
            self.persist(job)
            self.active = None
            self.proc = None

    def comparison(self, job):
        folder = self.root / job["id"]
        release = None
        error = None
        with self.lock:
            job["comparison_state"] = "running"
            self.active = job["id"]
            self.persist(job)
        try:
            with (folder / "comparison.log").open("a") as log:
                self.proc = subprocess.Popen(
                    command("--comparison-worker", str(folder)),
                    stdout=log,
                    stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                release = contain_worker(self.proc)
                try:
                    code = self.proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    terminate_tree(self.proc)
                    raise ValueError("Bicubic comparison timed out; main PNG is unchanged.")
                if code:
                    raise ValueError("Bicubic comparison failed; main PNG is unchanged.")
        except Exception as exc:
            error = str(exc)
        finally:
            if release:
                release()
        with self.lock:
            job["comparison_state"] = "failed" if error else "complete"
            job["comparison_message"] = error
            self.persist(job)
            self.active = None
            self.proc = None

    def shutdown(self):
        self.stop = True
        with self.lock:
            if self.active:
                job = next(j for j in self.jobs if j["id"] == self.active)
                self.clock.checkpoint(job, stop=True)
                if job.get("export_state") == "running":
                    job["export_state"] = "paused"
                elif job.get("comparison_state") == "running":
                    job["comparison_state"] = "paused"
                else:
                    job["state"] = "paused"
                    job["message"] = "Session closed; resume saved work."
                self.persist(job)
            terminate_tree(self.proc)


def workspace_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "workspace.lock").open("a+b")
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            raise ValueError("This output workspace is already open in another Untorn session.")
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b"0")
        handle.flush()
    return handle


def friendly_error(exc):
    if isinstance(exc, PermissionError):
        return "Windows locked a working file. Your original and completed tiles are safe; Resume to retry."
    if type(exc).__name__ == "UnidentifiedImageError":
        return "This file could not be decoded as an image."
    if isinstance(exc, subprocess.CalledProcessError):
        return f"A restoration helper stopped (code {exc.returncode}). Saved tiles are intact; details are in worker.log."
    if isinstance(exc, MemoryError):
        return (
            "The worker reached its memory allowance. Saved tiles are intact; try a smaller output."
        )
    return str(exc)


# ── Native desktop presentation ──────────────────────────────────────────────
def desktop(root):
    """Native wxWidgets window. No HTTP server, browser, or WebView runtime."""
    import wx
    import wx.lib.scrolledpanel
    import tempfile
    from PIL import ImageDraw

    app = wx.App(False)
    try:
        guard = workspace_lock(root)
    except ValueError as exc:
        wx.MessageBox(str(exc), "Untorn Engine", wx.OK | wx.ICON_INFORMATION)
        return
    studio = Studio(root)
    BG = "#0d0f12"
    PANEL = "#171a1f"
    LINE = "#30353c"
    INK = "#f3f1eb"
    MUTED = "#969da8"
    ACCENT = "#ef493c"
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-preview")

    def font(size=10, weight=wx.FONTWEIGHT_NORMAL):
        return wx.Font(wx.FontInfo(size).FaceName("Segoe UI").Weight(weight))

    def label(parent, text, size=10, colour=INK):
        w = wx.StaticText(parent, label=text)
        w.SetForegroundColour(colour)
        w.SetFont(font(size))
        return w

    def bitmap(pil):
        im = pil.convert("RGBA")
        return wx.Bitmap.FromBufferRGBA(im.width, im.height, im.tobytes())

    def thumb(path, size=(96, 96)):
        im, _ = decode(path)
        im.thumbnail((1600, 1000), Image.Resampling.LANCZOS)
        return im.convert("RGB")

    class RoundedPanel(wx.Panel):
        def __init__(self, parent):
            super().__init__(parent)
            self.SetBackgroundColour(PANEL)
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.Bind(wx.EVT_PAINT, self.paint_round)

        def paint_round(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(BG))
            dc.Clear()
            g = wx.GraphicsContext.Create(dc)
            w, h = self.GetClientSize()
            g.SetPen(wx.Pen(LINE))
            g.SetBrush(wx.Brush(PANEL))
            g.DrawRoundedRectangle(0.5, 0.5, w - 1, h - 1, self.FromDIP(18))

    class Button(wx.Control):
        def __init__(self, parent, text, action, primary=False, width=120, accent=False):
            super().__init__(
                parent, size=parent.FromDIP((width, 40)), style=wx.WANTS_CHARS | wx.BORDER_NONE
            )
            self.SetMinSize(self.GetSize())
            self.SetName(text)
            self.SetLabel(text)
            self.action = action
            self.primary = primary
            self.accent = accent
            self.hover = False
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.Bind(wx.EVT_PAINT, self.paint)
            self.Bind(wx.EVT_LEFT_UP, lambda e: self.fire())
            self.Bind(wx.EVT_KEY_DOWN, self.key)
            self.Bind(wx.EVT_ENTER_WINDOW, lambda e: self.hot(True))
            self.Bind(wx.EVT_LEAVE_WINDOW, lambda e: self.hot(False))
            self.Bind(wx.EVT_SET_FOCUS, lambda e: self.Refresh())
            self.Bind(wx.EVT_KILL_FOCUS, lambda e: self.Refresh())

        def hot(self, v):
            self.hover = v
            self.Refresh()

        def fire(self):
            if self.IsEnabled():
                self.SetFocus()
                self.action()

        def key(self, e):
            if e.AltDown() or e.ControlDown():
                e.Skip()
            elif e.GetKeyCode() in (wx.WXK_SPACE, wx.WXK_RETURN):
                self.fire()
            elif e.GetKeyCode() == wx.WXK_TAB:
                self.Navigate(not e.ShiftDown())
            else:
                e.Skip()

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(self.GetParent().GetBackgroundColour()))
            dc.Clear()
            g = wx.GraphicsContext.Create(dc)
            w, h = self.GetClientSize()
            on = self.IsEnabled()
            colour = (
                ("#ff5b4e" if self.hover else ACCENT)
                if self.accent
                else INK
                if self.primary
                else ("#343a43" if self.hover else "#242930")
            )
            if not on:
                colour = "#6b2e29" if self.accent else "#20242a"
            g.SetPen(wx.Pen(ACCENT if self.HasFocus() else colour))
            g.SetBrush(wx.Brush(colour))
            g.DrawRoundedRectangle(1, 1, w - 2, h - 2, self.FromDIP(10))
            g.SetFont(
                font(10, wx.FONTWEIGHT_MEDIUM),
                wx.Colour(
                    "#101318"
                    if self.primary and on and not self.accent
                    else INK
                    if on
                    else "#707781"
                ),
            )
            tw, th = g.GetTextExtent(self.GetLabel())
            g.DrawText(self.GetLabel(), (w - tw) / 2, (h - th) / 2)

        def SetLabel(self, v):
            super().SetLabel(v)
            self.SetName(v)
            self.Refresh()

        def Enable(self, v=True):
            r = super().Enable(v)
            self.Refresh()
            return r

    class Switch(wx.Control):
        def __init__(self, parent, choices, selected, callback):
            super().__init__(
                parent, size=parent.FromDIP((220, 42)), style=wx.WANTS_CHARS | wx.BORDER_NONE
            )
            self.SetMinSize(self.GetSize())
            self.choices = choices
            self.selected = selected
            self.callback = callback
            self.SetName(" / ".join(choices))
            self.SetLabel(" / ".join(choices))
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.Bind(wx.EVT_PAINT, self.paint)
            self.Bind(wx.EVT_LEFT_UP, self.click)
            self.Bind(wx.EVT_KEY_DOWN, self.key)
            self.Bind(wx.EVT_SET_FOCUS, lambda e: self.Refresh())
            self.Bind(wx.EVT_KILL_FOCUS, lambda e: self.Refresh())

        def choose(self, i):
            self.selected = i
            self.SetLabel(" / ".join(self.choices) + " · " + self.choices[i])
            self.Refresh()
            self.callback(i)

        def click(self, e):
            self.SetFocus()
            self.choose(
                min(
                    len(self.choices) - 1,
                    int(e.X / max(1, self.GetClientSize().width) * len(self.choices)),
                )
            )

        def key(self, e):
            if e.GetKeyCode() in (wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_SPACE):
                self.choose(
                    (self.selected + (-1 if e.GetKeyCode() == wx.WXK_LEFT else 1))
                    % len(self.choices)
                )
            elif e.GetKeyCode() == wx.WXK_TAB:
                self.Navigate(not e.ShiftDown())
            else:
                e.Skip()

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(self.GetParent().GetBackgroundColour()))
            dc.Clear()
            g = wx.GraphicsContext.Create(dc)
            w, h = self.GetClientSize()
            g.SetPen(wx.Pen(ACCENT if self.HasFocus() else LINE))
            g.SetBrush(wx.Brush("#101317"))
            g.DrawRoundedRectangle(1, 1, w - 2, h - 2, 12)
            cell = w / len(self.choices)
            g.SetPen(wx.TRANSPARENT_PEN)
            g.SetBrush(wx.Brush("#363b43"))
            g.DrawRoundedRectangle(self.selected * cell + 4, 4, cell - 8, h - 8, 9)
            for i, t in enumerate(self.choices):
                g.SetFont(
                    font(10, wx.FONTWEIGHT_MEDIUM), wx.Colour(INK if i == self.selected else MUTED)
                )
                tw, th = g.GetTextExtent(t)
                g.DrawText(t, i * cell + (cell - tw) / 2, (h - th) / 2)

    class PhotoPanel(wx.Panel):
        def __init__(self, parent):
            super().__init__(parent)
            self.SetBackgroundColour(PANEL)
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.photo = None
            self.cached = None
            self.dragging = False
            self.title = "Select an image to upscale"
            self.subtitle = "Drop an image here, or browse your computer."
            self.SetMinSize(self.FromDIP((420, 285)))
            self.Bind(wx.EVT_PAINT, self.paint)
            self.Bind(wx.EVT_SIZE, self.resize)

        def resize(self, e):
            self.cached = None
            self.Refresh()
            e.Skip()

        def set_photo(self, photo, title, subtitle):
            self.photo = photo
            self.title = title
            self.subtitle = subtitle
            self.cached = None
            self.Refresh()

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(BG))
            dc.Clear()
            w, h = self.GetClientSize()
            if w < 2 or h < 2:
                return
            if self.cached is None:
                base = Image.new("RGB", (w, h), PANEL)
                if self.photo:
                    cover = ImageOps.fit(self.photo, (w, h), Image.Resampling.BICUBIC)
                    base = Image.blend(base, cover, 0.5)  # Exactly 50% image opacity.
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).rounded_rectangle(
                    (0, 0, w - 1, h - 1), radius=self.FromDIP(22), fill=255
                )
                base.putalpha(mask)
                self.cached = bitmap(base)
            dc.DrawBitmap(self.cached, 0, 0, True)
            g = wx.GraphicsContext.Create(dc)
            g.SetPen(wx.Pen(ACCENT if self.dragging else LINE, 2 if self.dragging else 1))
            g.SetBrush(wx.TRANSPARENT_BRUSH)
            g.DrawRoundedRectangle(1, 1, w - 2, h - 2, self.FromDIP(22))
            # A compact dark scrim protects text without changing the backdrop opacity.
            pw = min(w - self.FromDIP(48), self.FromDIP(490))
            ph = self.FromDIP(113)
            g.SetPen(wx.TRANSPARENT_PEN)
            g.SetBrush(wx.Brush(wx.Colour(13, 15, 18, 205)))
            g.DrawRoundedRectangle((w - pw) / 2, (h - ph) / 2, pw, ph, 16)
            for text, size, dy, col in [
                (self.title, 17, -20, INK),
                (self.subtitle, 9, 17, "#c3c8cf"),
            ]:
                g.SetFont(font(size), wx.Colour(col))
                tw, th = g.GetTextExtent(text)
                while tw > pw - 26 and len(text) > 8:
                    text = text[:-2]
                    tw, th = g.GetTextExtent(text + "…")
                g.DrawText(
                    text if tw <= pw - 26 else text + "…", (w - tw) / 2, h / 2 + self.FromDIP(dy)
                )

    class Drop(wx.FileDropTarget):
        def __init__(self, callback, panel=None):
            super().__init__()
            self.callback = callback
            self.panel = panel

        def highlight(self, v):
            if self.panel:
                self.panel.dragging = v
                self.panel.Refresh()

        def OnDragOver(self, x, y, result):
            self.highlight(True)
            return wx.DragCopy

        def OnLeave(self):
            self.highlight(False)

        def OnDropFiles(self, x, y, files):
            self.highlight(False)
            self.callback(files)
            return True

    class ScaleSlider(wx.Control):
        def __init__(self, parent, callback):
            super().__init__(
                parent, size=parent.FromDIP((220, 32)), style=wx.WANTS_CHARS | wx.BORDER_NONE
            )
            self.SetMinSize(self.GetSize())
            self.value = 4.0
            self.callback = callback
            self.SetName("Scale slider")
            self.SetLabel("Scale slider 4×")
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.Bind(wx.EVT_PAINT, self.paint)
            self.Bind(wx.EVT_LEFT_DOWN, self.drag)
            self.Bind(wx.EVT_MOTION, self.drag)
            self.Bind(wx.EVT_KEY_DOWN, self.key)
            self.Bind(wx.EVT_SET_FOCUS, lambda e: self.Refresh())
            self.Bind(wx.EVT_KILL_FOCUS, lambda e: self.Refresh())

        def set(self, v):
            self.value = max(1.0, min(16.0, v))
            self.SetLabel(f"Scale slider {self.value:g}×")
            self.Refresh()

        def drag(self, e):
            if e.LeftIsDown():
                self.SetFocus()
                self.set(
                    round(
                        (
                            1
                            + 15
                            * max(0, min(1, (e.X - 9) / max(1, self.GetClientSize().width - 18)))
                        )
                        * 20
                    )
                    / 20
                )
                self.callback(self.value)

        def key(self, e):
            k = e.GetKeyCode()
            if k in (wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_HOME, wx.WXK_END):
                self.set(
                    1
                    if k == wx.WXK_HOME
                    else 16
                    if k == wx.WXK_END
                    else round(self.value + (0.05 if k == wx.WXK_RIGHT else -0.05), 2)
                )
                self.callback(self.value)
            elif k == wx.WXK_TAB:
                self.Navigate(not e.ShiftDown())
            else:
                e.Skip()

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(PANEL))
            dc.Clear()
            g = wx.GraphicsContext.Create(dc)
            w, h = self.GetClientSize()
            x = 9 + (w - 18) * (self.value - 1) / 15
            g.SetPen(wx.TRANSPARENT_PEN)
            g.SetBrush(wx.Brush("#3a3e46"))
            g.DrawRoundedRectangle(9, h / 2 - 2, w - 18, 4, 2)
            g.SetBrush(wx.Brush(ACCENT))
            g.DrawRoundedRectangle(9, h / 2 - 2, max(2, x - 9), 4, 2)
            g.DrawEllipse(x - 6, h / 2 - 6, 12, 12)
            if self.HasFocus():
                g.SetBrush(wx.TRANSPARENT_BRUSH)
                g.SetPen(wx.Pen(INK))
                g.DrawEllipse(x - 9, h / 2 - 9, 18, 18)

    class Progress(wx.Panel):
        def __init__(self, parent):
            super().__init__(parent, size=parent.FromDIP((-1, 3)))
            self.SetMinSize(self.GetSize())
            self.value = 0
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.Bind(wx.EVT_PAINT, self.paint)

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(PANEL))
            dc.Clear()
            w, h = self.GetClientSize()
            g = wx.GraphicsContext.Create(dc)
            g.SetPen(wx.TRANSPARENT_PEN)
            g.SetBrush(wx.Brush(ACCENT))
            g.DrawRoundedRectangle(0, 0, max(0, w * self.value), h, h / 2)

    class Thumbnail(wx.Panel):
        def __init__(self, parent):
            super().__init__(parent, size=parent.FromDIP((46, 46)))
            self.SetMinSize(self.GetSize())
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.bmp = None
            self.Bind(wx.EVT_PAINT, self.paint)

        def set_photo(self, photo):
            side = self.FromDIP(46)
            im = ImageOps.fit(photo, (side, side), Image.Resampling.LANCZOS).convert("RGBA")
            mask = Image.new("L", im.size)
            ImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, side - 1, side - 1), radius=self.FromDIP(10), fill=255
            )
            im.putalpha(mask)
            self.bmp = bitmap(im)
            self.Refresh()

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(PANEL))
            dc.Clear()
            if self.bmp:
                dc.DrawBitmap(self.bmp, 0, 0, True)

    class Wipe(wx.Panel):
        def __init__(self, parent, before, after):
            super().__init__(parent, style=wx.WANTS_CHARS | wx.BORDER_NONE)
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.before = before
            self.after = after
            self.position = 0.5
            self.cached = None
            self.dragging = False
            self.zoom = 1.0
            self.centre = (0.5, 0.5)
            self.on_view = None
            self.region = None
            self.SetName("Image comparison. Drag the divider, or use left and right arrow keys.")
            self.Bind(wx.EVT_PAINT, self.paint)
            self.Bind(wx.EVT_SIZE, self.resize)
            self.Bind(wx.EVT_LEFT_DOWN, self.begin)
            self.Bind(wx.EVT_MOTION, self.motion)
            self.Bind(wx.EVT_LEFT_UP, self.end)
            self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, lambda e: setattr(self, "dragging", False))
            self.Bind(wx.EVT_KEY_DOWN, self.key)

        def bounds(self):
            return self.view()[0]

        def view(self):
            return comparison_view(self.after.size, self.GetClientSize(), self.zoom, self.centre)

        def invalidate(self):
            self.cached = None
            self.region = None
            self.Refresh()
            if self.on_view:
                self.on_view()

        def resize(self, e):
            self.invalidate()
            e.Skip()

        def move(self, x):
            left, _, width, _ = self.bounds()
            self.position = max(0, min(1, (x - left) / width))
            self.Refresh()

        def begin(self, e):
            x, y, w, h = self.bounds()
            if x <= e.X <= x + w and y <= e.Y <= y + h:
                self.SetFocus()
                self.dragging = True
                self.CaptureMouse()
                self.move(e.X)

        def motion(self, e):
            x, y, w, h = self.bounds()
            inside = x <= e.X <= x + w and y <= e.Y <= y + h
            self.SetCursor(
                wx.Cursor(wx.CURSOR_SIZEWE if inside or self.dragging else wx.CURSOR_ARROW)
            )
            if self.dragging and e.LeftIsDown():
                self.move(e.X)

        def end(self, e):
            if self.dragging:
                self.move(e.X)
            self.dragging = False
            if self.HasCapture():
                self.ReleaseMouse()

        def key(self, e):
            k = e.GetKeyCode()
            if k in (wx.WXK_LEFT, wx.WXK_RIGHT):
                self.position = max(
                    0, min(1, self.position + (0.01 if k == wx.WXK_RIGHT else -0.01))
                )
                self.Refresh()
            elif k == wx.WXK_TAB:
                self.Navigate(not e.ShiftDown())
            else:
                e.Skip()

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(BG))
            dc.Clear()
            x, y, iw, ih = self.bounds()
            if not self.cached:
                rect = self.view()[1]
                a = comparison_crop(self.after, (iw, ih), rect)
                b = comparison_crop(self.before, (iw, ih), rect, Image.Resampling.BICUBIC)
                self.cached = (bitmap(a), bitmap(b))
            a, b = self.cached
            cut = round(iw * self.position)
            dc.DrawBitmap(a, x, y)
            if cut > 0:
                dc.SetClippingRegion(x, y, cut, ih)
                dc.DrawBitmap(b, x, y)
                dc.DestroyClippingRegion()
            blade = x + min(iw - 1, cut)
            dc.SetPen(wx.Pen(ACCENT, 2))
            dc.DrawLine(blade, y, blade, y + ih)
            g = wx.GraphicsContext.Create(dc)
            radius = self.FromDIP(15)
            cy = y + ih / 2
            g.SetPen(wx.Pen(ACCENT))
            g.SetBrush(wx.Brush(PANEL))
            g.DrawEllipse(blade - radius, cy - radius, radius * 2, radius * 2)
            g.SetPen(wx.Pen(INK, 1))
            g.StrokeLines([(blade - 4, cy - 4), (blade - 8, cy), (blade - 4, cy + 4)])
            g.StrokeLines([(blade + 4, cy - 4), (blade + 8, cy), (blade + 4, cy + 4)])

    class ImageMap(wx.Panel):
        def __init__(self, parent, photo, wipe, select):
            super().__init__(
                parent, size=parent.FromDIP((148, 116)), style=wx.WANTS_CHARS | wx.BORDER_NONE
            )
            self.SetMinSize(self.GetSize())
            self.photo = photo
            self.wipe = wipe
            self.select = select
            self.dragging = False
            self.bmp = None
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.SetName(
                "Image overview. Click or drag to select a zoom region. Arrow keys move the region."
            )
            self.SetToolTip("Click or drag to inspect an area")
            self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
            self.Bind(wx.EVT_PAINT, self.paint)
            self.Bind(wx.EVT_SIZE, lambda e: (setattr(self, "bmp", None), self.Refresh(), e.Skip()))
            self.Bind(wx.EVT_LEFT_DOWN, self.begin)
            self.Bind(wx.EVT_MOTION, self.move)
            self.Bind(wx.EVT_LEFT_UP, self.end)
            self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, lambda e: setattr(self, "dragging", False))
            self.Bind(wx.EVT_KEY_DOWN, self.key)

        def bounds(self):
            return comparison_view(self.photo.size, self.GetClientSize())[0]

        def point(self, e):
            x, y, w, h = self.bounds()
            self.select((max(0, min(1, (e.X - x) / w)), max(0, min(1, (e.Y - y) / h))))

        def begin(self, e):
            x, y, w, h = self.bounds()
            if x <= e.X <= x + w and y <= e.Y <= y + h:
                self.SetFocus()
                self.dragging = True
                self.CaptureMouse()
                self.point(e)

        def move(self, e):
            if self.dragging and e.LeftIsDown():
                self.point(e)

        def end(self, e):
            if self.dragging:
                self.point(e)
            self.dragging = False
            if self.HasCapture():
                self.ReleaseMouse()

        def key(self, e):
            k = e.GetKeyCode()
            directions = {
                wx.WXK_LEFT: (-0.04, 0),
                wx.WXK_RIGHT: (0.04, 0),
                wx.WXK_UP: (0, -0.04),
                wx.WXK_DOWN: (0, 0.04),
            }
            if k in directions:
                rect = self.wipe.view()[1]
                cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
                dx, dy = directions[k]
                self.select((max(0, min(1, cx + dx)), max(0, min(1, cy + dy))))
            elif k == wx.WXK_TAB:
                self.Navigate(not e.ShiftDown())
            else:
                e.Skip()

        def paint(self, e):
            dc = wx.AutoBufferedPaintDC(self)
            dc.SetBackground(wx.Brush(BG))
            dc.Clear()
            x, y, w, h = self.bounds()
            if self.bmp is None:
                self.bmp = bitmap(self.photo.resize((w, h), Image.Resampling.LANCZOS))
            dc.DrawBitmap(self.bmp, x, y)
            rect = self.wipe.view()[1]
            l, t, r, b = rect
            g = wx.GraphicsContext.Create(dc)
            g.SetPen(wx.Pen(ACCENT, 2))
            g.SetBrush(wx.Brush(wx.Colour(239, 73, 60, 22)))
            g.DrawRectangle(
                x + l * w + 1, y + t * h + 1, max(1, (r - l) * w - 2), max(1, (b - t) * h - 2)
            )

    class Compare(wx.Panel):
        def __init__(self, owner, job):
            super().__init__(owner)
            self.owner = owner
            self.job = job
            self.folder = root / job["id"]
            self.SetBackgroundColour(BG)
            self.region_proc = None
            self.region_release = None
            self.region_dir = None
            self.region_log = None
            self.pending = None
            self.closed = False
            p = self
            s = wx.BoxSizer(wx.VERTICAL)
            top = wx.BoxSizer(wx.HORIZONTAL)
            name = wx.StaticText(p, label=job["name"], style=wx.ST_ELLIPSIZE_END)
            name.SetForegroundColour(INK)
            name.SetFont(font(14))
            top.Add(name, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 20)
            top.Add(Button(p, "Save PNG", lambda: owner.save(job, "png")), 0, wx.RIGHT, 10)
            self.close_button = Button(p, "Close  ×", owner.close_compare, width=100)
            top.Add(self.close_button)
            s.Add(top, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 24)
            bar = wx.BoxSizer(wx.HORIZONTAL)
            self.choice = Switch(p, ["Original", "Bicubic"], 0, self.change)
            bar.Add(self.choice)
            self.note = label(p, "Drag the divider · Fit to window", 9, MUTED)
            bar.Add(self.note, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 18)
            bar.Add(label(p, "UNTORN DEFINITION GEN 1.0", 9, MUTED), 0, wx.ALIGN_CENTER_VERTICAL)
            s.Add(bar, 0, wx.EXPAND | wx.ALL, 24)
            with Image.open(self.folder / "original.png") as im:
                before = im.convert("RGB")
            with Image.open(self.folder / "preview.png") as im:
                after = im.convert("RGB")
            self.wipe = Wipe(p, before, after)
            body = wx.BoxSizer(wx.HORIZONTAL)
            body.Add(self.wipe, 1, wx.EXPAND | wx.RIGHT, 20)
            rail = wx.BoxSizer(wx.VERTICAL)
            rail.Add(label(p, "OVERVIEW", 8, MUTED), 0, wx.BOTTOM, 10)
            self.map = ImageMap(p, after, self.wipe, self.select_area)
            rail.Add(self.map, 0, wx.EXPAND)
            self.zoom_choice = Switch(p, ["Zoom", "Full"], 1, self.zoom_changed)
            self.zoom_choice.SetMinSize(self.FromDIP((148, 36)))
            self.zoom_choice.SetSize(self.FromDIP((148, 36)))
            rail.Add(self.zoom_choice, 0, wx.EXPAND | wx.TOP, 12)
            rail.Add(label(p, f"{job['scale']:g}× detail view", 9, INK), 0, wx.TOP, 12)
            rail.Add(label(p, "Click the overview\nto choose an area.", 9, MUTED), 0, wx.TOP, 8)
            elapsed = job.get(
                "elapsed_seconds", read_json(self.folder / "result.json", {}).get("seconds", 0)
            )
            rail.Add(label(p, "PROCESSING TIME", 8, MUTED), 0, wx.TOP, 28)
            rail.Add(label(p, duration_text(elapsed), 13), 0, wx.TOP, 6)
            body.Add(rail, 0, wx.EXPAND)
            s.Add(body, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 24)
            self.wipe.on_view = self.view_changed
            p.SetSizer(s)
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.poll, self.timer)
            self.timer.Start(150)
            self.Bind(wx.EVT_CHAR_HOOK, self.key)

        def select_area(self, centre):
            self.wipe.centre = centre
            self.zoom_choice.choose(0)

        def zoom_changed(self, i):
            self.wipe.zoom = float(self.job["scale"]) if i == 0 else 1.0
            self.wipe.invalidate()

        def view_changed(self):
            if self.closed:
                return
            self.map.Refresh()
            self.region_cleanup()
            self.pending = None
            self.note.SetLabel(
                "Drag the divider · "
                + (
                    f"{self.job['scale']:g}× zoom"
                    if self.zoom_choice.selected == 0
                    else "Full image"
                )
            )
            if self.zoom_choice.selected == 0:
                self.pending = time.monotonic() + 0.25

        def region_cleanup(self):
            if self.region_proc is not None:
                if self.region_proc.poll() is None:
                    self.region_proc.kill()
                    self.region_proc.wait(timeout=2)
                self.region_proc = None
            if self.region_release:
                self.region_release()
                self.region_release = None
            if self.region_log:
                self.region_log.close()
                self.region_log = None
            if self.region_dir:
                shutil.rmtree(self.region_dir, ignore_errors=True)
                self.region_dir = None

        def region_start(self):
            self.pending = None
            try:
                self.region_dir = Path(tempfile.mkdtemp(prefix="untorn-compare-"))
                bounds, rect = self.wipe.view()
                size = list(bounds[2:])
                request = self.region_dir / "request.json"
                save_json(
                    request,
                    dict(
                        folder=str(self.folder),
                        destination=str(self.region_dir),
                        size=size,
                        rect=rect,
                        bicubic=bool(self.choice.selected),
                    ),
                )
                self.region_log = (self.region_dir / "worker.log").open("w")
                self.region_proc = subprocess.Popen(
                    command("--comparison-region-worker", str(request)),
                    stdout=self.region_log,
                    stderr=self.region_log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self.region_release = contain_worker(self.region_proc)
                self.region_started = time.monotonic()
                self.note.SetLabel("Drag the divider · Loading output detail…")
            except Exception:
                self.region_cleanup()
                self.note.SetLabel("Detail unavailable · Showing preview")

        def key(self, e):
            if e.GetKeyCode() == wx.WXK_ESCAPE:
                self.owner.close_compare()
            else:
                e.Skip()

        def change(self, i):
            if self.choice.selected == 1 and not (self.folder / "bicubic-preview.png").exists():
                self.note.SetLabel("Preparing bicubic comparison…")
                self.choice.Disable()
                try:
                    studio.act(self.job["id"], "compare-bicubic")
                except ValueError as exc:
                    self.note.SetLabel(str(exc))
                    self.choice.Enable()
            else:
                self.load()

        def load(self):
            path = self.folder / ("bicubic-preview.png" if self.choice.selected else "original.png")
            if path.exists():
                with Image.open(path) as im:
                    self.wipe.before = im.convert("RGB")
                self.choice.Enable()
                self.wipe.invalidate()

        def poll(self, e):
            if self.pending is not None and time.monotonic() >= self.pending:
                self.region_start()
            if self.region_proc is not None:
                code = self.region_proc.poll()
                if code is not None:
                    try:
                        if code or not (self.region_dir / "ready.json").exists():
                            raise ValueError("Comparison detail failed")
                        with Image.open(self.region_dir / "after.png") as im:
                            a = bitmap(im)
                        with Image.open(self.region_dir / "before.png") as im:
                            b = bitmap(im)
                        self.wipe.cached = (a, b)
                        self.wipe.Refresh()
                        self.note.SetLabel(
                            f"Drag the divider · {self.job['scale']:g}× zoom · Output detail"
                        )
                    except Exception:
                        self.note.SetLabel("Detail unavailable · Showing preview")
                    finally:
                        self.region_cleanup()
                elif time.monotonic() - self.region_started > 120:
                    self.region_cleanup()
                    self.note.SetLabel("Detail timed out · Showing preview")
            if not self.choice.IsEnabled():
                j = next((j for j in studio.snapshot() if j["id"] == self.job["id"]), {})
                if j.get("comparison_state") == "complete":
                    self.load()
                elif j.get("comparison_state") == "failed":
                    self.note.SetLabel("Bicubic comparison failed. Select Original or retry.")
                    self.choice.Enable()

        def stop(self):
            self.closed = True
            self.timer.Stop()
            self.pending = None
            self.region_cleanup()
            if self.wipe.HasCapture():
                self.wipe.ReleaseMouse()
            if self.map.HasCapture():
                self.map.ReleaseMouse()

    class JpegDialog(wx.Dialog):
        def __init__(self, owner, job):
            super().__init__(owner, title="Export JPEG", style=wx.DEFAULT_DIALOG_STYLE)
            self.owner = owner
            self.job = job
            self.proc = None
            self.release = None
            self.log = None
            self.scratch = None
            self.temp = None
            self.syncing = False
            self.SetBackgroundColour(PANEL)
            self.SetForegroundColour(INK)
            meta = read_json(root / job["id"] / "result.json", {})
            self.original = tuple(meta["output_size"])
            w, h = self.original
            s = wx.BoxSizer(wx.VERTICAL)
            s.Add(label(self, "Export JPEG", 17), 0, wx.BOTTOM, 6)
            s.Add(label(self, f"{w} × {h} px · aspect ratio locked", 9, MUTED), 0, wx.BOTTOM, 24)
            self.quality_label = label(self, "Quality · 95%", 11)
            s.Add(self.quality_label, 0, wx.BOTTOM, 6)
            self.quality = wx.Slider(
                self, value=95, minValue=1, maxValue=100, style=wx.SL_HORIZONTAL
            )
            self.quality.SetName("JPEG quality")
            self.quality.Bind(wx.EVT_SLIDER, self.quality_changed)
            s.Add(self.quality, 0, wx.EXPAND)
            s.Add(
                label(self, "Lower quality → smaller file. JPEG is lossy.", 9, MUTED),
                0,
                wx.TOP | wx.BOTTOM,
                8,
            )
            s.Add(label(self, "Image size", 11), 0, wx.TOP | wx.BOTTOM, 10)
            self.preset = wx.Choice(
                self, choices=["Full size · 100%", "75%", "50%", "25%", "Custom"]
            )
            self.preset.SetSelection(0)
            self.preset.SetName("JPEG size preset")
            self.preset.Bind(wx.EVT_CHOICE, self.preset_changed)
            s.Add(self.preset, 0, wx.EXPAND | wx.BOTTOM, 12)
            row = wx.BoxSizer(wx.HORIZONTAL)
            self.width = wx.TextCtrl(self, value=str(w), size=self.FromDIP((155, 32)))
            self.height = wx.TextCtrl(self, value=str(h), size=self.FromDIP((155, 32)))
            for control, name in [(self.width, "Width"), (self.height, "Height")]:
                control.SetBackgroundColour("#101317")
                control.SetForegroundColour(INK)
                control.SetFont(font(11))
                control.SetName("JPEG " + name.lower())
                column = wx.BoxSizer(wx.VERTICAL)
                column.Add(label(self, name + " (px)", 9, MUTED), 0, wx.BOTTOM, 5)
                column.Add(control, 1, wx.EXPAND)
                row.Add(column, 1, wx.RIGHT, 12 if name == "Width" else 0)
            self.width.Bind(wx.EVT_TEXT, lambda e: self.dimension_changed("width"))
            self.height.Bind(wx.EVT_TEXT, lambda e: self.dimension_changed("height"))
            s.Add(row, 0, wx.EXPAND | wx.BOTTOM, 12)
            self.status = label(self, "Full colour detail · 4:4:4 JPEG", 9, MUTED)
            self.status.SetMinSize(self.FromDIP((370, 42)))
            s.Add(self.status, 0, wx.EXPAND | wx.BOTTOM, 14)
            actions = wx.BoxSizer(wx.HORIZONTAL)
            actions.AddStretchSpacer()
            self.cancel = Button(self, "Cancel", self.close_dialog, width=85)
            actions.Add(self.cancel, 0, wx.RIGHT, 8)
            self.save_button = Button(self, "Save JPEG", self.save, primary=True, width=120)
            actions.Add(self.save_button)
            s.Add(actions, 0, wx.EXPAND)
            wrap = wx.BoxSizer(wx.VERTICAL)
            wrap.Add(s, 1, wx.EXPAND | wx.ALL, 24)
            self.SetSizerAndFit(wrap)
            self.CentreOnParent()
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.poll, self.timer)
            self.Bind(wx.EVT_CLOSE, lambda e: self.close_dialog())
            self.Bind(wx.EVT_CHAR_HOOK, self.key)

        def key(self, e):
            if e.GetKeyCode() == wx.WXK_ESCAPE:
                self.close_dialog()
            else:
                e.Skip()

        def quality_changed(self, e):
            self.quality_label.SetLabel(f"Quality · {self.quality.GetValue()}%")

        def preset_changed(self, e):
            i = self.preset.GetSelection()
            if i < 4:
                f = [1, 0.75, 0.5, 0.25][i]
                w, h = self.original
                self.width.ChangeValue(str(max(1, int(w * f + 0.5))))
                self.height.ChangeValue(str(max(1, int(h * f + 0.5))))

        def dimension_changed(self, which):
            try:
                w, h = self.original
                v = int((self.width if which == "width" else self.height).GetValue())
                limit = w if which == "width" else h
                if not 1 <= v <= limit:
                    return
                other = self.height if which == "width" else self.width
                other.ChangeValue(
                    str(max(1, int(v * (h / w if which == "width" else w / h) + 0.5)))
                )
                self.preset.SetSelection(4)
            except ValueError:
                pass

        def status_text(self, text):
            self.status.SetLabel(text)
            self.status.Wrap(self.FromDIP(370))
            self.Layout()

        def save(self):
            try:
                size = [int(self.width.GetValue()), int(self.height.GetValue())]
                if any(v < 1 or v > m for v, m in zip(size, self.original)):
                    raise ValueError("Enter dimensions between 1 px and the completed output size.")
            except ValueError as exc:
                self.status_text(str(exc))
                return
            with wx.FileDialog(
                self,
                "Save JPEG",
                defaultFile=Path(self.job["name"]).stem + "-untorn.jpg",
                wildcard="JPEG image|*.jpg;*.jpeg",
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            ) as dialog:
                if dialog.ShowModal() != wx.ID_OK:
                    return
                destination = Path(dialog.GetPath())
            if not destination.suffix:
                destination = destination.with_suffix(".jpg")
            token = uuid.uuid4().hex
            self.scratch = root / self.job["id"] / "exports" / token
            original = read_json(root / self.job["id"] / "request.json", {}).get("input")
            request = dict(
                source=str(root / self.job["id"] / "result.png"),
                destination=str(destination),
                quality=self.quality.GetValue(),
                size=size,
                protected=[original] if original else [],
                token=token,
            )
            self.temp = destination.with_name("." + destination.name + "." + token + ".tmp")
            try:
                save_json(self.scratch / "request.json", request)
                self.log = (self.scratch / "worker.log").open("w")
                self.proc = subprocess.Popen(
                    command("--jpeg-worker", str(self.scratch / "request.json")),
                    stdout=self.log,
                    stderr=self.log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self.release = contain_worker(self.proc)
                self.started = time.monotonic()
                for c in [self.quality, self.preset, self.width, self.height, self.save_button]:
                    c.Disable()
                self.status_text("Encoding JPEG…")
                self.timer.Start(200)
            except Exception as exc:
                if self.log:
                    self.log.close()
                    self.log = None
                self.status_text(str(exc))

        def release_worker(self):
            if self.release:
                self.release()
                self.release = None
            if self.log:
                self.log.close()
                self.log = None

        def poll(self, e):
            if self.proc is None:
                return
            code = self.proc.poll()
            if code is None:
                if time.monotonic() - self.started > 600:
                    self.proc.kill()
                    self.status_text("Export timed out. Try a smaller image.")
                    return
                return
            self.timer.Stop()
            self.release_worker()
            self.proc = None
            for c in [self.quality, self.preset, self.width, self.height, self.save_button]:
                c.Enable()
            result = read_json(self.scratch / "result.json")
            if code == 0 and result:
                w, h = result["size"]
                length = result["bytes"]
                size_text = (
                    f"{length / 1024:.1f} KB" if length < 1024**2 else f"{length / 1024**2:.2f} MB"
                )
                message = f"Saved · {w} × {h} px · {size_text}"
                self.status_text(message)
                self.owner.notice(message)
                self.cancel.SetLabel("Close")
            else:
                self.status_text(
                    read_json(self.scratch / "error.json", {}).get(
                        "error", "JPEG export stopped. The PNG is intact."
                    )
                )
            self.cleanup_temp()

        def cleanup_temp(self):
            path = self.temp
            if path:

                def clean():
                    for _ in range(20):
                        try:
                            path.unlink(missing_ok=True)
                            return
                        except OSError:
                            time.sleep(0.1)

                threading.Thread(target=clean, daemon=True).start()

        def close_dialog(self):
            self.timer.Stop()
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()
            self.release_worker()
            self.cleanup_temp()
            self.EndModal(wx.ID_CANCEL)

    class JobCard(RoundedPanel):
        def __init__(self, parent, owner, job):
            super().__init__(parent)
            self.SetBackgroundColour(PANEL)
            self.owner = owner
            self.job = job
            self.meta = None
            self.SetMinSize(self.FromDIP((-1, 108)))
            row = wx.BoxSizer(wx.HORIZONTAL)
            self.thumb = Thumbnail(self)
            row.Add(self.thumb, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 16)
            words = wx.BoxSizer(wx.VERTICAL)
            self.name = label(self, job["name"], 11)
            self.name.SetMaxSize(self.FromDIP((350, -1)))
            words.Add(self.name, 0, wx.BOTTOM, 6)
            self.detail = label(self, "", 9, MUTED)
            words.Add(self.detail)
            self.resolution = label(self, "", 9, INK)
            words.Add(self.resolution, 0, wx.TOP, 5)
            row.Add(words, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
            self.actions = wx.BoxSizer(wx.HORIZONTAL)
            controls = wx.BoxSizer(wx.VERTICAL)
            controls.Add(self.actions, 0, wx.ALIGN_RIGHT)
            self.elapsed = label(self, "", 9, MUTED)
            controls.Add(self.elapsed, 0, wx.ALIGN_RIGHT | wx.TOP, 8)
            row.Add(controls, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 14)
            layout = wx.BoxSizer(wx.VERTICAL)
            layout.Add(row, 1, wx.EXPAND)
            self.progress = Progress(self)
            layout.Add(self.progress, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 16)
            self.SetSizer(layout)
            self.key = None
            self.update(job)
            request = read_json(root / job["id"] / "request.json", {})
            path = Path(request.get("input", ""))
            if path.is_file():
                future = pool.submit(thumb, path)

                def done(f):
                    try:
                        im = f.result()
                        wx.CallAfter(self.photo, im)
                    except Exception:
                        pass

                future.add_done_callback(done)

        def photo(self, im):
            if self and self.thumb:
                self.thumb.set_photo(im)

        def update(self, j):
            self.job = j
            p = j.get("progress", {})
            active = j["state"] in ("running", "queued")
            text = (
                f"{j['scale']:g}× · { ({'xray': 'X-Ray', 'xray-plus': 'X-Ray'}.get(j.get('mode'), j.get('mode', 'max').title())) } · "
                + (p.get("stage", j.get("message", "")) if active else j.get("message", ""))
            )
            if active and p.get("tiles"):
                text += f" · {p.get('tile', 0)}/{p['tiles']} tiles saved"
            self.detail.SetLabel(text[:100])
            self.detail.SetToolTip(text)
            self.elapsed.SetLabel(
                ("Processing time · " if j["state"] == "complete" else "Elapsed · ")
                + duration_text(j.get("elapsed_seconds", 0))
            )
            if j["state"] == "complete":
                if self.meta is None:
                    self.meta = read_json(root / j["id"] / "result.json", {})
                source = self.meta.get("input_size")
                target = self.meta.get("output_size")
                elapsed = duration_text(j.get("elapsed_seconds", self.meta.get("seconds", 0)))
                self.resolution.SetLabel(
                    f"{source[0]} × {source[1]} → {target[0]} × {target[1]} px"
                    if source and target
                    else ""
                )
                self.elapsed.SetLabel("Processing time · " + elapsed)
                timings = self.meta.get("timings", {}).get("stage_seconds", {})
                if timings:
                    slowest = max(timings, key=timings.get)
                    self.resolution.SetToolTip(
                        f"Slowest measured stage: {slowest} · {timings[slowest]:.1f} contributor-seconds (parallel tile times summed)"
                    )
            else:
                self.meta = None
                elapsed = duration_text(j.get("elapsed_seconds", 0))
                self.resolution.SetLabel("")
            self.progress.value = (
                max(self.progress.value, 0.01, min(1, p.get("progress", 0))) if active else 0
            )
            self.progress.Refresh()
            key = (j["state"], j.get("export_state"))
            if key == self.key:
                self.Layout()
                return
            self.key = key
            self.actions.Clear(True)

            def add(text, fn, width=88):
                self.actions.Add(Button(self, text, fn, width=width), 0, wx.LEFT, 7)

            if j["state"] == "complete":
                add("Compare", lambda: self.owner.open_compare(self.job))
                add("Save PNG", lambda: self.owner.save(self.job, "png"))
                add("JPEG", lambda: self.owner.save_jpeg(self.job), 70)
                add(
                    "SVG ready" if j.get("export_state") == "complete" else "Advanced",
                    self.advanced,
                    95,
                )
            else:
                add(
                    "Pause" if active else "Resume",
                    lambda: self.owner.act(self.job, "cancel" if active else "resume"),
                )
            self.Layout()

        def advanced(self):
            if self.job.get("export_state") == "complete":
                self.owner.save(self.job, "svg")
            elif self.job.get("export_state") in ("running", "queued"):
                self.owner.notice("Vector export is in progress. PNG is ready.")
            else:
                self.owner.act(self.job, "export-svg")

    class Frame(wx.Frame):
        def __init__(self):
            super().__init__(None, title="Untorn Engine", size=(1260, 920))
            self.SetMinSize((980, 730))
            self.SetBackgroundColour(BG)
            if os.name == "nt":
                import ctypes

                dark = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    ctypes.c_void_p(self.GetHandle()), 20, ctypes.byref(dark), ctypes.sizeof(dark)
                )
            icon = ASSETS / "untorn.ico"
            if icon.exists():
                self.SetIcon(wx.Icon(str(icon)))
            self.selected = []
            self.generation = 0
            self.closed = False
            self.adding = False
            self.cards = {}
            self.comparison = None
            self.scroll = wx.lib.scrolledpanel.ScrolledPanel(self)
            self.scroll.SetBackgroundColour(BG)
            self.scroll.SetupScrolling(scroll_x=False, rate_y=20)
            p = self.scroll
            outer = wx.BoxSizer(wx.VERTICAL)
            head = wx.BoxSizer(wx.HORIZONTAL)
            head.Add(label(p, "U N T O R N   E N G I N E", 15), 0, wx.ALIGN_CENTER_VERTICAL)
            head.AddStretchSpacer()
            head.Add(
                label(p, "●  LOCAL DESKTOP", 9, "#a5c8ae"),
                0,
                wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
                20,
            )
            head.Add(Button(p, "About", self.about, width=80))
            outer.Add(head, 0, wx.EXPAND | wx.BOTTOM, 36)
            outer.Add(
                label(p, "HIGH-DEFINITION UPSCALING / LOCAL ENGINE", 9, MUTED), 0, wx.BOTTOM, 12
            )
            self.hero = label(p, "Low resolution. Clearly defined.", 26)
            outer.Add(self.hero, 0, wx.BOTTOM, 10)
            outer.Add(
                label(
                    p,
                    "Source-derived texture. Controlled lighting. Precise output dimensions.",
                    11,
                    MUTED,
                ),
                0,
                wx.BOTTOM,
                28,
            )
            layout = wx.BoxSizer(wx.HORIZONTAL)
            left = wx.BoxSizer(wx.VERTICAL)
            modes = wx.BoxSizer(wx.HORIZONTAL)
            self.inputmode = Switch(p, ["Single image", "Batch"], 0, self.mode_change)
            modes.Add(self.inputmode)
            modes.AddStretchSpacer()
            self.count = label(p, "Ready when you are", 9, MUTED)
            modes.Add(self.count, 0, wx.ALIGN_CENTER_VERTICAL)
            left.Add(modes, 0, wx.EXPAND | wx.BOTTOM, 12)
            self.photo = PhotoPanel(p)
            self.photo.SetDropTarget(Drop(self.select, self.photo))
            self.photo.Bind(wx.EVT_LEFT_UP, lambda e: self.browse())
            left.Add(self.photo, 1, wx.EXPAND | wx.BOTTOM, 12)
            buttons = wx.BoxSizer(wx.HORIZONTAL)
            self.choose = Button(p, "Choose image", self.browse, True, width=145)
            buttons.Add(self.choose, 0, wx.RIGHT, 8)
            self.folder = Button(p, "Choose folder", self.browse_folder, width=135)
            buttons.Add(self.folder, 0, wx.RIGHT, 8)
            self.folder.Hide()
            buttons.Add(Button(p, "Clear", self.clear, width=75))
            buttons.AddStretchSpacer()
            buttons.Add(
                label(p, "PNG · JPEG · WEBP · TIFF · BMP", 8, MUTED), 0, wx.ALIGN_CENTER_VERTICAL
            )
            left.Add(buttons, 0, wx.EXPAND)
            layout.Add(left, 1, wx.EXPAND | wx.RIGHT, 24)
            side = RoundedPanel(p)
            side.SetMinSize(side.FromDIP((268, -1)))
            settings = wx.BoxSizer(wx.VERTICAL)
            settings.Add(label(side, "Output", 13), 0, wx.BOTTOM, 20)
            settings.Add(label(side, "Scale multiplier", 9, MUTED), 0, wx.BOTTOM, 8)
            self.scale = wx.TextCtrl(side, value="4", style=wx.BORDER_SIMPLE)
            self.scale.SetBackgroundColour("#101317")
            self.scale.SetForegroundColour(INK)
            self.scale.SetFont(font(23))
            self.scale.SetName("Scale multiplier")
            settings.Add(self.scale, 0, wx.EXPAND | wx.BOTTOM, 8)
            self.scale_slider = ScaleSlider(side, lambda v: self.scale.ChangeValue(f"{v:g}"))
            settings.Add(self.scale_slider, 0, wx.EXPAND | wx.BOTTOM, 3)
            self.scale.Bind(wx.EVT_TEXT, self.scale_changed)
            settings.Add(label(side, "1–16× · fractional precision", 9, MUTED), 0, wx.BOTTOM, 22)
            self.quality = Switch(side, ["Fast", "Max", "X-Ray"], 1, self.quality_change)
            settings.Add(self.quality, 0, wx.EXPAND | wx.BOTTOM, 9)
            self.qualityhint = label(side, "Quality and source faithfulness.", 9, MUTED)
            settings.Add(self.qualityhint, 0, wx.BOTTOM, 22)
            self.svg = wx.CheckBox(side, label="Advanced: true-vector SVG")
            self.svg.SetForegroundColour(MUTED)
            self.svg.SetFont(font(9))
            settings.Add(self.svg, 0, wx.BOTTOM, 7)
            settings.Add(
                label(side, "PNG first. Vector export up to 2 MP.", 8, MUTED), 0, wx.BOTTOM, 24
            )
            self.run = Button(side, "Upscale image  ↗", self.submit, True, width=220, accent=True)
            self.run.Disable()
            settings.Add(self.run, 0, wx.EXPAND | wx.BOTTOM, 12)
            settings.Add(label(side, "UNTORN / PRECISION UPSCALING", 9, MUTED))
            ss = wx.BoxSizer(wx.VERTICAL)
            ss.Add(settings, 1, wx.EXPAND | wx.ALL, 23)
            side.SetSizer(ss)
            layout.Add(side, 0, wx.EXPAND)
            outer.Add(layout, 0, wx.EXPAND | wx.BOTTOM, 16)
            self.message = label(p, "", 9, "#f0ad9e")
            outer.Add(self.message, 0, wx.EXPAND | wx.BOTTOM, 20)
            qhead = wx.BoxSizer(wx.HORIZONTAL)
            qhead.Add(label(p, "Your workspace", 14), 0, wx.ALIGN_CENTER_VERTICAL)
            qhead.AddStretchSpacer()
            qhead.Add(Button(p, "Open output folder ↗", lambda: os.startfile(root), width=180))
            outer.Add(qhead, 0, wx.EXPAND | wx.BOTTOM, 14)
            self.jobs = wx.BoxSizer(wx.VERTICAL)
            self.empty = label(p, "Finished images and saved progress appear here.", 10, MUTED)
            self.jobs.Add(self.empty, 0, wx.BOTTOM, 18)
            outer.Add(self.jobs, 0, wx.EXPAND)
            footer = wx.BoxSizer(wx.HORIZONTAL)
            footer.Add(label(p, "Parham (Leon) Faraji  ·  @leon-executes", 9, MUTED))
            footer.AddStretchSpacer()
            footer.Add(label(p, "PolyForm Noncommercial 1.0.0", 8, MUTED))
            outer.Add(footer, 0, wx.EXPAND | wx.TOP, 28)
            wrap = wx.BoxSizer(wx.VERTICAL)
            wrap.Add(outer, 1, wx.EXPAND | wx.ALL, 36)
            p.SetSizer(wrap)
            fs = wx.BoxSizer(wx.VERTICAL)
            fs.Add(p, 1, wx.EXPAND)
            self.SetSizer(fs)
            p.SetDropTarget(Drop(self.select))
            self.Bind(wx.EVT_CLOSE, self.close)
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.refresh, self.timer)
            self.timer.Start(700)
            self.refresh()
            self.Centre()
            self.Show()

        def notice(self, text):
            if not self.closed:
                self.message.SetLabel(text)
                self.message.Wrap(max(400, self.scroll.GetClientSize().width - 80))
                self.scroll.Layout()

        def open_compare(self, job):
            if self.comparison:
                return
            try:
                page = Compare(self, job)
            except Exception as exc:
                self.notice("Could not open comparison: " + str(exc))
                return
            self.comparison = page
            self.scroll.Hide()
            self.GetSizer().Add(page, 1, wx.EXPAND)
            self.Layout()
            page.Show()
            page.close_button.SetFocus()

        def close_compare(self):
            page = self.comparison
            if not page:
                return
            self.comparison = None
            page.stop()
            self.GetSizer().Detach(page)
            page.Hide()
            page.Destroy()
            self.scroll.Show()
            self.Layout()
            self.scroll.SetFocus()

        def about(self):
            wx.MessageBox(
                "Untorn Engine 2.1.2\nFaithful high-definition upscaling\n\nCreated by Parham (Leon) Faraji\ngithub.com/leon-executes\n\nOriginal application: PolyForm Noncommercial 1.0.0.\nBundled components retain their own licenses.\n\nNative Windows UI. No browser, AI models, account,\ntelemetry, or upload service.",
                "About Untorn Engine",
                wx.OK | wx.ICON_INFORMATION,
                self,
            )

        def mode_change(self, i):
            self.hero.SetLabel("Batch upscaling" if i else "Low resolution. Clearly defined.")
            self.choose.SetLabel("Choose images" if i else "Choose image")
            if not self.selected:
                self.photo.set_photo(
                    None,
                    "Select images to upscale" if i else "Select an image to upscale",
                    "Drop files or a folder here, or browse your computer."
                    if i
                    else "Drop an image here, or browse your computer.",
                )
            self.folder.Show(i == 1)
            self.run.SetLabel("Upscale batch  ↗" if i else "Upscale image  ↗")
            self.scroll.Layout()
            if i == 0 and len(self.selected) > 1:
                self.select([self.selected[0]])

        def quality_change(self, i):
            self.qualityhint.SetLabel(
                [
                    "Faithful detail. Prioritized speed.",
                    "Quality and source faithfulness.",
                    "Texture definition. Controlled tone.",
                ][i]
            )

        def scale_changed(self, e):
            try:
                value = float(self.scale.GetValue())
                if math.isfinite(value) and 1 <= value <= 16:
                    self.scale_slider.set(value)
            except ValueError:
                pass

        def browse(self):
            flags = (
                wx.FD_OPEN
                | wx.FD_FILE_MUST_EXIST
                | (wx.FD_MULTIPLE if self.inputmode.selected else 0)
            )
            with wx.FileDialog(
                self,
                "Choose images",
                wildcard="Images|*.png;*.jpg;*.jpeg;*.webp;*.tif;*.tiff;*.bmp",
                style=flags,
            ) as dialog:
                if dialog.ShowModal() == wx.ID_OK:
                    self.select(dialog.GetPaths())

        def browse_folder(self):
            with wx.DirDialog(self, "Choose an image folder") as dialog:
                if dialog.ShowModal() == wx.ID_OK:
                    self.select([dialog.GetPath()])

        def select(self, paths):
            if self.adding:
                return
            self.generation += 1
            generation = self.generation
            bulk = bool(self.inputmode.selected)
            self.notice("Reading selection…")

            def collect():
                files = []
                for name in paths:
                    path = Path(name)
                    if path.is_dir():
                        files.extend(
                            sorted(
                                p
                                for p in path.rglob("*")
                                if p.is_file()
                                and p.suffix.lower() in FORMATS
                                and not p.resolve().is_relative_to(root)
                            )
                        )
                    elif path.suffix.lower() in FORMATS:
                        files.append(path)
                files = list(dict.fromkeys(files))
                files = files if bulk else files[:1]
                if not files:
                    raise ValueError("No supported images in this selection.")
                return files, thumb(files[0])

            f = pool.submit(collect)

            def done(f):
                try:
                    files, im = f.result()
                    wx.CallAfter(self.selection_ready, generation, files, im)
                except Exception as exc:
                    wx.CallAfter(self.notice, str(exc))

            f.add_done_callback(done)

        def selection_ready(self, generation, files, im):
            if self.closed or generation != self.generation:
                return
            self.selected = files
            self.photo.set_photo(
                im,
                files[0].name,
                "Ready to upscale"
                if len(files) == 1
                else f"{len(files)} images · showing the first image",
            )
            self.count.SetLabel(
                f"{len(files)} image" + ("s" if len(files) > 1 else "") + " selected"
            )
            self.run.Enable()
            self.notice("")

        def clear(self):
            if self.adding:
                return
            self.generation += 1
            self.selected = []
            self.photo.set_photo(
                None,
                "Select images to upscale"
                if self.inputmode.selected
                else "Select an image to upscale",
                "Drop files or a folder here, or browse your computer."
                if self.inputmode.selected
                else "Drop an image here, or browse your computer.",
            )
            self.count.SetLabel("Ready when you are")
            self.run.Disable()
            self.notice("")

        def submit(self):
            try:
                value = float(self.scale.GetValue())
                dimensions((1, 1), value)
            except Exception:
                self.notice("Choose a finite scale between 1 and 16, such as 2.5.")
                return
            if not self.selected or self.adding:
                return
            selected = list(self.selected)
            mode = ("fast", "max", "xray")[self.quality.selected]
            svg = self.svg.GetValue()
            self.adding = True
            self.run.Disable()
            self.notice("Adding images to your workspace…")

            def add():
                errors = []
                for p in selected:
                    if self.closed:
                        break
                    try:
                        studio.add(p, value, svg, mode=mode)
                    except Exception as exc:
                        errors.append(p.name + ": " + str(exc))
                wx.CallAfter(self.added, errors)

            threading.Thread(target=add, daemon=True).start()

        def added(self, errors):
            if self.closed:
                return
            self.adding = False
            self.run.Enable(bool(self.selected))
            self.notice(
                "; ".join(errors)
                if errors
                else "Added. You can keep working while the engine runs."
            )
            self.refresh()

        def refresh(self, e=None):
            if self.closed:
                return
            changed = False
            for j in studio.snapshot():
                if j["id"] not in self.cards:
                    self.empty.Hide()
                    card = JobCard(self.scroll, self, j)
                    self.cards[j["id"]] = card
                    self.jobs.Insert(0, card, 0, wx.EXPAND | wx.BOTTOM, 10)
                    changed = True
                else:
                    self.cards[j["id"]].update(j)
            if changed:
                self.scroll.Layout()
                self.scroll.FitInside()

        def act(self, job, action):
            try:
                studio.act(job["id"], action)
                self.refresh()
            except Exception as exc:
                self.notice(str(exc))

        def save(self, job, extension):
            source = root / job["id"] / ("result." + extension)
            with wx.FileDialog(
                self,
                "Save " + extension.upper(),
                defaultFile=Path(job["name"]).stem + "-untorn." + extension,
                wildcard=extension.upper() + "|*." + extension,
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            ) as dialog:
                if dialog.ShowModal() != wx.ID_OK:
                    return
                destination = Path(dialog.GetPath())

            # Copy from a background thread so large PNG/SVG saves never block painting.
            def save_copy():
                try:
                    if source.resolve() != destination.resolve():
                        shutil.copy2(source, destination)
                    wx.CallAfter(self.notice, "Saved " + destination.name)
                except Exception as exc:
                    wx.CallAfter(self.notice, str(exc))

            threading.Thread(target=save_copy, daemon=True).start()

        def save_jpeg(self, job):
            try:
                with JpegDialog(self, job) as dialog:
                    dialog.ShowModal()
            except Exception as exc:
                self.notice(str(exc))

        def close(self, e):
            self.closed = True
            self.timer.Stop()
            if self.comparison:
                self.comparison.stop()
            studio.shutdown()
            pool.shutdown(wait=False, cancel_futures=True)
            self.Destroy()

    try:
        _frame = Frame()
        app.MainLoop()
    finally:
        studio.shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
        guard.close()


# ── Command-line entry point ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Untorn Engine — faithful high-definition upscaling. Native Windows, local processing, fractional output sizes."
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Image files or folders (recursive)")
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Output multiplier from 1 to 16, including decimals",
    )
    parser.add_argument(
        "--output", type=Path, help="Output directory; originals are never overwritten"
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Advanced: compile true-vector SVG after PNG; maximum 2 MP, no embedded raster",
    )
    parser.add_argument(
        "--desktop",
        "--serve",
        dest="serve",
        action="store_true",
        help="Open the native desktop app",
    )
    parser.add_argument(
        "--mode",
        type=normalize_mode,
        choices=["fast", "max", "xray"],
        default="max",
        help="Fast: faithfulness and speed, 32-reference detail. Max (default): quality and source faithfulness, 64 references. X-Ray: enhanced definition and tonal visibility, one shared 80-reference field and refined source-derived grain.",
    )
    parser.add_argument(
        "--version", action="version", version=f"Untorn Engine {APP_VERSION} / {AUTHOR}"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="Validated GPU for large stages, with CPU fallback; cpu disables GPU",
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(7),
        default=0,
        help="CPU tile workers: 0 automatic, 1 serial, up to 6 within the available memory budget",
    )
    parser.add_argument("--jpeg-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--svg-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--comparison-region-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--comparison-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.comparison_region_worker:
        try:
            comparison_region(read_json(args.comparison_region_worker))
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if args.jpeg_worker:
        try:
            save_json(
                args.jpeg_worker.parent / "result.json", jpeg_export(read_json(args.jpeg_worker))
            )
        except Exception as exc:
            save_json(args.jpeg_worker.parent / "error.json", dict(error=friendly_error(exc)))
            traceback.print_exc()
            return 1
        return 0
    if args.svg_worker:
        try:
            vector_export(args.svg_worker)
        except Exception as exc:
            save_json(args.svg_worker / "svg-error.json", dict(error=str(exc)))
            traceback.print_exc()
            return 1
        return 0
    if args.comparison_worker:
        try:
            bicubic_export(args.comparison_worker)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if args.worker:
        request = read_json(args.worker)
        try:
            process_request(request)
        except Exception as exc:
            save_json(
                Path(request["folder"]) / "error.json",
                dict(error=friendly_error(exc), details=str(exc), kind=type(exc).__name__),
            )
            traceback.print_exc()
            return 1
        return 0
    root = (args.output or ROOT / "Untorn-output").resolve()
    if args.serve or not args.inputs:
        desktop(root)
        return 0
    if not math.isfinite(args.scale) or not 1 <= args.scale <= 16:
        parser.error("--scale must be finite and between 1 and 16")
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for path in args.inputs:
        if path.is_dir():
            paths.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file()
                    and p.suffix.lower() in FORMATS
                    and not p.resolve().is_relative_to(root)
                )
            )
        else:
            paths.append(path)
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if not paths:
        parser.error("No supported input images found")
    lock = workspace_lock(root)
    studio = Studio(root)
    added = []
    try:
        for path in paths:
            try:
                added.append(
                    studio.add(
                        path,
                        args.scale,
                        args.svg,
                        mode=args.mode,
                        device=args.device,
                        workers=args.workers,
                    )["id"]
                )
            except Exception as exc:
                print(json.dumps({"input": str(path), "error": str(exc)}), flush=True)
        last = {}
        while True:
            jobs = [j for j in studio.snapshot() if j["id"] in added]
            for j in jobs:
                state = (j["state"], j["progress"].get("stage"), j["progress"].get("tile"))
                if last.get(j["id"]) != state:
                    print(
                        json.dumps(
                            dict(
                                name=j["name"],
                                state=j["state"],
                                stage=j["progress"].get("stage"),
                                message=j["message"],
                                folder=str(root / j["id"]),
                                elapsed_seconds=round(j["elapsed_seconds"], 3),
                            )
                        ),
                        flush=True,
                    )
                    last[j["id"]] = state
            if all(
                j["state"] in ["complete", "failed", "paused"]
                and j.get("export_state") not in ["queued", "running"]
                for j in jobs
            ):
                break
            time.sleep(0.5)
        return (
            0
            if len(added) == len(paths)
            and all(j["state"] == "complete" and j.get("export_state") != "failed" for j in jobs)
            else 1
        )
    except KeyboardInterrupt:
        return 130
    finally:
        studio.shutdown()
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if getattr(sys, "frozen", False) and sys.stderr is None:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, str(exc), "Untorn Engine", 0x10)
            raise SystemExit(1)
        raise
