import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Animal Walk Animator — Blender",
    page_icon="🦒",
    layout="wide",
)

st.title("🦒 Hand-Drawn Animal Walk Animator — Blender 3D")

st.caption(
    "Gemini analyzes your hand-drawn animal, segments the artwork, "
    "and generates a self-contained Python script to animate and render it inside Blender."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Animation Settings")

# Securely fetch API key from Streamlit Secrets or user input
DEFAULT_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

GEMINI_API_KEY = st.sidebar.text_input(
    "Gemini API Key",
    value=DEFAULT_API_KEY,
    type="password",
)

ANIMATION_MODE = st.sidebar.selectbox(
    "Animation mode",
    [
        "White canvas → walk in → stand → merge",
        "White canvas → walk in → stand",
        "Walk in place only",
    ],
)

TOTAL_FRAMES = st.sidebar.slider(
    "Total animation frames",
    30,
    160,
    80,
    2,
)

FPS = st.sidebar.slider(
    "FPS",
    4,
    20,
    7,
)

WALK_CYCLES = st.sidebar.slider(
    "Walking cycles",
    0,
    5,
    2,
)

STEP_ANGLE = st.sidebar.slider(
    "Leg swing angle",
    2,
    30,
    10,
)

BODY_BOB = st.sidebar.slider(
    "Body bob",
    0.0,
    0.04,
    0.006,
    0.001,
)

WALK_IN_FRACTION = st.sidebar.slider(
    "Walk-in fraction",
    0.10,
    0.50,
    0.25,
    0.05,
)

STAND_FRACTION = st.sidebar.slider(
    "Stand fraction",
    0.10,
    0.60,
    0.45,
    0.05,
)

MERGE_FRACTION = st.sidebar.slider(
    "Merge fraction",
    0.05,
    0.35,
    0.18,
    0.05,
)

INK_DILATION = st.sidebar.slider(
    "Original-pixel extraction dilation",
    1,
    7,
    3,
)

SHADOW_SUPPRESSION = st.sidebar.checkbox(
    "Suppress floor/background shadows",
    value=True,
)

ENTRY_SIDE = st.sidebar.selectbox(
    "Animal enters from",
    ["Left", "Right"],
)

ENTRY_EXTRA_DISTANCE = st.sidebar.slider(
    "Extra entry distance",
    0.00,
    0.30,
    0.08,
    0.01,
)

BLENDER_RESOLUTION = st.sidebar.selectbox(
    "Blender render resolution",
    [
        "512 × 512",
        "768 × 768",
        "1024 × 1024",
    ],
)

BLENDER_EXTRUSION = st.sidebar.slider(
    "3D extrusion depth",
    0.01,
    0.30,
    0.08,
    0.01,
)

BLENDER_BEVEL = st.sidebar.slider(
    "3D edge bevel",
    0.0,
    0.10,
    0.015,
    0.005,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "scene": None,
    "animal_prepared": None,
    "blender_script": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GEMINI HELPERS
# ============================================================

def clean_json_text(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def model_name(obj: Any) -> str:
    name = getattr(obj, "name", "") or ""
    return str(name).strip()


def model_actions(obj: Any) -> List[str]:
    actions = getattr(
        obj,
        "supported_actions",
        None,
    )

    if actions is None:
        actions = getattr(
            obj,
            "supportedActions",
            None,
        )

    if actions is None:
        return []

    try:
        return [str(a) for a in actions]
    except Exception:
        return []


def discover_models(client: genai.Client) -> List[str]:
    discovered = []

    try:
        for model_obj in client.models.list():

            name = model_name(model_obj)

            if not name:
                continue

            actions = model_actions(model_obj)

            if actions and "generateContent" not in actions:
                continue

            discovered.append(name)

    except Exception:
        return []

    unique = []
    seen = set()

    for name in discovered:

        key = name.lower()

        if key not in seen:
            seen.add(key)
            unique.append(name)

    return unique


def model_score(name: str) -> Tuple[int, str]:

    n = name.lower().replace(
        "models/",
        "",
    )

    if "gemini-3.7-flash" in n:
        return 0, n

    if "gemini-3.6-flash" in n:
        return 1, n

    if "gemini-3.5-flash" in n:
        return 2, n

    if "gemini-3.1-flash" in n and "lite" not in n:
        return 3, n

    if "gemini-3.1-flash-lite" in n:
        return 4, n

    if "gemini-3" in n and "flash" in n:
        return 5, n

    if "gemini-2.5-flash" in n and "lite" not in n:
        return 6, n

    if "gemini-2.5-flash-lite" in n:
        return 7, n

    if "gemini-2.5" in n:
        return 8, n

    if "gemini-2" in n and "flash" in n:
        return 10, n

    if "gemini" in n and "pro" in n:
        return 15, n

    if "gemini" in n and "flash" in n:
        return 20, n

    if "gemini" in n:
        return 30, n

    return 100, n


def safe_text(response: Any) -> str:

    text = getattr(
        response,
        "text",
        None,
    )

    if text:
        return str(text)

    try:

        pieces = []

        for candidate in (
            getattr(
                response,
                "candidates",
                [],
            )
            or []
        ):

            content = getattr(
                candidate,
                "content",
                None,
            )

            for part in (
                getattr(
                    content,
                    "parts",
                    [],
                )
                or []
            ):

                part_text = getattr(
                    part,
                    "text",
                    None,
                )

                if part_text:
                    pieces.append(str(part_text))

        return "\n".join(pieces).strip()

    except Exception:

        return ""


# ============================================================
# GEMINI SCENE ANALYSIS
# ============================================================

def analyze_scene(
    image_bytes: bytes,
    api_key: str,
) -> Optional[Dict[str, Any]]:

    if not api_key:

        st.error(
            "Please enter your Gemini API key."
        )

        return None

    try:

        client = genai.Client(
            api_key=api_key
        )

    except Exception as exc:

        st.error(
            f"Could not initialize Gemini: {exc}"
        )

        return None

    discovered = discover_models(client)

    if not discovered:

        st.error(
            "Gemini API did not return any models "
            "supporting generateContent."
        )

        return None

    discovered.sort(key=model_score)

    prompt = r"""
You are analyzing a SINGLE hand-drawn animal scene for programmatic 3D extraction and rigging.

Analyze the image and compute all structural data in normalized coordinates (0..100, using [y, x]).

Return ONLY valid JSON with this exact structure:

{
  "identified_character": "animal_name",
  "animal_bbox": [ymin, xmin, ymax, xmax],
  "animal_polygon": [[y, x], [y, x]],
  "complete_animal_check": {
    "head_included": true,
    "ears_included": true,
    "tail_included": true,
    "all_visible_legs_included": true,
    "small_visible_parts_included": true
  },
  "parts": [
    {
      "name": "body",
      "type": "body",
      "polygon": [[y, x], [y, x]]
    },
    {
      "name": "front_left_leg",
      "type": "leg",
      "side": "front_left",
      "polygon": [[y, x], [y, x]],
      "joints": {
        "proximal": [y, x],
        "middle": [y, x],
        "distal": [y, x]
      }
    }
  ]
}

For quadrupeds, use side tags: front_left_leg, front_right_leg, back_left_leg, back_right_leg.
Identify every visible leg and joint pivot point accurately.
"""

    errors = []

    for current_model in discovered:

        try:

            contents = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ]

            try:

                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.05,
                    ),
                )

            except Exception:

                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.05,
                    ),
                )

            text = clean_json_text(
                safe_text(response)
            )

            if not text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError(
                    "Gemini response was not a JSON object."
                )

            st.success(
                f"✅ Gemini analysis completed with `{current_model}`"
            )

            return data

        except Exception as exc:

            errors.append(
                f"{current_model}: "
                f"{str(exc).replace(chr(10), ' ')}"
            )

    st.error(
        "Gemini analysis failed after trying all available models."
    )

    with st.expander("Show model attempts"):

        for error in errors:
            st.code(error)

    return None


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def pt_px(
    p: List[float],
    width: int,
    height: int,
) -> Tuple[int, int]:

    y = float(
        np.clip(
            p[0],
            0,
            100,
        )
    )

    x = float(
        np.clip(
            p[1],
            0,
            100,
        )
    )

    return (
        int(round(x * width / 100.0)),
        int(round(y * height / 100.0)),
    )


def poly_px(
    polygon: List[List[float]],
    width: int,
    height: int,
) -> np.ndarray:

    pts = []

    for p in polygon:

        if (
            isinstance(p, (list, tuple))
            and len(p) >= 2
        ):

            x, y = pt_px(
                p,
                width,
                height,
            )

            pts.append([x, y])

    if len(pts) < 3:

        return np.empty(
            (0, 2),
            dtype=np.int32,
        )

    return np.asarray(
        pts,
        dtype=np.int32,
    )


def poly_mask(
    shape: Tuple[int, int],
    polygon: List[List[float]],
    dilation: int = 0,
) -> np.ndarray:

    h, w = shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    pts = poly_px(
        polygon,
        w,
        h,
    )

    if len(pts) >= 3:

        cv2.fillPoly(
            mask,
            [pts],
            255,
        )

    if dilation > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                dilation * 2 + 1,
                dilation * 2 + 1,
            ),
        )

        mask = cv2.dilate(
            mask,
            kernel,
        )

    return mask


def bbox_poly(
    polygon: List[List[float]],
    width: int,
    height: int,
    margin: int = 10,
) -> Tuple[int, int, int, int]:

    pts = poly_px(
        polygon,
        width,
        height,
    )

    if len(pts) == 0:

        return (
            0,
            0,
            width,
            height,
        )

    x, y, bw, bh = cv2.boundingRect(pts)

    return (
        max(0, x - margin),
        max(0, y - margin),
        min(width, x + bw + margin),
        min(height, y + bh + margin),
    )


# ============================================================
# FOREGROUND EXTRACTION
# ============================================================

def ink_mask(
    image_bgr: np.ndarray,
    region_mask: np.ndarray,
    shadow_suppress: bool = True,
    dilation: int = 3,
) -> np.ndarray:

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    _, saturation, value = cv2.split(hsv)

    colored = saturation > 30

    strong_color = (
        (saturation > 50)
        & (value > 35)
    )

    color_support = cv2.dilate(
        strong_color.astype(np.uint8),
        np.ones(
            (5, 5),
            np.uint8,
        ),
    ) > 0

    dark = value < 175

    if shadow_suppress:

        foreground = (
            colored
            | (dark & color_support)
        )

    else:

        foreground = (
            colored
            | dark
        )

    foreground = (
        foreground.astype(np.uint8)
        * 255
    )

    foreground = cv2.bitwise_and(
        foreground,
        region_mask,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones(
            (3, 3),
            np.uint8,
        ),
    )

    if dilation > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                dilation * 2 + 1,
                dilation * 2 + 1,
            ),
        )

        foreground = cv2.dilate(
            foreground,
            kernel,
        )

    return foreground


def complete_animal_mask(
    image_bgr: np.ndarray,
    scene: Dict[str, Any],
) -> np.ndarray:

    h, w = image_bgr.shape[:2]

    polygon = (
        scene.get("animal_polygon")
        or []
    )

    if len(polygon) < 3:
        return np.zeros(
            (h, w),
            dtype=np.uint8,
        )

    animal_region = poly_mask(
        (h, w),
        polygon,
        dilation=max(
            2,
            INK_DILATION,
        ),
    )

    pixels = ink_mask(
        image_bgr,
        animal_region,
        shadow_suppress=SHADOW_SUPPRESSION,
        dilation=INK_DILATION,
    )

    protected = animal_region.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    interior = cv2.erode(
        animal_region,
        kernel,
    )

    result = pixels.copy()

    result[interior > 0] = np.maximum(
        result[interior > 0],
        pixels[interior > 0],
    )

    result = cv2.bitwise_or(
        result,
        cv2.bitwise_and(
            protected,
            pixels,
        ),
    )

    return result


def leg_parts(
    scene: Dict[str, Any],
) -> List[Dict[str, Any]]:

    result = []

    for part in scene.get(
        "parts",
        [],
    ):

        if not isinstance(part, dict):
            continue

        ptype = str(
            part.get(
                "type",
                "",
            )
        ).lower()

        name = str(
            part.get(
                "name",
                "",
            )
        ).lower()

        if (
            ptype == "leg"
            or "leg" in name
        ):

            result.append(part)

    return result


def prepare_leg_sprite(
    image_bgr: np.ndarray,
    polygon: List[List[float]],
    current_part: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    h, w = image_bgr.shape[:2]

    if len(polygon) < 3:
        return None

    region = poly_mask(
        (h, w),
        polygon,
        dilation=max(
            2,
            INK_DILATION,
        ),
    )

    mask = ink_mask(
        image_bgr,
        region,
        shadow_suppress=SHADOW_SUPPRESSION,
        dilation=INK_DILATION,
    )

    x1, y1, x2, y2 = bbox_poly(
        polygon,
        w,
        h,
        margin=max(
            8,
            min(h, w) // 100,
        ),
    )

    crop = image_bgr[
        y1:y2,
        x1:x2,
    ].copy()

    local_mask = mask[
        y1:y2,
        x1:x2,
    ].copy()

    if crop.size == 0:
        return None

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGBA,
    )

    rgba[:, :, 3] = local_mask

    joints = {}

    for key in (
        "proximal",
        "middle",
        "distal",
    ):

        p = (
            (
                current_part.get("joints", {})
                or {}
            ).get(key)
        )

        if p is not None:

            gx, gy = pt_px(
                p,
                w,
                h,
            )

            joints[key] = [
                int(gx),
                int(gy),
            ]

    return {
        "name": str(
            current_part.get(
                "name",
                "leg",
            )
        ),
        "side": str(
            current_part.get(
                "side",
                current_part.get(
                    "name",
                    "",
                ),
            )
        ).lower(),
        "bbox": [
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        ],
        "joints": joints,
        "rgba": rgba,
    }


def prepare_body_sprite(
    image_bgr: np.ndarray,
    scene: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    h, w = image_bgr.shape[:2]

    animal_polygon = (
        scene.get("animal_polygon")
        or []
    )

    if len(animal_polygon) < 3:
        return None

    animal_mask = complete_animal_mask(
        image_bgr,
        scene,
    )

    x1, y1, x2, y2 = bbox_poly(
        animal_polygon,
        w,
        h,
        margin=max(
            12,
            min(h, w) // 70,
        ),
    )

    crop = image_bgr[
        y1:y2,
        x1:x2,
    ].copy()

    alpha = animal_mask[
        y1:y2,
        x1:x2,
    ].copy()

    if crop.size == 0:
        return None

    for part in leg_parts(scene):

        polygon = (
            part.get("polygon")
            or []
        )

        if len(polygon) < 3:
            continue

        leg_mask_full = poly_mask(
            (h, w),
            polygon,
            dilation=max(
                1,
                INK_DILATION,
            ),
        )

        joints = (
            part.get("joints")
            or {}
        )

        proximal = joints.get(
            "proximal"
        )

        if proximal:

            px, py = pt_px(
                proximal,
                w,
                h,
            )

            yy, xx = np.ogrid[
                :h,
                :w,
            ]

            radius = max(
                8,
                int(
                    min(h, w) * 0.025
                ),
            )

            keep_attachment = (
                (xx - px) ** 2
                + (yy - py) ** 2
                <= radius ** 2
            )

            leg_mask_full[
                keep_attachment
            ] = 0

        alpha = cv2.bitwise_and(
            alpha,
            cv2.bitwise_not(
                leg_mask_full[
                    y1:y2,
                    x1:x2,
                ]
            ),
        )

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGBA,
    )

    rgba[:, :, 3] = alpha

    return {
        "bbox": [
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        ],
        "rgba": rgba,
    }


def detection_overlay(
    image: np.ndarray,
    scene: Dict[str, Any],
) -> np.ndarray:

    overlay = image.copy()

    h, w = image.shape[:2]

    animal_polygon = (
        scene.get(
            "animal_polygon"
        )
        or []
    )

    pts = poly_px(
        animal_polygon,
        w,
        h,
    )

    if len(pts) >= 3:

        cv2.polylines(
            overlay,
            [pts],
            True,
            (0, 255, 0),
            3,
        )

    for part in scene.get(
        "parts",
        [],
    ):

        if not isinstance(
            part,
            dict,
        ):
            continue

        polygon = (
            part.get(
                "polygon",
                [],
            )
            or []
        )

        pts = poly_px(
            polygon,
            w,
            h,
        )

        if len(pts) >= 3:

            cv2.polylines(
                overlay,
                [pts],
                True,
                (0, 165, 255),
                2,
            )

        joints = (
            part.get(
                "joints",
                {},
            )
            or {}
        )

        for joint_name in (
            "proximal",
            "middle",
            "distal",
        ):

            if joint_name not in joints:
                continue

            x, y = pt_px(
                joints[joint_name],
                w,
                h,
            )

            cv2.circle(
                overlay,
                (x, y),
                6,
                (0, 0, 255),
                -1,
            )

    return overlay


def prepare_animal_for_blender(
    image_bgr: np.ndarray,
    scene: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    body = prepare_body_sprite(
        image_bgr,
        scene,
    )

    if body is None:
        return None

    prepared_legs = []

    for part in leg_parts(scene):

        prepared = prepare_leg_sprite(
            image_bgr,
            part.get(
                "polygon",
                [],
            )
            or [],
            part,
        )

        if prepared is not None:
            prepared_legs.append(prepared)

    return {
        "image_width": int(
            image_bgr.shape[1]
        ),
        "image_height": int(
            image_bgr.shape[0]
        ),
        "body": body,
        "legs": prepared_legs,
    }


def rgba_to_base64(
    rgba: np.ndarray,
) -> str:

    ok, encoded = cv2.imencode(
        ".png",
        rgba,
    )

    if not ok:
        raise ValueError(
            "Could not encode RGBA sprite as PNG."
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")


# ============================================================
# BLENDER SCRIPT GENERATOR
# ============================================================

def generate_blender_script(
    prepared: Dict[str, Any],
) -> str:

    resolution_map = {
        "512 × 512": 512,
        "768 × 768": 768,
        "1024 × 1024": 1024,
    }

    resolution = resolution_map.get(
        BLENDER_RESOLUTION,
        768,
    )

    body = prepared["body"]

    config = {
        "image_width": prepared["image_width"],
        "image_height": prepared["image_height"],
        "total_frames": TOTAL_FRAMES,
        "fps": FPS,
        "walk_cycles": WALK_CYCLES,
        "step_angle": STEP_ANGLE,
        "body_bob": BODY_BOB,
        "walk_in_fraction": WALK_IN_FRACTION,
        "stand_fraction": STAND_FRACTION,
        "merge_fraction": MERGE_FRACTION,
        "entry_side": ENTRY_SIDE,
        "entry_extra_distance": ENTRY_EXTRA_DISTANCE,
        "extrusion": BLENDER_EXTRUSION,
        "bevel": BLENDER_BEVEL,
        "resolution": resolution,
        "animation_mode": ANIMATION_MODE,
    }

    assets = {
        "body": {
            "bbox": body["bbox"],
            "png_b64": rgba_to_base64(
                body["rgba"]
            ),
        },
        "legs": [],
    }

    for leg in prepared["legs"]:

        assets["legs"].append(
            {
                "name": leg["name"],
                "side": leg["side"],
                "bbox": leg["bbox"],
                "joints": leg["joints"],
                "png_b64": rgba_to_base64(
                    leg["rgba"]
                ),
            }
        )

    config_text = json.dumps(
        config,
        indent=2,
    )

    assets_text = json.dumps(
        assets,
        indent=2,
    )

    script = r'''
import base64
import json
import math
import os

import bpy


# ============================================================
# GENERATED INPUT
# ============================================================

CONFIG = __CONFIG_PLACEHOLDER__

ASSETS = __ASSETS_PLACEHOLDER__


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
) if "__file__" in locals() or "__file__" in globals() else os.getcwd()

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "blender_render"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CLEAN SCENE
# ============================================================

bpy.ops.object.select_all(
    action="SELECT"
)

bpy.ops.object.delete(
    use_global=False
)

for datablocks in (
    bpy.data.meshes,
    bpy.data.curves,
    bpy.data.materials,
    bpy.data.cameras,
    bpy.data.lights,
):
    for block in list(datablocks):
        try:
            if block.users == 0:
                datablocks.remove(block)
        except Exception:
            pass


# ============================================================
# SCENE
# ============================================================

scene = bpy.context.scene

scene.frame_start = 1

scene.frame_end = int(
    CONFIG["total_frames"]
)

scene.render.fps = int(
    CONFIG["fps"]
)

resolution = int(
    CONFIG["resolution"]
)

scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = "PNG"

scene.render.film_transparent = False


# ============================================================
# RENDER ENGINE
# ============================================================

try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except Exception:
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        pass


# ============================================================
# WORLD
# ============================================================

world = scene.world

if world is None:

    world = bpy.data.worlds.new(
        "AnimalWorld"
    )

    scene.world = world

world.use_nodes = True

background = world.node_tree.nodes.get(
    "Background"
)

if background:

    background.inputs["Color"].default_value = (
        1.0,
        1.0,
        1.0,
        1.0,
    )

    background.inputs["Strength"].default_value = 0.7


# ============================================================
# IMAGE DECODING
# ============================================================

def write_asset(
    name,
    encoded,
):

    path = os.path.join(
        OUTPUT_DIR,
        name,
    )

    data = base64.b64decode(
        encoded
    )

    with open(
        path,
        "wb",
    ) as handle:

        handle.write(data)

    return path


BODY_IMAGE_PATH = write_asset(
    "body.png",
    ASSETS["body"]["png_b64"],
)


LEG_IMAGE_PATHS = []

for index, leg in enumerate(
    ASSETS["legs"]
):

    path = write_asset(
        "leg_%03d.png" % index,
        leg["png_b64"],
    )

    LEG_IMAGE_PATHS.append(
        path
    )


# ============================================================
# MATERIAL
# ============================================================

def make_image_material(
    name,
    image_path,
):

    image = bpy.data.images.load(
        image_path,
        check_existing=True,
    )

    material = bpy.data.materials.new(
        name
    )

    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output = nodes.new(
        "ShaderNodeOutputMaterial"
    )

    shader = nodes.new(
        "ShaderNodeBsdfPrincipled"
    )

    texture = nodes.new(
        "ShaderNodeTexImage"
    )

    texture.image = image

    links.new(
        texture.outputs["Color"],
        shader.inputs["Base Color"],
    )

    if "Alpha" in texture.outputs:

        links.new(
            texture.outputs["Alpha"],
            shader.inputs["Alpha"],
        )

    shader.inputs["Roughness"].default_value = 0.78

    links.new(
        shader.outputs["BSDF"],
        output.inputs["Surface"],
    )

    try:
        material.surface_render_method = "DITHERED"
    except Exception:
        pass

    try:
        material.blend_method = "BLEND"
    except Exception:
        pass

    try:
        material.use_screen_refraction = True
    except Exception:
        pass

    return material


BODY_MATERIAL = make_image_material(
    "Original_Animal_Body",
    BODY_IMAGE_PATH,
)


LEG_MATERIALS = []

for index, path in enumerate(
    LEG_IMAGE_PATHS
):

    LEG_MATERIALS.append(
        make_image_material(
            "Original_Leg_%03d" % index,
            path,
        )
    )


# ============================================================
# PIXEL → BLENDER SCALE
# ============================================================

image_width = float(
    CONFIG["image_width"]
)

image_height = float(
    CONFIG["image_height"]
)

PIXEL_SCALE = 0.01


# ============================================================
# ROOT / BOB CONTROLLERS
# ============================================================

root = bpy.data.objects.new(
    "ANIMAL_ROOT",
    None,
)

bpy.context.collection.objects.link(
    root
)

bob_controller = bpy.data.objects.new(
    "ANIMAL_BODY_BOB",
    None,
)

bpy.context.collection.objects.link(
    bob_controller
)

bob_controller.parent = root


# ============================================================
# MATERIAL PLANE CREATOR
# ============================================================

def create_sprite_plane(
    name,
    width_px,
    height_px,
    material,
    origin_x_px,
    origin_y_px,
    world_x_px,
    world_y_px,
    z_value,
):

    width = (
        float(width_px)
        * PIXEL_SCALE
    )

    height = (
        float(height_px)
        * PIXEL_SCALE
    )

    ox = (
        float(origin_x_px)
        * PIXEL_SCALE
    )

    oy = (
        float(origin_y_px)
        * PIXEL_SCALE
    )

    vertices = [
        (-ox, -oy, 0.0),
        (width - ox, -oy, 0.0),
        (width - ox, -height - oy, 0.0),
        (-ox, -height - oy, 0.0),
    ]

    faces = [
        (0, 3, 2, 1)
    ]

    mesh = bpy.data.meshes.new(
        name + "_Mesh"
    )

    mesh.from_pydata(
        vertices,
        [],
        faces,
    )

    mesh.update()

    obj = bpy.data.objects.new(
        name,
        mesh,
    )

    bpy.context.collection.objects.link(
        obj
    )

    obj.location = (
        float(world_x_px)
        * PIXEL_SCALE,
        -float(world_y_px)
        * PIXEL_SCALE,
        float(z_value),
    )

    obj.data.materials.append(
        material
    )

    uv_layer = (
        mesh.uv_layers.new(
            name="UVMap"
        )
    )

    uv_values = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]

    for loop in mesh.loops:

        uv_layer.data[
            loop.index
        ].uv = uv_values[
            mesh.loops[
                loop.index
            ].vertex_index
        ]

    solidify = obj.modifiers.new(
        "Shallow_3D_Depth",
        "SOLIDIFY",
    )

    solidify.thickness = float(
        CONFIG["extrusion"]
    )

    solidify.offset = 0.0

    bevel_amount = float(
        CONFIG["bevel"]
    )

    if bevel_amount > 0.0:

        bevel = obj.modifiers.new(
            "Soft_Edge",
            "BEVEL",
        )

        bevel.width = bevel_amount
        bevel.segments = 3

    return obj


# ============================================================
# BODY
# ============================================================

body_bbox = ASSETS["body"]["bbox"]

body_x1 = float(
    body_bbox[0]
)

body_y1 = float(
    body_bbox[1]
)

body_x2 = float(
    body_bbox[2]
)

body_y2 = float(
    body_bbox[3]
)

body_width = (
    body_x2
    - body_x1
)

body_height = (
    body_y2
    - body_y1
)

body = create_sprite_plane(
    "Animal_Body",
    body_width,
    body_height,
    BODY_MATERIAL,
    0.0,
    0.0,
    body_x1,
    body_y1,
    0.00,
)

body.parent = bob_controller


# ============================================================
# LEGS
# ============================================================

legs = []

for index, leg_data in enumerate(
    ASSETS["legs"]
):

    bbox = leg_data["bbox"]

    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = float(bbox[2])
    y2 = float(bbox[3])

    width = x2 - x1
    height = y2 - y1

    joints = (
        leg_data.get("joints")
        or {}
    )

    proximal = (
        joints.get("proximal")
        or [
            (x1 + x2) * 0.5,
            y1,
        ]
    )

    proximal_x = float(
        proximal[0]
    )

    proximal_y = float(
        proximal[1]
    )

    local_pivot_x = (
        proximal_x - x1
    )

    local_pivot_y = (
        proximal_y - y1
    )

    material = LEG_MATERIALS[index]

    leg = create_sprite_plane(
        "Leg_%03d_%s" % (
            index,
            leg_data.get(
                "name",
                "leg",
            ),
        ),
        width,
        height,
        material,
        local_pivot_x,
        local_pivot_y,
        proximal_x,
        proximal_y,
        0.03 + index * 0.002,
    )

    leg.parent = bob_controller

    legs.append(
        {
            "object": leg,
            "side": str(
                leg_data.get(
                    "side",
                    "",
                )
            ).lower(),
        }
    )


# ============================================================
# WALK PHASE
# ============================================================

def phase_for_leg(
    side,
    index,
):

    side = str(side).lower()

    if (
        "front_left" in side
        or "back_right" in side
    ):
        return 0.0

    if (
        "front_right" in side
        or "back_left" in side
    ):
        return math.pi

    return (
        0.0
        if index % 2 == 0
        else math.pi
    )


# ============================================================
# FRAME ALLOCATION
# ============================================================

total_frames = int(
    CONFIG["total_frames"]
)

mode = str(
    CONFIG["animation_mode"]
)

if mode == "Walk in place only":

    intro_n = 0
    walk_n = 0
    stand_n = total_frames
    merge_n = 0

else:

    walk_n = int(
        round(
            total_frames
            * float(
                CONFIG[
                    "walk_in_fraction"
                ]
            )
        )
    )

    stand_n = int(
        round(
            total_frames
            * float(
                CONFIG[
                    "stand_fraction"
                ]
            )
        )
    )

    if (
        mode
        == "White canvas → walk in → stand → merge"
    ):

        merge_n = int(
            round(
                total_frames
                * float(
                    CONFIG[
                        "merge_fraction"
                    ]
                )
            )
        )

    else:

        merge_n = 0

    intro_n = max(
        0,
        total_frames
        - walk_n
        - stand_n
        - merge_n,
    )


# ============================================================
# ENTRY POSITION
# ============================================================

character_width = body_width

if (
    CONFIG["entry_side"]
    == "Left"
):

    start_x = -(
        character_width
        + image_width
        * float(
            CONFIG[
                "entry_extra_distance"
            ]
        )
    )

else:

    start_x = (
        image_width
        + image_width
        * float(
            CONFIG[
                "entry_extra_distance"
            ]
        )
        - body_x1
    )


# ============================================================
# ANIMATION
# ============================================================

cycles = max(
    1,
    int(
        CONFIG["walk_cycles"]
    ),
)

step_angle = math.radians(
    float(
        CONFIG["step_angle"]
    )
)

body_bob_amount = (
    float(
        CONFIG["body_bob"]
    )
    * image_height
    * PIXEL_SCALE
)


def set_linear_interpolation(
    obj,
):

    animation_data = (
        obj.animation_data
    )

    if not animation_data:
        return

    action = animation_data.action

    if not action:
        return

    for fcurve in action.fcurves:

        for keyframe in fcurve.keyframe_points:

            keyframe.interpolation = "LINEAR"


# ------------------------------------------------------------
# ROOT MOTION
# ------------------------------------------------------------

for frame in range(
    1,
    total_frames + 1,
):

    root_x = 0.0

    if mode != "Walk in place only":

        if frame <= intro_n:

            root_x = start_x

        elif (
            frame
            <= intro_n + walk_n
        ):

            if walk_n <= 1:

                progress = 1.0

            else:

                progress = (
                    frame
                    - intro_n
                    - 1
                ) / float(
                    walk_n - 1
                )

            smooth = (
                progress
                * progress
                * (
                    3.0
                    - 2.0
                    * progress
                )
            )

            root_x = (
                start_x
                * (1.0 - smooth)
            )

        else:

            root_x = 0.0

    root.location.x = (
        root_x
        * PIXEL_SCALE
    )

    root.keyframe_insert(
        data_path="location",
        index=0,
        frame=frame,
    )


set_linear_interpolation(
    root
)


# ------------------------------------------------------------
# BODY BOB
# ------------------------------------------------------------

for frame in range(
    1,
    total_frames + 1,
):

    if mode == "Walk in place only":

        gait_progress = (
            frame - 1
        ) / float(
            max(
                1,
                total_frames - 1,
            )
        )

    elif frame <= intro_n:

        gait_progress = 0.0

    else:

        gait_progress = (
            frame - 1
        ) / float(
            max(
                1,
                total_frames - 1,
            )
        )

    bob = math.sin(
        gait_progress
        * math.pi
        * 2.0
        * cycles
    )

    bob_controller.location.y = (
        bob
        * body_bob_amount
    )

    bob_controller.keyframe_insert(
        data_path="location",
        index=1,
        frame=frame,
    )


set_linear_interpolation(
    bob_controller
)


# ------------------------------------------------------------
# LEG ROTATION
# ------------------------------------------------------------

for index, leg_data in enumerate(
    legs
):

    obj = leg_data["object"]

    side = leg_data["side"]

    phase = phase_for_leg(
        side,
        index,
    )

    for frame in range(
        1,
        total_frames + 1,
    ):

        if mode != "Walk in place only":

            if frame <= intro_n:

                gait_progress = 0.0

            elif (
                frame
                <= intro_n + walk_n
            ):

                if walk_n <= 1:

                    local_progress = 1.0

                else:

                    local_progress = (
                        frame
                        - intro_n
                        - 1
                    ) / float(
                        walk_n - 1
                    )

                gait_progress = (
                    local_progress
                    * cycles
                )

            else:

                after_walk = (
                    frame
                    - intro_n
                    - walk_n
                )

                gait_progress = (
                    after_walk
                    / float(
                        max(
                            1,
                            stand_n - 1,
                        )
                    )
                    * cycles
                )

        else:

            gait_progress = (
                frame - 1
            ) / float(
                max(
                    1,
                    total_frames - 1,
                )
            ) * cycles

        gait = math.sin(
            gait_progress
            * math.pi
            * 2.0
            + phase
        )

        angle = (
            step_angle
            * gait
        )

        if "back" in side:

            angle *= 0.90

        obj.rotation_euler.z = angle

        obj.keyframe_insert(
            data_path="rotation_euler",
            index=2,
            frame=frame,
        )

    set_linear_interpolation(
        obj
    )


# ============================================================
# CAMERA
# ============================================================

camera_x = (
    image_width
    * PIXEL_SCALE
    * 0.5
)

camera_y = -(
    image_height
    * PIXEL_SCALE
    * 0.5
)

bpy.ops.object.camera_add(
    location=(
        camera_x,
        camera_y,
        20.0,
    )
)

camera = bpy.context.object

camera.name = "Animation_Camera"

camera.data.type = "ORTHO"

camera.data.ortho_scale = (
    max(
        image_width,
        image_height,
    )
    * PIXEL_SCALE
    * 1.08
)

camera.rotation_euler = (
    0.0,
    0.0,
    0.0,
)

scene.camera = camera


# ============================================================
# LIGHTING
# ============================================================

bpy.ops.object.light_add(
    type="AREA",
    location=(
        camera_x,
        camera_y,
        10.0,
    ),
)

light = bpy.context.object

light.name = "Soft_3D_Light"

light.data.energy = 700

light.data.shape = "DISK"

light.data.size = 8.0


# ============================================================
# SECOND SOFT LIGHT
# ============================================================

bpy.ops.object.light_add(
    type="AREA",
    location=(
        camera_x - 4.0,
        camera_y + 3.0,
        6.0,
    ),
)

fill = bpy.context.object

fill.name = "Fill_Light"

fill.data.energy = 250

fill.data.size = 6.0


# ============================================================
# SAVE BLEND AND RENDER (SELF-CONTAINED EXECUTION)
# ============================================================

blend_path = os.path.join(
    OUTPUT_DIR,
    "animal_animation.blend",
)

bpy.ops.wm.save_as_mainfile(
    filepath=blend_path
)

scene.render.filepath = os.path.join(
    OUTPUT_DIR,
    "frame_",
)

scene.render.image_settings.file_format = "PNG"

print("--- STARTING LOCAL BLENDER RENDER ---")
bpy.ops.render.render(
    animation=True
)
print("--- BLENDER ANIMATION COMPLETE ---")
print("OUTPUT_DIRECTORY:", OUTPUT_DIR)
'''

    script = script.replace(
        "__CONFIG_PLACEHOLDER__",
        config_text,
    )

    script = script.replace(
        "__ASSETS_PLACEHOLDER__",
        assets_text,
    )

    return script


# ============================================================
# MAIN STREAMLIT UI FLOW
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your hand-drawn animal image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)

if uploaded_file is None:
    st.info("Upload a hand-drawn animal image to begin.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.read()
    image_pil = Image.open(
        io.BytesIO(
            uploaded_bytes
        )
    ).convert("RGB")

    image_bgr = cv2.cvtColor(
        np.array(image_pil),
        cv2.COLOR_RGB2BGR,
    )

except Exception as exc:
    st.error(f"Could not read uploaded image: {exc}")
    st.stop()

gemini_buffer = io.BytesIO()
image_pil.save(gemini_buffer, format="PNG")
gemini_bytes = gemini_buffer.getvalue()

c1, c2 = st.columns(2)

with c1:
    st.image(image_pil, caption="Original drawing", width="stretch")

with c2:
    st.markdown(
        """
### ☁️ Cloud Workflow

1. Upload your hand-drawn image here on **Streamlit Cloud**.
2. **Gemini** calculates creature boundaries and extracts anatomical pivots.
3. Click **Prepare and Generate Script** to package everything.
4. Download the `.py` script and run it inside Blender on your local desktop to render your animation.
        """
    )

if st.button(
    "🔍 Analyze drawing with Gemini",
    type="primary",
    width="stretch",
):
    with st.spinner("Gemini is analyzing structural geometry and joint pivots..."):
        scene = analyze_scene(
            gemini_bytes,
            GEMINI_API_KEY,
        )

    if scene:
        st.session_state.scene = scene
        st.session_state.animal_prepared = None
        st.session_state.blender_script = None

if not st.session_state.scene:
    st.stop()

scene = st.session_state.scene

st.success("Detected: " + str(scene.get("identified_character", "character")))

with st.expander("🧬 Gemini Anatomy JSON"):
    st.json(scene)

overlay = detection_overlay(image_bgr, scene)
st.image(
    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
    caption="Green = Animal Boundary | Orange = Limbs | Red = Joint Pivots",
    width="stretch",
)

st.markdown("---")
st.header("🧩 Step 2 — Prepare Artwork & Generate Script")

if st.button("🛠️ Build Self-Contained Blender Script", type="primary", width="stretch"):
    with st.spinner("Extracting layer sprites and building script..."):
        prepared = prepare_animal_for_blender(image_bgr, scene)

    if prepared is None:
        st.error("Could not prepare the animal layers.")
    else:
        st.session_state.animal_prepared = prepared
        script_code = generate_blender_script(prepared)
        st.session_state.blender_script = script_code
        st.success("✅ Blender Python script generated successfully!")

prepared = st.session_state.animal_prepared

if prepared:
    st.success(f"Successfully processed body and {len(prepared['legs'])} leg layers.")

if st.session_state.blender_script:
    st.markdown("---")
    st.subheader("🎬 Step 3 — Download & Run in Blender")
    st.info(
        "Since this app is hosted in the cloud, download the script below, "
        "open Blender locally, go to the **Scripting** tab, open the script, and press **Play**."
    )

    st.download_button(
        "⬇️ Download Blender Python Script (`animal_blender_animation.py`)",
        st.session_state.blender_script,
        "animal_blender_animation.py",
        "text/x-python",
        width="stretch",
    )
