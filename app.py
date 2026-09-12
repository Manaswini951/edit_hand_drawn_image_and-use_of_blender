import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
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
    "Gemini identifies the COMPLETE visible animal and its anatomy. "
    "The original drawing is then converted into separate Blender body "
    "and limb meshes, lightly extruded into 3D, rigged at anatomical "
    "pivots and animated using Blender."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Animation Settings")

GEMINI_API_KEY = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
)

BLENDER_PATH = st.sidebar.text_input(
    "Blender executable path",
    value="",
    help=(
        "Leave empty to automatically search for Blender. "
        "Example Windows path: "
        "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
    ),
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

if "scene" not in st.session_state:
    st.session_state.scene = None

if "animal_data" not in st.session_state:
    st.session_state.animal_data = None

if "blender_script" not in st.session_state:
    st.session_state.blender_script = None

if "blender_result" not in st.session_state:
    st.session_state.blender_result = None


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


def discover_models(
    client: genai.Client,
) -> List[str]:

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


def model_score(
    name: str,
) -> Tuple[int, str]:

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


def safe_text(
    response: Any,
) -> str:

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
                    pieces.append(
                        str(part_text)
                    )

        return "\n".join(
            pieces
        ).strip()

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

    discovered = discover_models(
        client
    )

    if not discovered:

        st.error(
            "Gemini API did not return any "
            "models supporting generateContent."
        )

        return None

    discovered.sort(
        key=model_score
    )

    prompt = r"""
You are analyzing a SINGLE hand-drawn animal scene.

The animal will later be imported into Blender and animated.

The goal is NOT to redraw the animal.

The goal is to provide highly accurate geometry so the ORIGINAL
drawing can be separated into a complete body and individual visible
limbs.

============================================================
HIGHEST PRIORITY — COMPLETE ANIMAL RECOGNITION
============================================================

Determine the COMPLETE visible animal from the ORIGINAL IMAGE.

The complete-animal boundary MUST include EVERY visible component
belonging to the animal.

Include, when visible:

- entire body
- head
- muzzle/nose
- every visible ear
- tiny ear tips
- every visible leg
- every visible foot/paw/hoof
- tail
- tail tip
- horns
- antlers
- wings
- eyes
- facial features
- whiskers
- claws
- toes
- thin appendages
- any other visible anatomical component

DO NOT omit small parts.

DO NOT reconstruct invisible anatomy.

DO NOT invent hidden limbs.

============================================================
BLENDER REQUIREMENT
============================================================

The animal will become a 2.5D/3D Blender character.

Therefore:

1. The complete animal polygon must contain every visible
   animal component.

2. Body geometry should represent the central body/head/neck
   mass but MUST NOT deliberately exclude small attached
   anatomy from the complete animal boundary.

3. Every visible leg must be identified separately.

4. Every visible leg must have:

   proximal
   middle
   distal

5. The proximal point must be the most useful anatomical pivot
   for Blender animation.

6. The leg polygon should tightly cover the ORIGINAL visible
   leg pixels.

7. Do not invent legs hidden behind the body.

8. Small ears, tail tips, paws, horns and other details do not
   need animation joints, but MUST remain inside the complete
   animal boundary.

9. The original pixels will be used as textures.

10. DO NOT redraw or beautify the animal.

============================================================
COORDINATES
============================================================

Coordinates must be normalized 0..100.

Use [y,x].

============================================================
COMPLETE-ANIMAL AUDIT
============================================================

Before returning JSON, inspect:

- head
- muzzle
- both sides of body
- every visible ear
- every visible leg
- every visible foot
- tail
- tail tip
- horns
- wings
- facial details
- thin appendages

Make sure the complete animal polygon encloses all visible
animal pixels.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Structure:

{
  "identified_character": "giraffe",

  "locomotion_profile": {
    "type": "quadruped",
    "stride_multiplier": 1.0
  },

  "animal_bbox": [
    ymin,
    xmin,
    ymax,
    xmax
  ],

  "animal_polygon": [
    [y,x],
    [y,x]
  ],

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
      "polygon": [
        [y,x],
        [y,x]
      ]
    },

    {
      "name": "front_left_leg",
      "type": "leg",
      "side": "front_left",

      "polygon": [
        [y,x],
        [y,x]
      ],

      "joints": {
        "proximal": [y,x],
        "middle": [y,x],
        "distal": [y,x]
      }
    }

  ],

  "notes": "short description"
}

For quadrupeds use:

front_left_leg
front_right_leg
back_left_leg
back_right_leg

If only two or three legs are actually visible,
return only those.

If no animal is visible:

{
  "identified_character": "none",
  "animal_bbox": null,
  "animal_polygon": [],
  "complete_animal_check": {
    "head_included": false,
    "ears_included": false,
    "tail_included": false,
    "all_visible_legs_included": false,
    "small_visible_parts_included": false
  },
  "parts": [],
  "notes": "No clear animal detected"
}
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

                response = (
                    client.models.generate_content(
                        model=current_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.05,
                        ),
                    )
                )

            except Exception:

                response = (
                    client.models.generate_content(
                        model=current_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.05,
                        ),
                    )
                )

            text = clean_json_text(
                safe_text(response)
            )

            if not text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            data = json.loads(text)

            if not isinstance(
                data,
                dict,
            ):
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
        "Gemini analysis failed after trying all "
        "available models."
    )

    with st.expander(
        "Show model attempts"
    ):

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
        int(x * width / 100.0),
        int(y * height / 100.0),
    )


def poly_px(
    polygon: List[List[float]],
    width: int,
    height: int,
) -> np.ndarray:

    pts = []

    for p in polygon:

        if (
            isinstance(
                p,
                (list, tuple),
            )
            and len(p) >= 2
        ):

            x, y = pt_px(
                p,
                width,
                height,
            )

            pts.append(
                [x, y]
            )

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

    x, y, bw, bh = cv2.boundingRect(
        pts
    )

    return (
        max(0, x - margin),
        max(0, y - margin),
        min(width, x + bw + margin),
        min(height, y + bh + margin),
    )


# ============================================================
# FOREGROUND / ORIGINAL PIXEL EXTRACTION
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

    _, S, V = cv2.split(
        hsv
    )

    colored = S > 35

    strong_color = (
        (S > 55)
        & (V > 45)
    )

    color_support = cv2.dilate(
        strong_color.astype(
            np.uint8
        ),
        np.ones(
            (5, 5),
            np.uint8,
        ),
    ) > 0

    dark = V < 145

    if shadow_suppress:

        dark_attached = (
            dark
            & color_support
        )

        foreground = (
            colored
            | dark_attached
        )

    else:

        foreground = (
            colored
            | dark
        )

    foreground = (
        foreground.astype(
            np.uint8
        )
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

        foreground = cv2.dilate(
            foreground,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    dilation * 2 + 1,
                    dilation * 2 + 1,
                ),
            ),
        )

    return foreground


# ============================================================
# PREPARE BLENDER PART DATA
# ============================================================

def leg_parts(
    scene: Dict[str, Any],
) -> List[Dict[str, Any]]:

    result = []

    for part in scene.get(
        "parts",
        [],
    ):

        if not isinstance(
            part,
            dict,
        ):
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


def prepare_blender_part(
    image_bgr: np.ndarray,
    polygon: List[List[float]],
    dilation: int = 3,
) -> Optional[Dict[str, Any]]:

    h, w = image_bgr.shape[:2]

    if len(polygon) < 3:
        return None

    region_mask = poly_mask(
        (h, w),
        polygon,
        dilation=max(
            2,
            dilation,
        ),
    )

    mask = ink_mask(
        image_bgr,
        region_mask,
        shadow_suppress=SHADOW_SUPPRESSION,
        dilation=dilation,
    )

    x1, y1, x2, y2 = bbox_poly(
        polygon,
        w,
        h,
        margin=max(
            10,
            min(h, w) // 80,
        ),
    )

    crop = image_bgr[
        y1:y2,
        x1:x2
    ].copy()

    local_mask = mask[
        y1:y2,
        x1:x2
    ].copy()

    if crop.size == 0:
        return None

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGBA,
    )

    rgba[:, :, 3] = local_mask

    return {
        "bbox": (
            x1,
            y1,
            x2,
            y2,
        ),
        "rgba": rgba,
        "polygon": polygon,
    }


# ============================================================
# DETECTION OVERLAY
# ============================================================

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


# ============================================================
# BLENDER DISCOVERY
# ============================================================

def find_blender(
    configured_path: str,
) -> Optional[str]:

    configured_path = (
        configured_path or ""
    ).strip()

    if configured_path:

        p = Path(
            configured_path
        )

        if p.exists():
            return str(p)

    found = shutil.which(
        "blender"
    )

    if found:
        return found

    windows_candidates = [
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    ]

    for candidate in windows_candidates:

        if os.path.exists(candidate):
            return candidate

    return None


# ============================================================
# BLENDER PYTHON SCRIPT GENERATOR
# ============================================================

def generate_blender_script(
    image_path: str,
    output_path: str,
    scene_json_path: str,
) -> str:

    resolution_map = {
        "512 × 512": 512,
        "768 × 768": 768,
        "1024 × 1024": 1024,
    }

    resolution = resolution_map[
        BLENDER_RESOLUTION
    ]

    config = {
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

    config_json = json.dumps(
        config
    )

    script = f'''
import bpy
import json
import math
import os
import mathutils


# ============================================================
# INPUTS
# ============================================================

IMAGE_PATH = {image_path!r}
OUTPUT_PATH = {output_path!r}
SCENE_JSON_PATH = {scene_json_path!r}

CONFIG = json.loads(
    {config_json!r}
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


# ============================================================
# LOAD SCENE JSON
# ============================================================

with open(
    SCENE_JSON_PATH,
    "r",
    encoding="utf-8"
) as f:
    DATA = json.load(f)


# ============================================================
# WORLD
# ============================================================

world = bpy.context.scene.world

if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

world.use_nodes = True

bg = world.node_tree.nodes.get(
    "Background"
)

if bg:
    bg.inputs["Color"].default_value = (
        1.0,
        1.0,
        1.0,
        1.0
    )

    bg.inputs["Strength"].default_value = 0.8


# ============================================================
# IMAGE
# ============================================================

image = bpy.data.images.load(
    IMAGE_PATH,
    check_existing=True
)


# ============================================================
# MATERIAL
# ============================================================

def make_material():

    mat = bpy.data.materials.new(
        "Original_Drawing_Material"
    )

    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

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

    shader.inputs["Roughness"].default_value = 0.8

    links.new(
        texture.outputs["Color"],
        shader.inputs["Base Color"]
    )

    links.new(
        texture.outputs["Alpha"],
        shader.inputs["Alpha"]
    )

    shader.inputs["Alpha"].default_value = 1.0

    links.new(
        shader.outputs["BSDF"],
        output.inputs["Surface"]
    )

    mat.surface_render_method = 'DITHERED'

    return mat


MATERIAL = make_material()


# ============================================================
# NORMALIZED POINT → BLENDER
# ============================================================

def convert_point(p):

    y = float(p[0])
    x = float(p[1])

    # x: 0..100 -> -5..5
    # y: 0..100 -> 5..-5

    bx = (
        x / 100.0
        - 0.5
    ) * 10.0

    by = (
        0.5
        - y / 100.0
    ) * 10.0

    return (
        bx,
        by
    )


# ============================================================
# POLYGON MESH
# ============================================================

def make_polygon_object(
    name,
    polygon,
    z_depth=0.0
):

    if not polygon:
        return None

    points = []

    for p in polygon:

        if (
            isinstance(p, (list, tuple))
            and len(p) >= 2
        ):

            x, y = convert_point(p)

            points.append(
                (
                    x,
                    y,
                    0.0
                )
            )

    if len(points) < 3:
        return None

    mesh = bpy.data.meshes.new(
        name + "_Mesh"
    )

    verts = points

    faces = [
        tuple(
            range(
                len(verts)
            )
        )
    ]

    mesh.from_pydata(
        verts,
        [],
        faces
    )

    mesh.update()

    obj = bpy.data.objects.new(
        name,
        mesh
    )

    bpy.context.collection.objects.link(
        obj
    )

    obj.data.materials.append(
        MATERIAL
    )

    # --------------------------------------------------------
    # SOLIDIFY
    # --------------------------------------------------------

    solid = obj.modifiers.new(
        "3D_Extrusion",
        "SOLIDIFY"
    )

    solid.thickness = CONFIG[
        "extrusion"
    ]

    # --------------------------------------------------------
    # BEVEL
    # --------------------------------------------------------

    bevel_amount = CONFIG[
        "bevel"
    ]

    if bevel_amount > 0:

        bevel = obj.modifiers.new(
            "Soft_3D_Edge",
            "BEVEL"
        )

        bevel.width = bevel_amount
        bevel.segments = 3

    return obj


# ============================================================
# CREATE COMPLETE ANIMAL BODY
# ============================================================

animal_polygon = DATA.get(
    "animal_polygon",
    []
)

body = make_polygon_object(
    "Animal_Body",
    animal_polygon
)

if body is None:

    raise RuntimeError(
        "Could not create complete animal mesh."
    )


# ============================================================
# CREATE LEG OBJECTS
# ============================================================

legs = []

for index, part in enumerate(
    DATA.get("parts", [])
):

    if not isinstance(
        part,
        dict
    ):
        continue

    if str(
        part.get(
            "type",
            ""
        )
    ).lower() != "leg":

        continue

    polygon = (
        part.get(
            "polygon",
            []
        )
        or []
    )

    obj = make_polygon_object(
        part.get(
            "name",
            f"Leg_{{index}}"
        ),
        polygon
    )

    if obj is None:
        continue

    joints = (
        part.get(
            "joints",
            {}
        )
        or {}
    )

    proximal = joints.get(
        "proximal"
    )

    if proximal:

        px, py = convert_point(
            proximal
        )

        # Put origin at proximal joint.
        obj.location.x = px
        obj.location.y = py

        # Move mesh opposite to origin.
        for vertex in obj.data.vertices:

            vertex.co.x -= px
            vertex.co.y -= py

    legs.append(
        {
            "object": obj,
            "side": str(
                part.get(
                    "side",
                    part.get(
                        "name",
                        ""
                    )
                )
            ).lower(),
        }
    )


# ============================================================
# LEG PHASE
# ============================================================

def leg_phase(
    side,
    index
):

    if "front_left" in side:
        return 0.0

    if "back_right" in side:
        return 0.0

    if "front_right" in side:
        return math.pi

    if "back_left" in side:
        return math.pi

    return (
        0.0
        if index % 2 == 0
        else math.pi
    )


# ============================================================
# WALK ANIMATION
# ============================================================

scene = bpy.context.scene

scene.frame_start = 1
scene.frame_end = CONFIG[
    "total_frames"
]

scene.render.fps = CONFIG[
    "fps"
]

scene.render.resolution_x = CONFIG[
    "resolution"
]

scene.render.resolution_y = CONFIG[
    "resolution"
]

scene.render.resolution_percentage = 100


# ============================================================
# BODY BOB
# ============================================================

body_start_z = body.location.z

for frame in range(
    1,
    CONFIG["total_frames"] + 1
):

    t = (
        frame - 1
    ) / max(
        1,
        CONFIG["total_frames"] - 1
    )

    bob = math.sin(
        t
        * math.pi
        * 2.0
        * max(
            1,
            CONFIG["walk_cycles"]
        )
    )

    body.location.z = (
        body_start_z
        + bob
        * CONFIG["body_bob"]
        * 10.0
    )

    body.keyframe_insert(
        data_path="location",
        index=2,
        frame=frame
    )


# ============================================================
# LEG WALK
# ============================================================

for index, leg_data in enumerate(
    legs
):

    obj = leg_data[
        "object"
    ]

    side = leg_data[
        "side"
    ]

    phase = leg_phase(
        side,
        index
    )

    base_rotation = obj.rotation_euler.z

    for frame in range(
        1,
        CONFIG["total_frames"] + 1
    ):

        t = (
            frame - 1
        ) / max(
            1,
            CONFIG["total_frames"] - 1
        )

        gait = math.sin(
            t
            * math.pi
            * 2.0
            * max(
                1,
                CONFIG["walk_cycles"]
            )
            + phase
        )

        angle = (
            math.radians(
                CONFIG["step_angle"]
            )
            * gait
        )

        if "back" in side:
            angle *= 0.90

        obj.rotation_euler.z = (
            base_rotation
            + angle
        )

        obj.keyframe_insert(
            data_path="rotation_euler",
            index=2,
            frame=frame
        )


# ============================================================
# CAMERA
# ============================================================

bpy.ops.object.camera_add(
    location=(
        0.0,
        0.0,
        14.0
    )
)

camera = bpy.context.object

camera.data.type = "ORTHO"

camera.data.ortho_scale = 11.5

camera.rotation_euler = (
    0.0,
    0.0,
    0.0
)

# Point camera downward toward XY plane.
camera.rotation_euler = (
    0.0,
    0.0,
    0.0
)

scene.camera = camera


# ============================================================
# LIGHT
# ============================================================

bpy.ops.object.light_add(
    type="AREA",
    location=(
        0,
        0,
        8
    )
)

light = bpy.context.object

light.data.energy = 800

light.data.shape = "DISK"

light.data.size = 10


# ============================================================
# RENDER
# ============================================================

scene.render.image_settings.file_format = "PNG"

scene.render.film_transparent = False

scene.render.filepath = OUTPUT_PATH

scene.render.engine = "BLENDER_EEVEE_NEXT"

scene.render.resolution_x = CONFIG[
    "resolution"
]

scene.render.resolution_y = CONFIG[
    "resolution"
]

scene.render.resolution_percentage = 100

scene.render.fps = CONFIG[
    "fps"
]


# ============================================================
# SAVE BLEND
# ============================================================

blend_path = os.path.splitext(
    OUTPUT_PATH
)[0] + ".blend"

bpy.ops.wm.save_as_mainfile(
    filepath=blend_path
)


# ============================================================
# RENDER ANIMATION
# ============================================================

scene.render.filepath = (
    os.path.splitext(
        OUTPUT_PATH
    )[0]
    + "_"
    + "####"
    + ".png"
)

bpy.ops.render.render(
    animation=True
)

print(
    "BLENDER_ANIMATION_COMPLETE"
)
'''

    return script


# ============================================================
# CREATE BLENDER PROJECT
# ============================================================

def run_blender_animation(
    image_pil: Image.Image,
    scene: Dict[str, Any],
) -> Optional[Dict[str, str]]:

    blender = find_blender(
        BLENDER_PATH
    )

    if blender is None:

        st.error(
            "Blender executable was not found."
        )

        st.info(
            "Install Blender and either add it to PATH "
            "or enter the full blender.exe path in the sidebar."
        )

        return None

    work_dir = tempfile.mkdtemp(
        prefix="animal_blender_"
    )

    image_path = os.path.join(
        work_dir,
        "original_animal.png",
    )

    scene_json_path = os.path.join(
        work_dir,
        "scene.json",
    )

    blender_script_path = os.path.join(
        work_dir,
        "animate.py",
    )

    output_path = os.path.join(
        work_dir,
        "animal_animation",
    )

    image_pil.save(
        image_path,
        format="PNG",
    )

    with open(
        scene_json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            scene,
            f,
            indent=2,
        )

    blender_script = generate_blender_script(
        image_path,
        output_path,
        scene_json_path,
    )

    with open(
        blender_script_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            blender_script
        )

    st.session_state.blender_script = (
        blender_script
    )

    command = [
        blender,
        "--background",
        "--python",
        blender_script_path,
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
        )

    except subprocess.TimeoutExpired:

        st.error(
            "Blender rendering timed out."
        )

        return None

    except Exception as exc:

        st.error(
            f"Could not start Blender: {exc}"
        )

        return None

    if result.returncode != 0:

        st.error(
            "Blender returned an error."
        )

        with st.expander(
            "Blender console output"
        ):

            st.code(
                result.stdout
                + "\n"
                + result.stderr
            )

        return None

    blend_path = (
        output_path
        + ".blend"
    )

    frame_pattern = (
        output_path
        + "_"
    )

    return {
        "work_dir": work_dir,
        "blend_path": blend_path,
        "frame_pattern": frame_pattern,
        "stdout": result.stdout,
    }


# ============================================================
# FRAMES → GIF
# ============================================================

def collect_blender_frames(
    frame_pattern: str,
) -> List[np.ndarray]:

    directory = os.path.dirname(
        frame_pattern
    )

    prefix = os.path.basename(
        frame_pattern
    )

    files = []

    if not os.path.exists(
        directory
    ):
        return []

    for name in os.listdir(
        directory
    ):

        if not name.startswith(
            prefix
        ):
            continue

        if not name.lower().endswith(
            ".png"
        ):
            continue

        files.append(
            os.path.join(
                directory,
                name,
            )
        )

    files.sort()

    frames = []

    for path in files:

        img = cv2.imread(
            path,
            cv2.IMREAD_COLOR,
        )

        if img is not None:
            frames.append(img)

    return frames


def gif_bytes(
    frames: List[np.ndarray],
    fps: int,
) -> bytes:

    if not frames:
        return b""

    pil_frames = []

    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_frames.append(
            Image.fromarray(rgb)
        )

    buffer = io.BytesIO()

    duration = int(
        1000
        / max(
            1,
            fps,
        )
    )

    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )

    return buffer.getvalue()


# ============================================================
# ZIP
# ============================================================

def zip_files(
    files: List[str],
) -> bytes:

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        for path in files:

            if os.path.exists(path):

                archive.write(
                    path,
                    os.path.basename(
                        path
                    ),
                )

    return buffer.getvalue()


# ============================================================
# UI — UPLOAD
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

    st.info(
        "Upload a hand-drawn animal image to begin."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

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

    st.error(
        f"Could not read uploaded image: {exc}"
    )

    st.stop()


# ============================================================
# GEMINI PNG
# ============================================================

gemini_buffer = io.BytesIO()

image_pil.save(
    gemini_buffer,
    format="PNG",
)

gemini_bytes = (
    gemini_buffer.getvalue()
)


# ============================================================
# ORIGINAL
# ============================================================

c1, c2 = st.columns(2)

with c1:

    st.image(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        ),
        caption="Original drawing",
        width="stretch",
    )

with c2:

    st.markdown(
        """
### 🧠 Blender animation pipeline

1. Gemini recognizes the COMPLETE animal.
2. Tiny ears, paws, tail tips and other details are protected.
3. Original pixels are preserved.
4. Gemini identifies visible legs and anatomical pivots.
5. Blender creates separate body and leg geometry.
6. The drawing receives shallow 3D depth.
7. Legs rotate around their proximal joints.
8. Blender creates the walking animation.
9. Blender renders the final frames.
10. The result can be exported as a GIF/MP4.
        """
    )


# ============================================================
# GEMINI
# ============================================================

if st.button(
    "🔍 Analyze drawing with Gemini",
    type="primary",
    width="stretch",
):

    with st.spinner(
        "Analyzing complete animal anatomy..."
    ):

        scene = analyze_scene(
            gemini_bytes,
            GEMINI_API_KEY,
        )

    if scene:

        st.session_state.scene = scene
        st.session_state.blender_result = None


# ============================================================
# STOP
# ============================================================

if not st.session_state.scene:

    st.stop()


scene = st.session_state.scene


# ============================================================
# DETECTION
# ============================================================

st.success(
    f"Detected: "
    f"**{scene.get('identified_character', 'character')}**"
)

with st.expander(
    "🧬 Gemini anatomy JSON"
):

    st.json(scene)


overlay = detection_overlay(
    image_bgr,
    scene,
)

st.image(
    cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    ),
    caption=(
        "Green = COMPLETE animal boundary | "
        "Orange = animation parts | "
        "Red = anatomical joints"
    ),
    width="stretch",
)


# ============================================================
# COMPLETE ANIMAL CHECK
# ============================================================

check = (
    scene.get(
        "complete_animal_check",
        {}
    )
    or {}
)

checks = [
    (
        "Head",
        check.get(
            "head_included",
            False
        ),
    ),
    (
        "Ears",
        check.get(
            "ears_included",
            False
        ),
    ),
    (
        "Tail",
        check.get(
            "tail_included",
            False
        ),
    ),
    (
        "Visible legs",
        check.get(
            "all_visible_legs_included",
            False
        ),
    ),
    (
        "Small parts",
        check.get(
            "small_visible_parts_included",
            False
        ),
    ),
]

cols = st.columns(
    len(checks)
)

for col, (label, value) in zip(
    cols,
    checks,
):

    with col:

        if value:
            st.success(
                f"✓ {label}"
            )
        else:
            st.warning(
                f"⚠ {label}"
            )


# ============================================================
# BLENDER STATUS
# ============================================================

st.markdown("---")

st.header(
    "🧊 Step 2 — Blender 3D animation"
)

blender_path_found = find_blender(
    BLENDER_PATH
)

if blender_path_found:

    st.success(
        f"Blender found: `{blender_path_found}`"
    )

else:

    st.warning(
        "Blender was not found on this computer."
    )


st.write(
    "The original drawing remains the source artwork. "
    "Blender adds shallow depth and handles the actual "
    "animation and rendering."
)


# ============================================================
# GENERATE BLENDER
# ============================================================

if st.button(
    "🚀 Create Blender animation",
    type="primary",
    width="stretch",
):

    if not blender_path_found:

        st.error(
            "Please install Blender or provide the "
            "Blender executable path."
        )

    else:

        with st.spinner(
            "Blender is creating the 3D animal and rendering..."
        ):

            result = run_blender_animation(
                image_pil,
                scene,
            )

        if result:

            st.session_state.blender_result = (
                result
            )

            st.success(
                "✅ Blender animation completed."
            )


# ============================================================
# BLENDER RESULT
# ============================================================

result = st.session_state.blender_result

if result:

    frame_pattern = result[
        "frame_pattern"
    ]

    frames = collect_blender_frames(
        frame_pattern
    )

    if frames:

        st.header(
            "🎬 Blender result"
        )

        st.success(
            f"Rendered {len(frames)} Blender frames."
        )

        gif_data = gif_bytes(
            frames,
            FPS,
        )

        st.image(
            gif_data,
            caption="Blender-rendered animation",
            width="stretch",
        )

        st.download_button(
            "⬇️ Download Blender GIF",
            gif_data,
            "blender_animal_animation.gif",
            "image/gif",
            width="stretch",
        )

        png_files = []

        directory = os.path.dirname(
            frame_pattern
        )

        prefix = os.path.basename(
            frame_pattern
        )

        for name in sorted(
            os.listdir(directory)
        ):

            if (
                name.startswith(prefix)
                and name.endswith(".png")
            ):

                png_files.append(
                    os.path.join(
                        directory,
                        name,
                    )
                )

        if png_files:

            zip_data = zip_files(
                png_files
            )

            st.download_button(
                "⬇️ Download Blender PNG frames",
                zip_data,
                "blender_animation_frames.zip",
                "application/zip",
                width="stretch",
            )

        blend_path = result[
            "blend_path"
        ]

        if os.path.exists(
            blend_path
        ):

            with open(
                blend_path,
                "rb"
            ) as f:

                blend_data = f.read()

            st.download_button(
                "⬇️ Download Blender project",
                blend_data,
                "animal_animation.blend",
                "application/octet-stream",
                width="stretch",
            )

    else:

        st.warning(
            "Blender completed, but no rendered frames "
            "were found."
        )


# ============================================================
# BLENDER SCRIPT DOWNLOAD
# ============================================================

if st.session_state.blender_script:

    st.markdown("---")

    st.subheader(
        "🧩 Generated Blender Python script"
    )

    st.download_button(
        "⬇️ Download Blender Python script",
        st.session_state.blender_script,
        "animal_blender_animation.py",
        "text/plain",
        width="stretch",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🦒 Blender 3D — Gemini complete-animal recognition, "
    "original drawing preservation, shallow 3D extrusion, "
    "anatomical leg pivots and Blender-based animation."
)
