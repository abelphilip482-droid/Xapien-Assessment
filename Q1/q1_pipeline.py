# Q1 - Garment & Body Understanding
# Full pipeline will be added here.

# ============================================================
# Q1 — COMPLETE PIPELINE
# MiniCPM-V 2.6 + MediaPipe Pose Landmarker
# ============================================================

import os
import re
import json
import argparse

import numpy as np
import torch
import mediapipe as mp

from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openbmb/MiniCPM-V-2_6-int4"
MODEL_USED = "MiniCPM-V-2.6"

DEFAULT_BASE_DIR = "/content/sample_files"
DEFAULT_OUTPUT_DIR = "/content/q1_outputs"
DEFAULT_POSE_MODEL = "/content/pose_landmarker.task"


# ============================================================
# PATHS
# ============================================================

def get_paths(base_dir, output_dir):

    person_dir = os.path.join(
        base_dir,
        "test_pairs",
        "person"
    )

    garment_dir = os.path.join(
        base_dir,
        "test_pairs",
        "garment"
    )

    edge_dir = os.path.join(
        base_dir,
        "edge_cases"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    return (
        person_dir,
        garment_dir,
        edge_dir,
        output_dir
    )


# ============================================================
# MINICPM-V MODEL LOADING
# ============================================================

def load_minicpm():

    print("=" * 60)
    print("Loading MiniCPM-V 2.6 INT4")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required to run MiniCPM-V 2.6."
        )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print("Loading model...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    model = model.to("cuda")
    model.eval()

    print("MiniCPM-V 2.6 loaded successfully.")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    return model, tokenizer, processor


# ============================================================
# MEDIAPIPE POSE LANDMARKER
# ============================================================

def load_landmarker(pose_model_path):

    print("=" * 60)
    print("Loading MediaPipe Pose Landmarker")
    print("=" * 60)

    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(
            "Pose landmarker model not found:\n"
            + pose_model_path
        )

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=pose_model_path
        ),
        running_mode=VisionRunningMode.IMAGE
    )

    landmarker = (
        PoseLandmarker.create_from_options(
            options
        )
    )

    print(
        "Pose Landmarker loaded successfully."
    )

    return landmarker


# ============================================================
# MINICPM INFERENCE
# ============================================================

def ask_minicpm(
    image_path,
    question,
    model,
    tokenizer
):

    image = Image.open(
        image_path
    ).convert("RGB")

    msgs = [
        {
            "role": "user",
            "content": [
                image,
                question
            ]
        }
    ]

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    return str(answer).strip()


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    if not isinstance(text, str):
        return {}

    text = re.sub(
        r"```json|```",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        return {}

    try:
        return json.loads(
            match.group(0)
        )

    except Exception:
        return {}


# ============================================================
# POSE DETECTION
# ============================================================

def get_pose(
    image,
    landmarker
):

    image_np = np.array(
        image.convert("RGB")
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    result = landmarker.detect(
        mp_image
    )

    if not result.pose_landmarks:
        return None

    return result.pose_landmarks[0]


def detect_landmarks(
    image_path,
    landmarker
):

    image = Image.open(
        image_path
    ).convert("RGB")

    return get_pose(
        image,
        landmarker
    )


# ============================================================
# POSE CLASSIFICATION
# ============================================================

def classify_pose(landmarks):

    if landmarks is None:

        return (
            "unknown",
            "No body pose detected."
        )

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_HIP = 23
    RIGHT_HIP = 24

    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    try:

        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]

        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]

        lk = landmarks[LEFT_KNEE]
        rk = landmarks[RIGHT_KNEE]

    except Exception:

        return (
            "unknown",
            "Incomplete pose landmarks."
        )

    hip_visibility = min(
        lh.visibility,
        rh.visibility
    )

    knee_visibility = min(
        lk.visibility,
        rk.visibility
    )

    hip_y = (
        lh.y + rh.y
    ) / 2

    knee_y = (
        lk.y + rk.y
    ) / 2

    knee_hip_delta = (
        knee_y - hip_y
    )

    # --------------------------------------------------------
    # Seated
    # --------------------------------------------------------

    if (
        hip_visibility > 0.5
        and knee_visibility > 0.45
        and knee_hip_delta < 0.13
    ):

        return (
            "seated",
            (
                "Knees are close to hip level "
                f"(delta={knee_hip_delta:.3f})."
            )
        )

    # --------------------------------------------------------
    # Side pose
    # --------------------------------------------------------

    shoulder_width = abs(
        ls.x - rs.x
    )

    if shoulder_width < 0.20:

        return (
            "side",
            (
                "Narrow apparent shoulder span "
                f"({shoulder_width:.3f})."
            )
        )

    # --------------------------------------------------------
    # Front-facing
    # --------------------------------------------------------

    return (
        "front-facing",
        (
            "Broad apparent shoulder span "
            f"({shoulder_width:.3f})."
        )
    )


# ============================================================
# SAFE POSE CLASSIFIER
# ============================================================

def classify_pose_safe(
    image,
    landmarker
):

    try:

        pose_landmarks = get_pose(
            image,
            landmarker
        )

        if pose_landmarks is None:

            return (
                "no_person",
                "No person detected."
            )

        result = classify_pose(
            pose_landmarks
        )

        if isinstance(
            result,
            tuple
        ):

            return (
                result[0],
                result[1]
            )

        if isinstance(
            result,
            dict
        ):

            return (
                result.get(
                    "pose_category",
                    "unknown"
                ),
                result.get(
                    "confidence_notes",
                    ""
                )
            )

        return (
            str(result),
            ""
        )

    except Exception as e:

        print(
            "Pose classification error:",
            e
        )

        return (
            "unknown",
            str(e)
        )


# ============================================================
# BODY VISIBILITY
# ============================================================

def determine_visibility(landmarks):

    if landmarks is None:

        return (
            False,
            False
        )

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_HIP = 23
    RIGHT_HIP = 24

    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    shoulders = [
        landmarks[LEFT_SHOULDER],
        landmarks[RIGHT_SHOULDER]
    ]

    hips = [
        landmarks[LEFT_HIP],
        landmarks[RIGHT_HIP]
    ]

    knees = [
        landmarks[LEFT_KNEE],
        landmarks[RIGHT_KNEE]
    ]

    upper_body_visible = (
        min(
            x.visibility
            for x in shoulders
        ) > 0.40

        and

        min(
            x.visibility
            for x in hips
        ) > 0.40
    )

    lower_body_visible = (
        max(
            x.visibility
            for x in knees
        ) > 0.30
    )

    return (
        upper_body_visible,
        lower_body_visible
    )


# ============================================================
# GARMENT NORMALIZATION
# ============================================================

def normalize_garment(data):

    if not isinstance(
        data,
        dict
    ):
        data = {}

    garment_type = str(
        data.get(
            "type",
            "other"
        )
    ).strip().lower()

    sleeve = str(
        data.get(
            "sleeve_length",
            "unknown"
        )
    ).strip().lower()

    neckline = str(
        data.get(
            "neckline",
            "unknown"
        )
    ).strip().lower()

    color = str(
        data.get(
            "primary_color",
            "unknown"
        )
    ).strip().lower()

    pattern = str(
        data.get(
            "pattern",
            "other"
        )
    ).strip().lower()

    # Garment type

    if "tank" in garment_type:

        garment_type = "tank-top"

    elif (
        "t-shirt" in garment_type
        or garment_type == "tee"
    ):

        garment_type = "t-shirt"

    # Sleeve

    if (
        "sleeveless" in sleeve
        or "no sleeve" in sleeve
    ):

        sleeve = "sleeveless"

    elif "short" in sleeve:

        sleeve = "short"

    elif "long" in sleeve:

        sleeve = "long"

    else:

        sleeve = "unknown"

    # Neckline

    if neckline in [
        "round",
        "round neck",
        "crew neck",
        "crew-neck"
    ]:

        neckline = "crew"

    elif "crew" in neckline:

        neckline = "crew"

    elif (
        "v-neck" in neckline
        or "v neck" in neckline
    ):

        neckline = "v-neck"

    elif "square" in neckline:

        neckline = "square"

    elif "scoop" in neckline:

        neckline = "scoop"

    elif "high" in neckline:

        neckline = "high neck"

    # Pattern

    if pattern in [
        "plain",
        "none",
        "no pattern"
    ]:

        pattern = "solid"

    return {
        "type": garment_type,
        "sleeve_length": sleeve,
        "neckline": neckline,
        "primary_color": color,
        "pattern": pattern
    }


# ============================================================
# PROMPTS
# ============================================================

GARMENT_QUESTION = """
Analyze this garment image.

Return ONLY a JSON object with exactly these fields:

{
"type": "t-shirt | shirt | tank-top | blouse | dress | other",
"sleeve_length": "sleeveless | short | long | unknown",
"neckline": "crew | v-neck | square | scoop | high neck | other | unknown",
"primary_color": "color",
"pattern": "solid | graphic print | striped | checked | floral | other"
}

Use your best visual judgment.
Do not include explanations.
"""


PERSON_QUESTION = """
Analyze this person's image.

Determine:

1. Whether the upper body is visible.
2. Whether the lower body is visible.

Return ONLY valid JSON:

{
"upper_body_visible": true,
"lower_body_visible": true
}

Do not classify the pose.
Do not add any explanation.
"""


# ============================================================
# GARMENT ANALYSIS
# ============================================================

def analyze_garment(
    image_path,
    model,
    tokenizer
):

    raw = ask_minicpm(
        image_path,
        GARMENT_QUESTION,
        model,
        tokenizer
    )

    parsed = extract_json(
        raw
    )

    attributes = normalize_garment(
        parsed
    )

    return (
        attributes,
        raw
    )


# ============================================================
# PERSON ANALYSIS
# ============================================================

def analyze_person(
    image_path,
    model,
    tokenizer,
    landmarker
):

    image = Image.open(
        image_path
    ).convert("RGB")

    # Safe pose classification

    pose, pose_note = (
        classify_pose_safe(
            image,
            landmarker
        )
    )

    # Landmark-based visibility

    landmarks = detect_landmarks(
        image_path,
        landmarker
    )

    mp_upper, mp_lower = (
        determine_visibility(
            landmarks
        )
    )

    # MiniCPM analysis

    raw = ask_minicpm(
        image_path,
        PERSON_QUESTION,
        model,
        tokenizer
    )

    vlm = extract_json(
        raw
    )

    if landmarks is not None:

        upper_visible = bool(
            mp_upper
        )

        lower_visible = bool(
            mp_lower
        )

    else:

        upper_visible = bool(
            vlm.get(
                "upper_body_visible",
                False
            )
        )

        lower_visible = bool(
            vlm.get(
                "lower_body_visible",
                False
            )
        )

    return (
        {
            "pose_category": pose,
            "upper_body_visible": upper_visible,
            "lower_body_visible": lower_visible
        },
        pose_note,
        raw
    )


# ============================================================
# JSON SAVING
# ============================================================

def save_json(
    output_path,
    data
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


# ============================================================
# PROCESS PERSON IMAGES
# ============================================================

def process_person_images(
    person_dir,
    output_dir,
    model,
    tokenizer,
    landmarker
):

    results = {}

    for i in range(1, 6):

        filename = (
            f"person_{i:02d}.png"
        )

        path = os.path.join(
            person_dir,
            filename
        )

        if not os.path.exists(path):

            print(
                f"Missing: {filename}"
            )

            continue

        print(
            f"Processing person: {filename}"
        )

        try:

            attributes, pose_note, raw = (
                analyze_person(
                    path,
                    model,
                    tokenizer,
                    landmarker
                )
            )

            result = {
                "person_image": filename,
                "person_attributes": attributes,
                "model_used": MODEL_USED,
                "confidence_notes": pose_note
            }

            results[
                filename
            ] = result

            output_path = os.path.join(
                output_dir,
                filename.replace(
                    ".png",
                    ".json"
                )
            )

            save_json(
                output_path,
                result
            )

            print(
                "  Pose:",
                attributes[
                    "pose_category"
                ]
            )

            print(
                "  Upper:",
                attributes[
                    "upper_body_visible"
                ]
            )

            print(
                "  Lower:",
                attributes[
                    "lower_body_visible"
                ]
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    return results


# ============================================================
# PROCESS GARMENT IMAGES
# ============================================================

def process_garment_images(
    garment_dir,
    output_dir,
    model,
    tokenizer
):

    results = {}

    for i in range(1, 6):

        filename = (
            f"garment_{i:02d}.jpg"
        )

        path = os.path.join(
            garment_dir,
            filename
        )

        if not os.path.exists(path):

            print(
                f"Missing: {filename}"
            )

            continue

        print(
            f"Processing garment: {filename}"
        )

        try:

            attributes, raw = (
                analyze_garment(
                    path,
                    model,
                    tokenizer
                )
            )

            result = {
                "garment_image": filename,
                "garment_attributes": attributes,
                "model_used": MODEL_USED,
                "confidence_notes": (
                    "Attributes extracted "
                    "using MiniCPM-V-2.6."
                )
            }

            results[
                filename
            ] = result

            output_path = os.path.join(
                output_dir,
                filename.replace(
                    ".jpg",
                    ".json"
                )
            )

            save_json(
                output_path,
                result
            )

            print(
                "  Type:",
                attributes["type"]
            )

            print(
                "  Sleeve:",
                attributes["sleeve_length"]
            )

            print(
                "  Neckline:",
                attributes["neckline"]
            )

            print(
                "  Color:",
                attributes["primary_color"]
            )

            print(
                "  Pattern:",
                attributes["pattern"]
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    return results


# ============================================================
# CREATE PAIR OUTPUTS
# ============================================================

def create_pair_outputs(
    person_results,
    garment_results,
    output_dir
):

    pairs = [
        (
            "pair_01",
            "person_01.png",
            "garment_01.jpg"
        ),
        (
            "pair_02",
            "person_02.png",
            "garment_02.jpg"
        ),
        (
            "pair_03",
            "person_03.png",
            "garment_03.jpg"
        )
    ]

    for (
        pair_id,
        person_file,
        garment_file
    ) in pairs:

        if person_file not in person_results:

            print(
                f"Missing person result "
                f"for {pair_id}"
            )

            continue

        if garment_file not in garment_results:

            print(
                f"Missing garment result "
                f"for {pair_id}"
            )

            continue

        person_data = (
            person_results[
                person_file
            ]
        )

        garment_data = (
            garment_results[
                garment_file
            ]
        )

        combined = {
            "person_image": person_file,
            "garment_image": garment_file,
            "garment_attributes":
                garment_data[
                    "garment_attributes"
                ],
            "person_attributes":
                person_data[
                    "person_attributes"
                ],
            "model_used": MODEL_USED,
            "confidence_notes": (
                "Pose: "
                + person_data[
                    "confidence_notes"
                ]
                + " Garment attributes analyzed "
                "using MiniCPM-V-2.6."
            )
        }

        output_path = os.path.join(
            output_dir,
            f"{pair_id}.json"
        )

        save_json(
            output_path,
            combined
        )

        print(
            f"Created {pair_id}.json"
        )


# ============================================================
# EDGE CASE PROCESSING
# ============================================================

def process_edge_cases(
    edge_dir,
    output_dir,
    model,
    tokenizer,
    landmarker
):

    edge_files = [
        "person_crossed_arms.jpg",
        "person_side_pose.jpg",
        "person_seated.jpg"
    ]

    for filename in edge_files:

        path = os.path.join(
            edge_dir,
            filename
        )

        if not os.path.exists(path):

            print(
                f"Missing edge case: {filename}"
            )

            continue

        print(
            f"Processing edge case: {filename}"
        )

        try:

            attributes, pose_note, raw = (
                analyze_person(
                    path,
                    model,
                    tokenizer,
                    landmarker
                )
            )

            result = {
                "person_image": filename,
                "person_attributes": attributes,
                "model_used": MODEL_USED,
                "confidence_notes": pose_note
            }

            output_path = os.path.join(
                output_dir,
                filename.replace(
                    ".jpg",
                    ".json"
                )
            )

            save_json(
                output_path,
                result
            )

            print(
                "  Pose:",
                attributes[
                    "pose_category"
                ]
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )


# ============================================================
# EDGE CASE VERIFICATION
# ============================================================

def verify_edge_cases(
    edge_dir,
    landmarker
):

    print()
    print("=" * 60)
    print("EDGE CASE VERIFICATION")
    print("=" * 60)

    tests = [
        (
            "person_side_pose.jpg",
            "side"
        ),
        (
            "person_seated.jpg",
            "seated"
        ),
        (
            "no_person.jpg",
            "no_person"
        )
    ]

    for filename, expected in tests:

        path = os.path.join(
            edge_dir,
            filename
        )

        if not os.path.exists(path):

            print(
                f"{filename}: FILE MISSING"
            )

            continue

        image = Image.open(
            path
        ).convert("RGB")

        predicted, note = (
            classify_pose_safe(
                image,
                landmarker
            )
        )

        status = (
            "PASS"
            if predicted == expected
            else "FAIL"
        )

        print(
            f"{filename} | "
            f"expected={expected} | "
            f"predicted={predicted} | "
            f"{status}"
        )

        if note:
            print(
                "  Note:",
                note
            )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Q1 MiniCPM-V 2.6 "
            "attribute extraction pipeline"
        )
    )

    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--pose-model",
        default=DEFAULT_POSE_MODEL
    )

    args = parser.parse_args()

    (
        person_dir,
        garment_dir,
        edge_dir,
        output_dir
    ) = get_paths(
        args.base_dir,
        args.output_dir
    )

    print()
    print("=" * 60)
    print("Q1 PIPELINE START")
    print("=" * 60)

    print(
        "Person directory:",
        person_dir
    )

    print(
        "Garment directory:",
        garment_dir
    )

    print(
        "Edge-case directory:",
        edge_dir
    )

    print(
        "Output directory:",
        output_dir
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    model, tokenizer, processor = (
        load_minicpm()
    )

    landmarker = load_landmarker(
        args.pose_model
    )

    # --------------------------------------------------------
    # Person images
    # --------------------------------------------------------

    person_results = (
        process_person_images(
            person_dir,
            output_dir,
            model,
            tokenizer,
            landmarker
        )
    )

    # --------------------------------------------------------
    # Garment images
    # --------------------------------------------------------

    garment_results = (
        process_garment_images(
            garment_dir,
            output_dir,
            model,
            tokenizer
        )
    )

    # --------------------------------------------------------
    # Pair outputs
    # --------------------------------------------------------

    create_pair_outputs(
        person_results,
        garment_results,
        output_dir
    )

    # --------------------------------------------------------
    # Edge cases
    # --------------------------------------------------------

    process_edge_cases(
        edge_dir,
        output_dir,
        model,
        tokenizer,
        landmarker
    )

    # --------------------------------------------------------
    # Verify edge cases
    # --------------------------------------------------------

    verify_edge_cases(
        edge_dir,
        landmarker
    )

    # --------------------------------------------------------
    # Final output listing
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Q1 OUTPUT FILES")
    print("=" * 60)

    json_files = sorted(
        filename
        for filename in os.listdir(
            output_dir
        )
        if filename.endswith(".json")
    )

    for filename in json_files:
        print(filename)

    print()
    print(
        "Total JSON files:",
        len(json_files)
    )

    print()
    print("=" * 60)
    print("Q1 PIPELINE COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
