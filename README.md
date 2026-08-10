# Virtual Try-On Assessment - Submission

**Candidate name:** Abel Philip Thomas  
**Email:** abelphilip482@gmail.com
**Date:** August 9, 2026  
**GitHub repo link:** https://github.com/abelphilip482-droid/Xapien-Assessment  
**Demo video link (max 5 min):** To be added  
**Colab notebook links (if used):** To be added  

---

## Q1 - Garment & Body Understanding

### VLM chosen and why:

**MiniCPM-V 2.6** was selected for Q1. It is an open-source vision-language model released under the Apache 2.0 license, satisfying the assessment requirement for an open and freely downloadable model.

The quantized `MiniCPM-V-2_6-int4` version was used to make inference practical on the available Google Colab GPU environment.

MiniCPM-V 2.6 was selected because it supports visual understanding of both person and garment images and can extract structured attributes such as garment type, sleeve length, neckline, primary color, pattern, body visibility, and other visual information.

A MediaPipe pose-landmark-based classifier is additionally used for pose classification. This provides a deterministic method for the required pose categories and ensures that the dedicated side-facing and seated edge cases are classified correctly.

The final Q1 JSON outputs follow the structure provided in `sample_output_q1.json`.

### How to run:

The Q1 pipeline loads MiniCPM-V 2.6 and processes the provided person and garment images.

The main processing steps are:

1. Load the person image.
2. Detect body landmarks for pose and visibility analysis.
3. Analyze the person image using MiniCPM-V 2.6.
4. Analyze the garment image using MiniCPM-V 2.6.
5. Normalize the garment attributes into the required categories.
6. Combine the person and garment attributes into the required JSON structure.
7. Save the resulting JSON files.

### Known limitations:

- Pose classification may be less reliable when the body is heavily occluded.
- VLM-generated garment attributes can occasionally be uncertain for visually ambiguous or partially occluded regions.
- The quantized model may produce slower inference depending on the available Colab GPU resources.

### Q1 validation:

The dedicated pose edge cases were tested successfully:

| Image | Expected | Predicted | Status |
|---|---|---|---|
| `person_side_pose.jpg` | side | side | PASS |
| `person_seated.jpg` | seated | seated | PASS |
| `edge_person_crossed_arms.json` | seated | seated | PASS |

### Q1 outputs:

The Q1 output directory contains JSON results for the provided person and garment images and the required edge-case person images.

---

## Q2 - Human Parsing & Segmentation

- Models used (parsing / background removal): To be completed.
- How to run: To be completed.
- Edge cases handled / failed: To be completed.

## Q3 - End-to-End Try-On

- Try-on model chosen and why: To be completed.
- Hardware used (GPU, VRAM): To be completed.
- Constraints hit and workarounds: To be completed.
- How to run: To be completed.

## Q4 - Automated Quality Evaluation

- Metrics implemented: To be completed.
- VLM-as-judge rubric prompt (paste it here): To be completed.
- Results: `evaluation_template_q4.csv` will be completed and committed to the repository.

## Q5 - Web Demo

- Framework (Gradio/Streamlit): To be completed.
- How to launch: To be completed.
- Guardrails implemented: To be completed.

## Honest failure log

### Q1

- MiniCPM-V 2.6 successfully processed the provided person and garment images.
- A safe pose-classification wrapper was implemented to handle cases where no pose landmarks are detected.
- The side-facing and seated edge cases were classified correctly.
- The no-person edge case was detected correctly.
- Non-fatal warnings from Transformers/bitsandbytes were observed during inference but did not prevent successful processing.

Additional failures and limitations for Q2-Q5 will be documented as those components are implemented.
