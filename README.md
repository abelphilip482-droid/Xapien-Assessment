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

* Pose classification may be less reliable when the body is heavily occluded.
* VLM-generated garment attributes can occasionally be uncertain for visually ambiguous or partially occluded regions.
* The quantized model may produce slower inference depending on the available Colab GPU resources.

### Q1 validation:

The dedicated pose edge cases were tested successfully:

| Image                  | Expected | Predicted | Status |
| ---------------------- | -------- | --------- | ------ |
| `person_side_pose.jpg` | side     | side      | PASS   |
| `person_seated.jpg`    | seated   | seated    | PASS   |

The no-person edge case was also detected correctly.

### Q1 outputs:

The Q1 output directory contains JSON results for the provided person and garment images and the required edge-case person images.

---

## Q2 - Human Parsing & Segmentation

* **Models used (parsing / background removal):**
  **Grounding DINO** was used for object/region detection, including the person, face, hair, arms, upper clothing, and lower body. Confidence thresholds of **0.25, 0.15, and 0.10** were used for the detections.

  **SAM (Segment Anything Model)** was used to convert the Grounding DINO bounding boxes into pixel-level segmentation masks and generate the required person/semantic masks.

  **U2Net** was also used for human segmentation/background removal during the Q2 preprocessing pipeline.

* **How to run:**
  Run the Q2 notebook/script. The pipeline loads Grounding DINO, SAM, and U2Net, processes the person images, generates the required segmentation/agnostic representations, and processes the garment images for background removal and masking.

* **Edge cases handled / failed:**
  The crossed-arms edge case (`person_crossed_arms.jpg`) was **not successfully handled** in the human parsing pipeline.

  For the other five person images, the **left-hand region was not parsed correctly**.

  This parsing issue did not affect the final Q3 try-on outputs because the incorrectly parsed hand region did not materially affect the garment-transfer area.

  The remaining person and garment segmentation outputs were generated successfully.

---

## Q3 - End-to-End Try-On

* **Try-on model chosen and why:**
  **CatVTON** was selected as the end-to-end virtual try-on model because it is an open-source diffusion-based try-on model and is relatively lightweight and suitable for running on a free Google Colab T4 GPU.

* **Hardware used (GPU, VRAM):**
  Google Colab **Tesla T4 GPU — 14.56 GB VRAM**. Inference was performed at **384×512 resolution** with **30 inference steps**.

* **Constraints hit and workarounds:**
  GPU memory was a constraint when running the diffusion-based try-on pipeline. The inference configuration was kept at **384×512 resolution** with **30 inference steps** to make the pipeline practical within the available Tesla T4 GPU memory.

* **How to run:**
  Run the Q3 inference notebook/script. The pipeline takes the person image, garment image, and clothing mask produced during preprocessing, passes them to CatVTON, and saves the generated try-on image.

* **Output:**
  End-to-end inference was successfully completed for all five required pairs. The correct person-garment ordering was maintained:

  1. `person_01 + garment_01`
  2. `person_02 + garment_02`
  3. `person_03 + garment_03`
  4. `person_04 + garment_04`
  5. `person_05 + garment_05`

  All five corresponding try-on outputs were successfully generated and saved.

---

## Q4 - Automated Quality Evaluation

* **Metrics implemented:** To be completed.
* **VLM-as-judge rubric prompt (paste it here):** To be completed.
* **Results:** `evaluation_template_q4.csv` will be completed and committed to the repository.

## Q5 - Web Demo

* **Framework (Gradio/Streamlit):** To be completed.
* **How to launch:** To be completed.
* **Guardrails implemented:** To be completed.

---

## Honest Failure Log

### Q1

* MiniCPM-V 2.6 successfully processed the provided person and garment images.
* A safe pose-classification wrapper was implemented to handle cases where no pose landmarks are detected.
* The side-facing and seated edge cases were classified correctly.
* The no-person edge case was detected correctly.
* Non-fatal warnings from Transformers/bitsandbytes were observed during inference but did not prevent successful processing.

### Q2

* The crossed-arms edge case (`person_crossed_arms.jpg`) was **not successfully handled** in the human parsing pipeline.
* The crossed-arms parsing output was therefore not completed.
* In the other five person images, the **left-hand region was not parsed correctly**.
* This parsing issue did not affect the final try-on outputs in Q3 because the incorrectly parsed hand region did not materially affect the garment-transfer area used by the try-on pipeline.

### Q3

* The end-to-end try-on inference was successfully completed for all five required person-garment pairs.
* The outputs were generated in the correct corresponding order:

  * `person_01 + garment_01`
  * `person_02 + garment_02`
  * `person_03 + garment_03`
  * `person_04 + garment_04`
  * `person_05 + garment_05`
* No pairing or ordering mismatch occurred during inference.
* All five generated try-on outputs were successfully saved.

