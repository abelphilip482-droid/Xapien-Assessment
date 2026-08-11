Virtual Try-On Assessment - Submission

Candidate name: Abel Philip Thomas

Email: abelphilip482@gmail.com

Date: August 11, 2026

GitHub repo link: https://github.com/abelphilip482-droid/Xapien-Assessment

Demo video link (max 5 min): To be added

⸻

Models Used & Licenses

Model	Used in	License
MiniCPM-V 2.6 (int4 quantized)	Q1	Apache 2.0
MediaPipe (pose landmarks)	Q1	Apache 2.0
Grounding DINO	Q2	Apache 2.0
SAM (Segment Anything Model)	Q2	Apache 2.0
U2Net	Q2	Apache 2.0
CatVTON	Q3	Non-commercial research license
Qwen2-VL	Q4 (VLM-as-judge)	Qwen license; non-commercial/research terms apply

⸻

Q1 - Garment & Body Understanding

VLM Chosen and Why

MiniCPM-V 2.6 was selected for Q1.

It is an open-source vision-language model released under the Apache 2.0 license.

The quantized MiniCPM-V-2_6-int4 version was used to make inference practical on the available Google Colab GPU environment.

MiniCPM-V 2.6 was selected because it supports visual understanding of both person and garment images.

It can extract structured attributes such as garment type, sleeve length, neckline, primary color, pattern, body visibility, and other visual information.

A MediaPipe pose-landmark-based classifier is additionally used for pose classification.

This provides a deterministic method for the required pose categories and helps classify the side-facing and seated edge cases correctly.

⸻

How to Run

The Q1 pipeline loads MiniCPM-V 2.6 and processes the provided person and garment images.

The main processing steps are:

1. Load the person image.
2. Detect body landmarks for pose and visibility analysis.
3. Analyze the person image using MiniCPM-V 2.6.
4. Analyze the garment image using MiniCPM-V 2.6.
5. Normalize the garment attributes into the required categories.
6. Combine the person and garment attributes into the required JSON structure.
7. Save the resulting JSON files.

⸻

Known Limitations

Pose classification may be less reliable when the body is heavily occluded.

VLM-generated garment attributes can occasionally be uncertain for visually ambiguous or partially occluded regions.

The quantized model may produce slower inference depending on the available Colab GPU resources.

⸻

Q1 Validation

The dedicated pose edge cases were tested successfully.

Image	Expected	Predicted	Status
person_side_pose.jpg	side	side	PASS
person_seated.jpg	seated	seated	PASS

The no-person edge case was also detected correctly.

⸻

Q2 - Human Parsing & Segmentation

Models Used

Grounding DINO was used for object and region detection.

SAM (Segment Anything Model) was used to convert Grounding DINO bounding boxes into pixel-level segmentation masks.

U2Net was also used for human segmentation and background removal.

How to Run

Run the Q2 notebook/script.

The pipeline loads Grounding DINO, SAM, and U2Net.

It processes the person images.

It generates the required segmentation and agnostic representations.

It processes the garment images for background removal and masking.

Edge Cases

The crossed-arms edge case (person_crossed_arms.jpg) was not successfully handled.

For the other five person images, the left-hand region was not parsed correctly.

This did not materially affect identity preservation.

However, the parsing limitation is related to the sleeve-length and silhouette limitation observed in Q3.

⸻

Q3 - End-to-End Try-On

Try-On Model Chosen and Why

CatVTON was selected as the end-to-end virtual try-on model.

It is an open-source diffusion-based virtual try-on model.

It is relatively lightweight and suitable for running on a free Google Colab T4 GPU.

Hardware Used

Google Colab Tesla T4 GPU — 14.56 GB VRAM was used.

Inference was performed at 384×512 resolution.

The inference process used 30 inference steps.

Constraints and Workarounds

GPU memory was a constraint when running the diffusion-based try-on pipeline.

The inference configuration was kept at 384×512 resolution with 30 inference steps.

Observed Limitation

In some pairs, the generated garment region appears constrained to the silhouette or length of the person’s original clothing.

For example, a longer-sleeve garment can appear shorter than expected.

This is most likely caused by the Q2 agnostic mask being derived tightly from the parsed clothing region.

With additional time and GPU budget, this could be improved using a dedicated pose-guided agnostic-mask generation method.

Output

End-to-end inference was successfully completed for all five required pairs.

The correct person-garment ordering was maintained:

1. person_01 + garment_01
2. person_02 + garment_02
3. person_03 + garment_03
4. person_04 + garment_04
5. person_05 + garment_05

All five corresponding try-on outputs were successfully generated and saved.

⸻

Q4 - Automated Quality Evaluation

Metrics Implemented

Two quantitative metrics were calculated for each generated try-on result.

Garment Fidelity Score

Measures similarity between the reference garment and the generated try-on result.

Identity Preservation Score

Measures similarity between the original person’s face and the face in the generated try-on result using face embeddings and cosine similarity.

The combined quantitative score was calculated as:

0.5 × Garment Fidelity Score + 0.5 × Identity Preservation Score

VLM-as-Judge

Qwen2-VL was used as the VLM-as-judge for qualitative evaluation.

It was used instead of MiniCPM-V 2.6 because it was more readily available in a stable configuration during the remaining Colab session.

For each pair, Qwen2-VL received:

* Original person image
* Reference garment image
* Generated try-on image

The model evaluated:

* Garment color and appearance
* Garment shape and pattern
* Neckline and sleeves
* Visible garment details
* Identity preservation
* Fit and placement
* Visible artifacts

The VLM returned identical numerical values of 0.50 for all five pairs.

Therefore, these repeated numerical values were not used as the final quantitative scores.

The qualitative explanations and artifact observations were retained.

Results

Pair	Garment Fidelity	Identity Preservation	Combined Score
pair_01	0.3398	0.4977	0.4188
pair_02	0.2471	0.5719	0.4095
pair_03	0.3830	0.5781	0.4806
pair_04	0.3216	0.6725	0.4971
pair_05	0.3819	0.4538	0.4179

The completed evaluation results are stored in q4_progress.csv.

⸻

Q5 - Mini Try-On Web Demo

Implementation Status

A basic web-demo prototype was attempted using Gradio.

The purpose was to expose the virtual try-on workflow through an interactive interface.

However, the final Q5 implementation did not fully meet the requirements of the assessment.

The virtual try-on results produced through the demo were not consistently accurate when compared with the outputs generated during the dedicated Q3 inference stage.

The Q3 CatVTON pipeline and its five generated outputs remain the more reliable results.

Limitations

Due to the limited time available for the assessment, the complete Q2 → Q3 → Q4 integration could not be fully completed and validated inside the web application.

The interactive virtual try-on results were not consistently comparable to the dedicated Q3 CatVTON outputs.

The Q4 automated evaluation was not fully integrated into the web application.

The required guardrail behavior for all provided edge cases was not fully validated.

End-to-end testing was also limited by the available development time and Colab GPU constraints.

Intended Architecture

Person Image + Garment Image
              ↓
        Q1 Understanding
              ↓
       Q2 Preprocessing
              ↓
        CatVTON (Q3)
              ↓
       Q4 Evaluation
              ↓
    Result + Quality Scores

Intended Guardrails

The intended application was designed to:

* Reject images where no person is detected.
* Warn when the person is seated.
* Warn when the person is shown from a side view.
* Display an estimated processing time.

These components represent the intended production flow.

However, the complete implementation was not sufficiently validated within the available assessment time.

Honest Assessment

Q5 should therefore be considered a prototype-level and incomplete integration rather than a fully validated mini web application.

The main limitation was not the absence of the underlying Q3 try-on pipeline.

The main limitation was the time required to reliably connect and validate all previous stages inside a single interactive application.

With additional development time and proper technical guidance, the implementation could be improved by:

1. Reusing the exact Q3 CatVTON inference configuration.
2. Connecting the Q2 preprocessing pipeline directly to uploaded inputs.
3. Integrating the Q4 garment-fidelity and identity-preservation metrics.
4. Implementing and validating all required guardrails.
5. Adding robust error handling and input validation.
6. Testing the application against normal inputs and all provided edge cases.
7. Optimizing GPU memory usage and inference time.

The incomplete Q5 integration is documented explicitly rather than presenting unvalidated demo results as equivalent to the dedicated Q3 results.

⸻

Honest Failure Log

Q1

MiniCPM-V 2.6 successfully processed the provided person and garment images.

The side-facing and seated edge cases were classified correctly.

The no-person edge case was detected correctly.

Non-fatal Transformers/bitsandbytes warnings were observed but did not prevent successful processing.

Q2

The crossed-arms edge case was not successfully handled.

The crossed-arms parsing output was therefore not completed.

The left-hand region was also not parsed correctly in the other person images.

This is the most likely root cause of the sleeve-length and silhouette limitation observed in Q3.

Q3

The end-to-end try-on inference was successfully completed for all five required pairs.

All five generated try-on outputs were successfully saved.

No pairing or ordering mismatch occurred.

Some outputs showed limitations in garment silhouette and sleeve-length transfer.

This limitation is attributed to the tight parsing-derived agnostic mask.

Q4

The quantitative garment-fidelity and identity-preservation metrics were calculated for all five pairs.

Qwen2-VL was successfully loaded and used for qualitative evaluation.

The VLM returned identical numerical scores of 0.50 across the evaluated pairs.

These values were therefore not treated as meaningful quantitative measurements.

The final quantitative scores were calculated using the independent garment-fidelity and identity-preservation metrics.

The final evaluation CSV was generated as q4_progress.csv.

Q5

A Gradio-based web demo prototype was attempted.

The virtual try-on results produced through the demo were not sufficiently accurate or consistent compared with the dedicated Q3 CatVTON outputs.

The Q4 evaluation was not fully integrated and validated within the web application.

The required guardrails were not completely validated.

These limitations were primarily due to the limited time available for completing and testing the complete Q2 → Q3 → Q4 → Q5 integration.

GPU and inference constraints also affected the development and testing process.

The Q3 inference results remain the reliable set of generated try-on outputs submitted for the assessment.

With additional development time and appropriate technical guidance, the web demo could be improved by reusing the validated Q3 pipeline, integrating Q4 correctly, and fully implementing and testing the required guardrails.
