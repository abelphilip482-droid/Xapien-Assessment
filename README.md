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

CatVTON’s non-commercial research license is acceptable under the assessment rules.

⸻

Q1 - Garment & Body Understanding

VLM Chosen and Why

MiniCPM-V 2.6 was selected for Q1.

It is an open-source vision-language model released under the Apache 2.0 license.

This satisfies the assessment requirement for an open and freely downloadable model.

The quantized MiniCPM-V-2_6-int4 version was used to make inference practical on the available Google Colab GPU environment.

MiniCPM-V 2.6 was selected because it supports visual understanding of both person and garment images.

It can extract structured attributes such as garment type, sleeve length, neckline, primary color, pattern, body visibility, and other visual information.

A MediaPipe pose-landmark-based classifier is additionally used for pose classification.

This provides a deterministic method for the required pose categories.

It also helps ensure that the dedicated side-facing and seated edge cases are classified correctly.

The final Q1 JSON outputs follow the structure provided in sample_output_q1.json.

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

Known Limitations

Pose classification may be less reliable when the body is heavily occluded.

VLM-generated garment attributes can occasionally be uncertain for visually ambiguous or partially occluded regions.

The quantized model may produce slower inference depending on the available Colab GPU resources.

Q1 Validation

The dedicated pose edge cases were tested successfully.

Image	Expected	Predicted	Status
person_side_pose.jpg	side	side	PASS
person_seated.jpg	seated	seated	PASS

The no-person edge case was also detected correctly.

Q1 Outputs

The Q1 output directory contains JSON results for the provided person and garment images.

It also contains the required edge-case person image results.

⸻

Q2 - Human Parsing & Segmentation

Models Used

Grounding DINO was used for object and region detection.

The detected regions included the person, face, hair, arms, upper clothing, and lower body.

Confidence thresholds of 0.25, 0.15, and 0.10 were used for the detections.

SAM (Segment Anything Model) was used to convert the Grounding DINO bounding boxes into pixel-level segmentation masks.

SAM was also used to generate the required person and semantic masks.

U2Net was additionally used for human segmentation and background removal during the Q2 preprocessing pipeline.

How to Run

Run the Q2 notebook/script.

The pipeline loads Grounding DINO, SAM, and U2Net.

It processes the person images.

It generates the required segmentation and agnostic representations.

It also processes the garment images for background removal and masking.

Edge Cases Handled / Failed

The crossed-arms edge case (person_crossed_arms.jpg) was not successfully handled in the human parsing pipeline.

For the other five person images, the left-hand region was not parsed correctly.

This parsing issue did not affect the identity-relevant garment-transfer area directly.

However, it is related to a downstream limitation observed in Q3.

The agnostic mask used for try-on inference was derived from this parsing output.

Its tightness around the arm and sleeve region is the most likely cause of the sleeve-length constraint observed in some Q3 outputs.

The remaining person and garment segmentation outputs were generated successfully.

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

Constraints Hit and Workarounds

GPU memory was a constraint when running the diffusion-based try-on pipeline.

The inference configuration was kept at 384×512 resolution with 30 inference steps to make the pipeline practical within the available Tesla T4 GPU memory.

Observed Limitation

In some pairs, the generated garment region appears constrained to the silhouette or length of the person’s original clothing.

As a result, the generated garment does not always fully adopt the reference garment’s actual shape.

For example, a longer-sleeve garment can appear shorter than expected in the generated result.

This is most likely caused by the Q2 agnostic mask being derived tightly from the parsed clothing region.

A more generous, pose-guided garment-agnostic area would provide the model with more freedom to render garments with different shapes.

Standard try-on pipelines such as IDM-VTON and OOTDiffusion typically use a dedicated agnostic-generation step driven by pose or DensePose keypoints.

Such approaches mask a larger region than only the existing clothing.

This allows the model to render garments with shapes that differ from the person’s original clothing.

With more time and GPU budget, this limitation could be addressed by either:

1. Using a dedicated pose-guided agnostic-mask generation script instead of a parsing-derived mask.
2. Applying morphological dilation to the existing mask around the arm and torso region before inference.

How to Run

Run the Q3 inference notebook/script.

The pipeline takes the person image, garment image, and clothing mask produced during preprocessing.

These inputs are passed to CatVTON.

The generated try-on image is then saved to the output directory.

Output

End-to-end inference was successfully completed for all five required pairs.

The correct person-garment ordering was maintained:

1. person_01 + garment_01
2. person_02 + garment_02
3. person_03 + garment_03
4. person_04 + garment_04
5. person_05 + garment_05

All five corresponding try-on outputs were successfully generated and saved.

The outputs are subject to the sleeve-length and silhouette limitation described above.

⸻

Q4 - Automated Quality Evaluation

Metrics Implemented

Two quantitative metrics were calculated for each generated try-on result.

1. Garment Fidelity Score

This measures similarity between the reference garment and the generated try-on result.

2. Identity Preservation Score

This measures similarity between the original person’s face and the face in the generated try-on result using face embeddings and cosine similarity.

A combined quantitative score was calculated as:

0.5 × Garment Fidelity Score + 0.5 × Identity Preservation Score

VLM-as-Judge

Qwen2-VL was used as the VLM-as-judge for qualitative evaluation of the five generated try-on outputs.

This was used in place of MiniCPM-V 2.6, which was used in Q1.

This deviation from reusing the Q1 model is noted here for transparency.

Qwen2-VL was tried as the judge model because it was more readily available in a stable, non-quantized configuration for this evaluation step within the remaining Colab session.

The quantized MiniCPM-V-2_6-int4 setup used in Q1 was reserved for the attribute-extraction task it had already been validated against.

Both models are open-source and license-compatible with this assessment.

For each pair, Qwen2-VL was provided with:

* Original person image.
* Reference garment image.
* Generated try-on image.

The evaluation prompt instructed the model to assess:

* Garment color and appearance.
* Garment shape and pattern.
* Neckline and sleeves.
* Visible garment details.
* Identity preservation.
* Fit and placement.
* Visible artifacts.

The VLM was also instructed to provide qualitative reasons and identify visible artifacts.

During testing, the VLM returned identical numerical values of 0.50 for all five pairs despite providing different qualitative explanations.

Therefore, these repeated VLM numerical values were not used as the final quantitative scores.

The VLM output was retained as a qualitative judge for reasons and artifact observations.

The quantitative evaluation uses the independently calculated garment-fidelity and identity-preservation metrics.

Results

Pair	Garment Fidelity	Identity Preservation	Combined Score
pair_01	0.3398	0.4977	0.4188
pair_02	0.2471	0.5719	0.4095
pair_03	0.3830	0.5781	0.4806
pair_04	0.3216	0.6725	0.4971
pair_05	0.3819	0.4538	0.4179

The completed evaluation results are stored in:

q4_progress.csv

This file serves as the completed version of evaluation_template_q4.csv required by the submission instructions.

It has been committed to the repository under this filename.

It contains the required evaluation columns.

Q4 Output

The Q4 pipeline produces an evaluation CSV containing:

* pair_id
* person_image
* garment_image
* tryon_model
* garment_fidelity_score
* identity_preservation_score
* vlm_judge_score
* vlm_judge_reasons
* artifacts_observed
* notes

⸻

Q5 - Mini Try-On Web Demo

Implementation Status

A basic web-demo prototype was attempted using Gradio to expose the virtual try-on workflow through an interactive interface.

However, the final Q5 implementation did not fully meet the requirements of the assessment.

The virtual try-on results produced through the demo were not consistently accurate when compared with the outputs generated during the dedicated Q3 inference stage.

The Q3 CatVTON pipeline and its five generated outputs remain the more reliable results of the implementation.

Limitations

Due to the limited time available for the assessment, the complete Q2 → Q3 → Q4 integration required for a reliable interactive demo could not be completed and validated properly.

The interactive virtual try-on results were not consistently comparable to the dedicated Q3 CatVTON outputs.

The Q4 automated evaluation was not fully integrated into the web application.

The required guardrail behavior for all provided edge cases was not validated within the final application.

End-to-end testing of the complete interactive pipeline was limited by the available development time and Colab GPU constraints.

Intended Design

The intended Q5 application architecture was:

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

The intended guardrails were:

* Reject inputs where no person is detected.
* Warn when the person is seated.
* Warn when the person is shown from a side view.
* Display an estimated processing time.

These components represent the intended production flow.

However, the complete implementation was not sufficiently validated within the available assessment time.

Honest Assessment

Q5 should therefore be considered a prototype-level and incomplete integration rather than a fully validated mini web application.

The main limitation was not the absence of the underlying Q3 try-on pipeline.

The main limitation was the time required to reliably connect and validate all previous stages inside a single interactive application.

With additional development time and proper technical guidance, the implementation could be significantly improved by:

1. Reusing the exact Q3 CatVTON inference configuration inside the web application.
2. Connecting the Q2 preprocessing pipeline directly to the uploaded inputs.
3. Integrating the independently calculated Q4 garment-fidelity and identity-preservation metrics.
4. Implementing and validating all three required guardrails.
5. Adding robust error handling and input validation.
6. Testing the complete application against normal inputs and all provided edge cases.
7. Optimizing GPU memory usage and inference time for interactive use.

The incomplete Q5 integration is documented explicitly rather than presenting unvalidated demo results as equivalent to the dedicated Q3 results.

⸻

Honest Failure Log

Q1

MiniCPM-V 2.6 successfully processed the provided person and garment images.

A safe pose-classification wrapper was implemented to handle cases where no pose landmarks are detected.

The side-facing and seated edge cases were classified correctly.

The no-person edge case was detected correctly.

Non-fatal warnings from Transformers/bitsandbytes were observed during inference but did not prevent successful processing.

Q2

The crossed-arms edge case (person_crossed_arms.jpg) was not successfully handled in the human parsing pipeline.

The crossed-arms parsing output was therefore not completed.

In the other five person images, the left-hand region was not parsed correctly.

This did not materially affect identity preservation.

However, it is the most likely root cause of the sleeve-length and silhouette limitation observed in Q3.

The Q3 agnostic mask was derived from this parsing output.

Q3

The end-to-end try-on inference was successfully completed for all five required person-garment pairs.

The outputs were generated in the correct corresponding order:

* person_01 + garment_01
* person_02 + garment_02
* person_03 + garment_03
* person_04 + garment_04
* person_05 + garment_05

No pairing or ordering mismatch occurred during inference.

All five generated try-on outputs were successfully saved.

In some pairs, the generated garment region was constrained to the approximate silhouette or length of the person’s original clothing rather than the full shape of the reference garment.

For example, sleeve length was not always transferred accurately.

This is attributed to the Q2 agnostic mask being derived tightly from the parsing output rather than a more generous, pose-guided garment-agnostic region.

This is documented as a known limitation rather than presenting the output as fully correct.

Q4

The quantitative garment-fidelity and identity-preservation metrics were successfully calculated for all five pairs.

Qwen2-VL was successfully loaded and used for qualitative VLM-based evaluation.

Qwen2-VL was used in place of MiniCPM-V 2.6 for the reasons explained in the Q4 section.

The VLM returned identical numerical scores of 0.50 across the evaluated pairs despite producing different textual assessments.

To avoid treating these non-discriminative VLM values as meaningful quantitative measurements, the final Q4 quantitative score was calculated from the independently measured garment-fidelity and identity-preservation scores.

Qwen2-VL qualitative explanations and artifact observations were retained in the Q4 evaluation output.

The final Q4 evaluation CSV was successfully generated as q4_progress.csv.

Q5

A Gradio-based web demo prototype was attempted.

The virtual try-on results produced through the demo were not sufficiently accurate or consistent compared with the dedicated Q3 CatVTON outputs.

The Q4 evaluation was not fully integrated and validated within the web application.

The required guardrails were not completely validated in the final interactive application.

These limitations were primarily due to the limited time available for completing and testing the complete Q2 → Q3 → Q4 → Q5 integration.

GPU and inference constraints also affected the development and testing process.

The Q3 inference results remain the reliable set of generated try-on outputs submitted for the assessment.

With additional development time and appropriate technical guidance, the web demo could be improved by reusing the validated Q3 inference pipeline.

The Q4 evaluation could then be integrated correctly.

The required guardrails could also be fully implemented and tested against the provided edge cases.

Overall, the limitations in Q5 are documented transparently rather than presenting an insufficiently validated interactive demo as a fully working implementation.
