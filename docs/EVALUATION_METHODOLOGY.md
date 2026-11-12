# Evaluation Methodology — ITMS-VirtualTrack (Milestone 1 Baseline)

> **SYNTHETIC BENCHMARK NOTICE**: Evaluation results measure model behavior on **SYNTHETIC** Blender-rendered railway runs.
> **Status**: Evaluation protocol specified in Milestone 1; evaluation code will be implemented in Milestone 7.

## 1. Matching & Deduplication

- **Frame-level matching**: Predictions are matched to frame ground truth using IoU (`>= 0.50` primary threshold, plus `0.50:0.95` sweep for mAP50-95).
- **Anomaly-level deduplication**: Consecutive detections of the same anomaly ID (`A1`–`A4`) are grouped by identity rather than counted as separate discoveries.

## 2. Planned Metrics (Per Speed Condition)

- Total ground-truth anomaly instances, correctly detected anomalies, missed anomalies, false positives, false-positive rate.
- Precision, Recall, F1-score, mAP50, mAP50-95.
- Mean inference latency (ms/frame).
- **Earliest reliable detection distance (m)**: Camera-to-anomaly longitudinal distance at the first valid detection of an anomaly identity.
- **Detection latency (s)**: Elapsed time from first ground-truth visibility (`visible == true` and above minimum pixel threshold) to first valid detection.

## 3. Planned Comparative Plots

1. Train speed vs. recall
2. Train speed vs. precision
3. Train speed vs. F1-score
4. Train speed vs. detection distance
5. Train speed vs. inference latency
6. Anomaly category vs. detection performance

