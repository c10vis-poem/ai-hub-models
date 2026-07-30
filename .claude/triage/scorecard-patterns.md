# Scorecard-Specific Patterns

Patterns specific to scorecard performance and numerics regressions.
Use alongside `error-patterns.md` and `runtime-guide.md` for triage.

---

## Performance Regression Signatures

### Device Firmware / Driver Updates

| Signal | Confidence | Team |
|--------|-----------|------|
| Multiple unrelated models regress on same device, same run | HIGH | Tungsten |
| Regression appears only on one chipset generation | MEDIUM | Tungsten |
| All runtimes affected on same device | HIGH | Tungsten (firmware) |
| Only QNN runtime affected on a device, TFLite fine | MEDIUM | Compiler/ONNX2EP |

### Runtime Version Bumps

| Signal | Confidence | Team |
|--------|-----------|------|
| All ONNX Runtime models regress, QNN fine | HIGH | Tungsten (ORT) |
| All TFLite models regress | HIGH | Compiler/ONNX2EP (delegate) |
| Only context binary path affected | HIGH | Compiler/ONNX2EP |
| QNN models regress across all devices | HIGH | Tungsten (QNN runtime update) |

### Noise vs Real (Threshold Guidance)

| Factor | Duration | Classification |
|--------|----------|---------------|
| 2-3x | Single run, then recovers | FLAKY — likely device load/thermal |
| 2-3x | 3+ consecutive runs | SUSTAINED — real regression |
| >5x | Any | CRITICAL — always investigate |
| 2x | Absolute time diff < 1ms | NOISE — within measurement error |
| 2-3x | Only on automotive devices (SA8775P) | Check if device was under load — automotive benchmarks are noisier |

### Infrastructure / Cloud Issues

| Signal | Confidence | Team |
|--------|-----------|------|
| Regression coincides with known Hub maintenance window | HIGH | Cloud Services |
| Same model shows wildly different times across retries | MEDIUM | Cloud Services (device pool contention) |
| All models on one device show ~same factor increase | HIGH | Cloud Services (throttled device) |
| All entries on one device report `-inf` / no valid measurement + null Job IDs for subset | HIGH | Cloud Services / Scorecard infra (device pool unavailable or pipeline skipped submissions) |
| Samsung Galaxy S25 `-inf` across all runtimes, sustained across multiple consecutive runs (2026-06-05: 203, 2026-06-08: 57, 2026-06-12: 15 entries) | HIGH | Cloud Services / Tungsten — S25 device pool or firmware. Improving trend but not fully resolved. (low confidence — verify with next scorecard run) |

---

## Numerics Regression Signatures

### QAIRT Version Bump — w8a16 Catastrophic Accuracy Drops

| Signal | Confidence | Team |
|--------|-----------|------|
| Multiple w8a16 models on same device drop 30–50%+ device accuracy, FP accuracy stable | HIGH | Compiler/ONNX2EP (QAIRT regression) |
| Catastrophic drop (>30% accuracy loss) concentrated on one device + one precision | HIGH | Compiler/ONNX2EP — likely QAIRT quantizer path regression. External: AISW JIRA. |
| Drop disappears when switching to quantizer_v2 (PSNR recovers) | HIGH | Compiler/ONNX2EP — QAIRT quantizer v1 path regression confirmed. See tetracode#20197 (Jul 2026, S25 w8a16 cluster). |

### Quantization Drift

| Signal | Confidence | Team |
|--------|-----------|------|
| w8a8/w8a16 precision only, float16 fine | HIGH | Quantization (AIMET) |
| PSNR/mAP drops across many models same run | HIGH | Quantization (calibration data or AIMET update) |
| Single model, single metric | MEDIUM | AI Hub Models (model code change) |
| `QcQuantizeOp_` in any related error | HIGH | Quantization |

### Reference Model Updates

| Signal | Confidence | Team |
|--------|-----------|------|
| FP Accuracy changed vs previous (compare "Previous FP Accuracy") | HIGH | AI Hub Models (torch model weights updated) |
| Device accuracy changed but FP accuracy stable | HIGH | Compiler/Tungsten (runtime or compile change) |
| New metric appears with no previous data | LOW | Not a regression — new coverage added |

### Device Accuracy Issues

| Signal | Confidence | Team |
|--------|-----------|------|
| Same model fails on all devices | MEDIUM | AI Hub Models or Quantization |
| Same model fails on one device only | HIGH | Tungsten (device-specific runtime bug) |
| Accuracy within 1% of threshold | LOW | Borderline — may be measurement variance |

---

## Deployment-Specific Patterns

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Regression in prod but NOT in dev | Prod-specific issue or already fixed in dev | Check dev scorecard results |
| Regression in dev but NOT in prod | Compiler/runtime team testing unreleased changes | Flag but don't escalate |
| Regression in BOTH prod and dev | Systemic issue (shared infrastructure or model change) | Escalate — affects all environments |
| New regression only in staging | Staging environment instability | Low priority unless sustained |

---

## Known Flaky Model/Device Combos

<!-- This section is populated over time by the kb-weekly-update agent.
     Add entries as patterns are confirmed via multiple scorecard runs. -->

| Model | Device | Runtime | Notes |
|-------|--------|---------|-------|
| `amt_torchscript` | All | `tflite`, `qnn_dlc` | Chronically flaky inference (5P/17F tflite, 12P/10F qnn_dlc over 22 nightlies). Replaced by `cdcn_torchscript` in integration test set (PR #3462, Jun 2026). |
| `bevfusion_det` (decoder) | Samsung Galaxy S25 | `qnn_context_binary`, `precompiled_qnn_onnx` | 15x regression on S25 only (QAIRT 2.47 suspected). Tracked in tetracode#19932, #19807. Sustained across Jun 12 + Jun 18 dev scorecards. |
| `esrgan` | Samsung Galaxy S25 | `qnn_dlc` | Reports `-inf` timing on S25. Part of broader S25/qnn_dlc cluster. Tracked in tetracode#19051, #19917. Sustained across Jun 12 + Jun 18. |
| `vit` | Samsung Galaxy S25 | `qnn_dlc` | Reports `-inf` timing (w8a8). QAIRT 2.47 regression suspected. Tracked in tetracode#19775. Sustained Jun 12 through Jul 23 (confirmed in #20451, #20535). |
| `maskrcnn` (roi_head) | Samsung Galaxy S25 | `qnn_dlc` | 2x+ slowdown or `-inf` on S25. Appeared in Jun 25 dev + Jun 29 prod + Jul 2 dev scorecards. Tracked jointly with S25/QAIRT cluster. |
| `yolor` | Samsung Galaxy S25 | `qnn_dlc`, `onnx` | `-inf` or 12x slowdown on S25. Appeared Jun 25 dev + Jun 29 prod scorecards. Part of broader S25/QAIRT 2.47 cluster. |
| `detr_resnet101` | Samsung Galaxy S25 | `onnx` | `-inf` timing (w8a16_mixed_int16). Appeared Jul 2 dev + Jul 9 dev + Jul 16 dev + Jul 23 dev scorecards. Layer norm issue under investigation (QAIRT 2.48 vs 2.45). Tracked jointly with S25 cluster. |
| `detr_resnet50_dc5` | Samsung Galaxy S25 | `onnx` | `-inf` timing (w8a16_mixed_int16). Same root cause as `detr_resnet101`. Appeared Jul 2 dev + Jul 9 dev + Jul 16 dev + Jul 23 dev scorecards. |
| `convnext_tiny` | Samsung Galaxy S25 | `qnn_dlc` | `-inf` timing (w8a8). Appeared Jul 16 dev (#20451) + Jul 23 dev (#20535) scorecards. Part of broader S25/QAIRT cluster. |
| `detr_resnet50` | Samsung Galaxy S25 | `qnn_dlc` | `-inf` timing (w8a16_mixed_fp16). Appeared Jul 16 dev (#20451) + Jul 23 dev (#20535) scorecards. Part of broader S25/QAIRT cluster. |

---

## Cross-Referencing with Trend Data

When the trend report (`trend-report.json`) is available:

- **NEW regressions** → investigate first (highest priority)
- **SUSTAINED regressions** → known issues, link to existing tracking tickets if available
- **FLAKY regressions** → likely noise, mention but don't escalate
- **RECOVERED regressions** → good news, mention briefly for awareness
