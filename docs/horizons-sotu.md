# Horizons App — State of the Union
**Date:** 2026-06-17  
**Branch:** `claude/focused-noether-evoknv`

---

## App Overview

An always-on Android AI assistant with:
- Floating accessibility overlay tile (mic + screenshot shortcuts)
- System-wide custom IME (replaces Google mic anywhere)
- Live screen reader (passive TTS of propagated text)
- Full chat UI with file/image/camera/terminal access
- Model router (OpenRouter, Ollama, OpenAI-compatible endpoints)
- On-device models on NPU + GPU

---

## Model Decisions

| Role | Model | Runtime | Hardware |
|---|---|---|---|
| STT / ASR | `ibm-granite/granite-speech-4.1-2b-nar` | ONNX Runtime QNN EP | Hexagon NPU |
| Chat + Vision | `Mer0vin8ian/gemma-4-E4B-it-qat-mobile-ONNX` | ONNX Runtime QNN EP | Adreno 830 GPU |
| Fallback STT | `whisper_base` (compiled via AI Hub) | QNN context binary | NPU |

### Why these models
- **Granite Speech 4.1:** IBM + Qualcomm + Nexa AI explicitly validated on Hexagon HTP. BYOM path — pull weights, compile via AI Hub pipeline (safetensors → AIMET quant → ONNX → QNN context binary). IBM licensing means no precompiled `.bin` redistribution.
- **Gemma QAT ONNX (mobile):** `Mer0vin8ian/gemma-4-E4B-it-qat-mobile-ONNX` — 3.61GB, QAT quality, split into ONNX components (audio encoder, decoder, vision encoder). Runs via ONNX Runtime QNN EP on Adreno 830. E4B = 17B total params, 4.5B active (MoE architecture).
- **Whisper Base:** Already validated in this repo, 100% NPU on SM8750, encoder 22ms / decoder 2.6ms. Submit compile job at aihub.qualcomm.com.

### Unified runtime: ONNX Runtime with QNN EP
Both Granite and Gemma go through ONNX — Granite as part of the AI Hub compile pipeline, Gemma as pre-exported ONNX components. Single runtime (ONNX Runtime + QNN execution provider) handles both instead of mixing llama.cpp + QNN separately.

### Models to download (browser only — hf CLI crashes on Termux/XET)
- `ibm-granite/granite-speech-4.1-2b-nar` ✅ Already downloaded
- `google/gemma-4-E4B-it-qat-q4_0-gguf` — ~4.5GB, grab from browser
- After download: `mv ~/storage/downloads/gemma* ./gemma-qat/`

---

## Performance Stack

### Game Mode (declare app as game — no special permissions needed)
```xml
<!-- AndroidManifest.xml -->
<application android:appCategory="game">
    <meta-data
        android:name="android.game_mode_config"
        android:resource="@xml/game_mode_config" />
</application>
```
```xml
<!-- res/xml/game_mode_config.xml -->
<game-mode-config
    android:supportsBatteryGameMode="true"
    android:supportsPerformanceGameMode="true" />
```

### ADPF Performance Hints (Kotlin)
```kotlin
val phm = context.getSystemService(PerformanceHintManager::class.java)
val session = phm?.createHintSession(intArrayOf(Process.myTid()), 16_666_666L)
// Per inference cycle:
session?.reportActualWorkDuration(actualNs)
```

### Android 16 (API 36) Headroom APIs
```kotlin
val shm = context.getSystemService(SystemHealthManager::class.java)
val cpuHeadroom = shm?.getCpuHeadroom(CpuHeadroomParams.create())
val gpuHeadroom = shm?.getGpuHeadroom(GpuHeadroomParams.create())
// Back off inference batch size when headroom < 0.3
```

### Thermal Monitoring
```kotlin
powerManager.addThermalStatusListener { status ->
    if (status >= PowerManager.THERMAL_STATUS_SEVERE) reduceBatchSize()
}
```

### NPU Note
No direct NPU ADPF hint — wrap CPU threads that feed the NPU in the hint session. NPU power state managed internally by QNN HAL.

### ADB testing
```bash
adb shell cmd game mode performance <your.package.name>
```

---

## SDK Decisions

| Component | SDK | Notes |
|---|---|---|
| Granite NPU inference | ONNX Runtime QNN EP | ONNX export via AI Hub pipeline, QNN EP targets Hexagon NPU |
| Gemma GPU/NPU inference | ONNX Runtime QNN EP | Pre-exported ONNX components, QNN EP on Adreno 830 |
| Whisper NPU | QNN context binary via AI Hub | Compile at aihub.qualcomm.com |
| Screen capture | MediaProjection API | Standard Android |
| Overlay tile | AccessibilityService | Standard Android |
| System STT/IME | Custom InputMethodService | Standard Android |
| TTS readback | Android TextToSpeech | Standard Android |
| Thermal/perf | ADPF + Game Mode API | Standard Android, no Unreal needed |
| Model compile pipeline | Qualcomm AI Hub Models (this repo) | BYOM path for Granite |

### SDKs ruled out
- **Snapdragon Game AI SDK** — Unreal Engine plugin only, no Android AAR
- **GenIE SDK standalone** — requires NDK/JNI bridging, NexaSDK is easier
- **Qualcomm IM SDK** — Linux IoT only, not Android mobile
- **LiteRT for Gemma** — 9.59GB too large; QAT GGUF is better path

---

## UI/UX Spec

### Overlay Tile (always visible)
- Floating accessibility tile, two buttons: mic + screenshot
- Mic tap → single utterance STT → response via TTS
- Screenshot tap → Gemma vision → TTS response
- No full UI, works over any app

### In-Chat Mic Behavior
- Tap → push-to-talk, sends on release
- Hold → live conversational mode, continuous VAD loop
- Tap during response → interrupt TTS, start listening (barge-in)
- Granite handles VAD + silence detection natively

### System IME
- Custom keyboard input method
- Replaces Google mic anywhere keyboard appears
- Granite STT → text injected into any text field

### Live Screen Reader
- MediaProjection running in background
- Watches for new text not originating from STT
- Auto-reads via TTS when triggered
- Works across all apps (Claude, Gemini, Perplexity, etc.)

### Full Chat UI
- Traditional chat window
- File / screenshot / camera upload
- Terminal access (Termux integration or built-in shell)
- Model router: OpenRouter, Ollama, LM Studio, OpenAI-compatible
- Model manager: download, load, swap, configure
- API key management per provider

---

## Granite 4.0 NPU Path (future)

Granite 4.0 uses Mamba-2 + Transformer hybrid (9:1 ratio). IBM statement:
> "IBM worked with Qualcomm Technologies, Inc. and Nexa AI to ensure Granite 4.0 models' compatibility with Hexagon™ NPUs"

Pipeline (same as Llama/Qwen QNN workflow):
```
HF weights (safetensors)
  → AIMET quantization (w4a16 / w8a16)
  → ONNX export
  → Split: Prompt Processor (AR-128) + Token Generator (AR-1)
  → Compile for SM8750 / v79 HTP
  → Link into QNN context binaries
  → Deploy via Genie T2T or ONNX Runtime QNN EP
```

Status in this repo: `gemma4.py` driver exists, no `ibm_granite_v4_0` model dir yet. Granite 3.1 8B is the current validated recipe.

---

## Reading List / Next Session

### Immediate tasks
1. Submit Whisper Base compile job at aihub.qualcomm.com (web UI, no CLI needed)
2. Download `google/gemma-4-E4B-it-qat-q4_0-gguf` from browser
3. Check Qualcomm forum response re: Granite 4.0 Mamba-2 HTP compiler support

### Docs to read
- `tutorials/llm/onboarding.md` — LLM BYOM pipeline walkthrough
- `tutorials/llm/quantize_llama3.md` — AIMET quantization reference
- `.claude/agents/onboarding.md` — model onboarding workflow
- `src/qai_hub_models/models/ibm_granite_v3_1_8b_instruct/` — reference impl for Granite NPU path
- `src/qai_hub_models/models/_shared/lm_driver/gemma4.py` — Gemma4 driver pattern to follow for Granite4

### External links
- NexaSDK for Android: qualcomm.com/developer/blog/2025/11/nexa-ai-for-android
- ADPF docs: developer.android.com/games/optimize/adpf
- Game Mode API: developer.android.com/games/optimize/adpf/gamemode/gamemode-api
- AI Hub web: aihub.qualcomm.com
- Granite 4.0 HF: ibm-granite/granite-4.0-tiny-preview
- Gemma QAT GGUF: google/gemma-4-E4B-it-qat-q4_0-gguf
- Unsloth QAT alternative: unsloth/gemma-4-E4B-it-qat-GGUF (UD-Q4_K_XL ~4.22GB)

### Open questions
- Does v79 HTP compiler support Mamba-2 ops? (posted in Qualcomm forum)
- Can NexaSDK be added as Android AAR/Maven dep directly?
- Granite 4.0 speech variant availability on HF?
