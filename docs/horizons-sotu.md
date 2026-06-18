# Horizons App — State of the Union
**Date:** 2026-06-18  
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
- Terminal access to load/hot-swap models from device storage or API
- Dual model slots: daily driver + coding/math specialist, swap on demand

---

## Model Decisions

| Role | Model | Runtime | Hardware |
|---|---|---|---|
| STT / ASR | `whisper_base` (compiled via AI Hub) | QNN context binary | Hexagon NPU |
| Chat + Vision (daily) | `google/gemma-4-12b-it-qat-q4_0.gguf` | llama.cpp / QNN backend | Adreno 830 GPU + NPU auto-distribute |
| Chat (coding/math) | `unsloth/gemma-4-12b-it-qat-GGUF` (hot-swap) | llama.cpp / QNN backend | Adreno 830 GPU + NPU auto-distribute |
| TTS | VoxSherpa | Android TextToSpeech API | CPU |

### Why these models
- **Whisper Base:** Already validated in this repo, 100% NPU on SM8750, encoder 22ms / decoder 2.6ms. Handles technical vocabulary (Qualcomm, Termux, etc.) correctly — Android's built-in SpeechRecognizer does not. Submit compile job at aihub.qualcomm.com.
- **Gemma 4 12B QAT Q4_0 (Google):** Downloaded directly from Google. Best overall logic, vibe coding, and general capabilities. ~7GB. Benchmarks crush any other model at this size.
- **Gemma 4 12B QAT (Unsloth) — hot-swap:** Scores higher on code logic and math. Keep stored on device, load when running a heavy coding session from terminal. Not always-loaded.
- **Qualcomm auto-distribution:** Running via llama.cpp + QNN backend, the runtime auto-distributes layers across NPU/GPU/CPU based on load. No manual layer splitting needed.

### Ruled out
- **Moonshot** — no validated on-device NPU path
- **Granite Speech 4.1 as primary STT** — Whisper Base already compiled for NPU, simpler path
- **Android native SpeechRecognizer** — Google's engine, poor technical vocabulary, bad punctuation
- **E4B MoE ONNX** — 12B dense benchmarks are dramatically better; size tradeoff worth it

### Models to download (browser only — hf CLI crashes on Termux/XET)
- `whisper_base` — compile at aihub.qualcomm.com (no download needed, AI Hub handles it)
- `google/gemma-4-12b-it-qat-q4_0.gguf` ✅ Downloaded
- `unsloth/gemma-4-12b-it-qat-GGUF` — download when needed for coding sessions

---

## Two API Stacks — These Are Separate

### 1. Performance Stack (ADPF + Game Mode)
Controls CPU/GPU/NPU scheduler behavior. Has nothing to do with audio.

**Game Mode** (declare app as game — no special permissions needed):
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

**ADPF Performance Hints (Kotlin):**
```kotlin
val phm = context.getSystemService(PerformanceHintManager::class.java)
val session = phm?.createHintSession(intArrayOf(Process.myTid()), 16_666_666L)
session?.reportActualWorkDuration(actualNs)
```

**Android 16 (API 36) Headroom APIs:**
```kotlin
val shm = context.getSystemService(SystemHealthManager::class.java)
val cpuHeadroom = shm?.getCpuHeadroom(CpuHeadroomParams.create())
val gpuHeadroom = shm?.getGpuHeadroom(GpuHeadroomParams.create())
// Back off inference batch size when headroom < 0.3
```

**Thermal Monitoring:**
```kotlin
powerManager.addThermalStatusListener { status ->
    if (status >= PowerManager.THERMAL_STATUS_SEVERE) reduceBatchSize()
}
```

**NPU Note:** No direct NPU ADPF hint — wrap CPU threads that feed the NPU in the hint session. NPU power state managed internally by QNN HAL.

**ADB testing:**
```bash
adb shell cmd game mode performance <your.package.name>
```

### 2. Audio Stack (STT / TTS)
Completely separate from the performance stack. Whisper handles STT via QNN context binary. VoxSherpa handles TTS.

**VoxSherpa STT exploration:** VoxSherpa has an internal STT layer it does not expose externally — only accessible via its built-in shell script recorder. Explore two paths:
- **Tasker + AccessibilityService:** Intercept VoxSherpa UI events, pipe audio through its recorder programmatically
- **Shell script bridge:** If VoxSherpa's script feature accepts stdin or file input, pipe audio to it and capture transcript output
- Goal: expose VoxSherpa's STT as a system-level resource without modifying the app

---

## SDK Decisions

| Component | SDK | Notes |
|---|---|---|
| Gemma inference | llama.cpp + QNN backend | Auto-distributes to NPU/GPU/CPU, `-ngl 99` |
| Whisper NPU | QNN context binary via AI Hub | Compile at aihub.qualcomm.com |
| Screen capture | MediaProjection API | Standard Android |
| Overlay tile | AccessibilityService | Standard Android |
| System STT/IME | Custom InputMethodService + Whisper | Whisper replaces Google mic system-wide |
| TTS readback | VoxSherpa via Android TextToSpeech | Standard Android |
| Thermal/perf | ADPF + Game Mode API | Standard Android, no Unreal needed |
| Model compile pipeline | Qualcomm AI Hub Models (this repo) | BYOM path |
| Model hot-swap | Terminal access (Termux / built-in shell) | Load from device storage or API on demand |

### SDKs ruled out
- **Snapdragon Game AI SDK** — Unreal Engine plugin only, no Android AAR
- **GenIE SDK standalone** — requires NDK/JNI bridging, unnecessary complexity
- **Qualcomm IM SDK** — Linux IoT only, not Android mobile
- **NexaSDK** — superseded by direct llama.cpp + QNN backend path
- **ONNX Runtime QNN EP for Gemma** — llama.cpp handles GGUF natively, simpler

---

## UI/UX Spec

### Overlay Tile (always visible)
- Floating accessibility tile, two buttons: mic + screenshot
- Mic tap → single utterance Whisper STT → Gemma → TTS response
- Screenshot tap → Gemma vision → TTS response
- No full UI, works over any app

### In-Chat Mic Behavior
- Tap → push-to-talk, sends on release
- Hold → live conversational mode, continuous VAD loop
- Tap during response → interrupt TTS, start listening (barge-in)

### System IME
- Custom keyboard input method
- Replaces Google mic anywhere keyboard appears
- Whisper STT → text injected into any text field

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
- Model manager: download from HF, load from device storage, hot-swap between slots
- API key management per provider
- Dual model slots: daily driver (Gemma QAT Google) + coding slot (Unsloth hot-swap)

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
1. Submit Whisper Base compile job at aihub.qualcomm.com (web UI → Models → Whisper Base → Run on device → NPU → SM8750)
2. Download Unsloth Gemma 12B when needed for a coding session
3. Check Qualcomm forum response re: Granite 4.0 Mamba-2 HTP compiler support
4. Explore VoxSherpa STT bridge via Tasker or shell script

### Docs to read
- `tutorials/llm/onboarding.md` — LLM BYOM pipeline walkthrough
- `tutorials/llm/quantize_llama3.md` — AIMET quantization reference
- `.claude/agents/onboarding.md` — model onboarding workflow
- `src/qai_hub_models/models/ibm_granite_v3_1_8b_instruct/` — reference impl for Granite NPU path
- `src/qai_hub_models/models/_shared/lm_driver/gemma4.py` — Gemma4 driver pattern to follow for Granite4

### External links
- ADPF docs: developer.android.com/games/optimize/adpf
- Game Mode API: developer.android.com/games/optimize/adpf/gamemode/gamemode-api
- AI Hub web: aihub.qualcomm.com
- Granite 4.0 HF: ibm-granite/granite-4.0-tiny-preview
- Gemma 12B QAT (Google): google/gemma-4-12b-it-qat-q4_0-gguf
- Gemma 12B QAT (Unsloth): unsloth/gemma-4-12b-it-qat-GGUF

### Open questions
- Does v79 HTP compiler support Mamba-2 ops? (posted in Qualcomm forum)
- VoxSherpa STT: does shell script recorder accept file/stdin input for programmatic use?
- Granite 4.0 speech variant availability on HF?
