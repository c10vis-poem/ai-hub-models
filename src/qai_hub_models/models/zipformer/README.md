# [Zipformer: Transformer-based automatic speech recognition (ASR) model for English and Chinese language](https://aihub.qualcomm.com/models/zipformer)

Zipformer streaming ASR (Automatic Speech Recognition) model is a state-of-the-art system designed for transcribing spoken language into written text streamingly. This model is based on the transformer architecture and has been optimized for edge inference by replacing linear layers with convolutional (conv) layers. It exhibits robust performance in realistic, noisy environments, making it highly reliable for real-world applications. Specifically, it excels in long-form transcription, capable of accurately transcribing audios. Time to the first token is the encoder's latency, while time to each additional token is joiner's latency, where we assume a max decoded length specified below.

This is based on the implementation of Zipformer found [here](https://github.com/k2-fsa/icefall).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/zipformer).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Zipformer:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Zipformer

# Print performance and accuracy metrics
qai-hub-models perf Zipformer
qai-hub-models numerics Zipformer

# Download a ready-to-deploy asset
qai-hub-models fetch Zipformer --runtime qnn_context_binary --precision float
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install torch==2.9.0+cpu -f https://download.pytorch.org/whl/torch/
pip install "qai-hub-models[zipformer]"
pip install k2==1.24.4.dev20251029+cpu.torch2.9.0 -f https://k2-fsa.github.io/k2/cpu.html
```

### 2. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.zipformer.demo { --quantize mixed }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct.
Use the following command to export the model:
```bash
qai-hub-models export zipformer --target-runtime qnn_context_binary --precision float --device "Samsung Galaxy S25 (Family)"
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Zipformer can be found
  [here](https://github.com/huggingface/transformers/blob/v4.42.3/LICENSE).

## References
* [Zipformer A faster and better encoder for automatic speech recognition](https://arxiv.org/abs/2310.11230)
* [Source Model Implementation](https://github.com/k2-fsa/icefall)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
