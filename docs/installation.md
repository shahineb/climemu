# Installation

Python ≥ 3.11 is required. GPU support is recommended for practical usage.

## From PyPI

| Platform   | Command                              |
|------------|--------------------------------------|
| CPU        | `pip install earthsampler`           |
| NVIDIA GPU | `pip install earthsampler[cuda12]`   |
| Google TPU | `pip install earthsampler[tpu]`      |

## From source

```bash
git clone https://github.com/shahineb/mit-earthsampler.git
cd mit-earthsampler
pip install -e .
```
