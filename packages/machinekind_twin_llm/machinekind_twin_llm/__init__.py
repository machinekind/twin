"""The digital twin's LLM detector: camera frames to the locally installed claude or codex CLI.

Each question runs the CLI headless as a subprocess, on the subscription it is logged in with. The
package never talks to a model API itself and holds no API key.

* `cli`: running the CLIs, the login check, one question per CLI;
* `images`: the JPEGs a question sends and whether the scene moved since the last one;
* `prompt`: the answer schema and the prompt;
* `answer`: the model's answer checked and scaled back to Detections in frame pixels;
* `smoke`: one real question per CLI, by hand (`python -m machinekind_twin_llm.smoke`).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
