"""The digital twin's seeded detector: the LLM detector names the blocks, colour follows them.

Every `seed_interval_s` the LLM detector (`machinekind_twin_llm`) is asked, on a thread of its own,
where each block is. Between its answers the detector follows each block by colour at the twin's
`interval_s` and hands the twin the block's silhouette mask and size, so the twin's box fit places
it with its yaw at frame rate.

* `detector`: `SeededDetector`, the detector the twin finds under the entry point name `seeded`;
* `settings`: `SeededSettings`, what the operator sets, the label tables' parsers, the checks and
  the settings handed to the LLM detector;
* `grow`: the colour mask, its blobs and the blob nearest a pixel; pure functions.
"""

from .detector import SeededDetector
from .settings import SeededSettings

__version__ = "0.1.0"

__all__ = ["SeededDetector", "SeededSettings", "__version__"]
