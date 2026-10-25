"""Detectors from installed packages: found through entry points, built strictly from their settings.

Most detectors here are inline fakes that `install_detector` installs the way pip would. Their
dist names start apart from the strategy tests' ones: a module once imported stays in
`sys.modules` for the whole run. The example cube detector (`examples/detectors/cube`) is
installed by the `cube_example` fixture, the LLM detector (`packages/machinekind_twin_llm`) by
the `llm_package` fixture.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest
from conftest import BOT, CUBE, install_detector
from machinekind_twin_api.entry_points import DETECTORS

from machinekind_twin.detectors import (
    ENTRY_POINT_GROUP,
    DetectorPlugin,
    config_section,
    detectors,
    load_detector,
    shared_section,
)

GROUP = "machinekind_twin.detectors"

#: A detector on the contract's base class, with settings.
WITH_SETTINGS = '''
from dataclasses import dataclass

from machinekind_twin_api import Detector as Base


@dataclass
class Settings:
    color: str
    size_m: float = 0.03
    interval_s: float = 0.33


class Detector(Base[Settings]):
    """Sees one coloured block."""

    settings = Settings

    def __init__(self, settings):
        self.config = settings
        self.interval_s = settings.interval_s

    def detect(self, request):
        return []
'''

#: No base class: the twin checks the attributes only.
BASE_FREE = '''
class Detector:
    """Sees nothing, without the contract's base class."""

    settings = None
    interval_s = 2.0
    pixel_error_px = 3.0

    def start(self, info):
        pass

    def stop(self, reason):
        pass

    def detect(self, request):
        return []
'''

NO_DETECT = '''
class Detector:
    interval_s = 1.0
'''

A_FUNCTION = '''
def Detector(settings=None):
    return None
'''

NOT_A_DATACLASS = '''
class Detector:
    settings = {"color": "red"}

    def detect(self, request):
        return []
'''

LIST_SETTING = '''
from dataclasses import dataclass, field

@dataclass
class Settings:
    colors: list = field(default_factory=list)

class Detector:
    settings = Settings

    def __init__(self, settings):
        self.config = settings

    def detect(self, request):
        return []
'''


# --------------------------------------------------------------------------- discovery


def test_the_twin_reads_the_entry_point_group_machinekind_twin_api_names():
    assert ENTRY_POINT_GROUP == DETECTORS == GROUP


def test_a_detector_from_an_installed_package_is_found_through_its_entry_point(install_dist):
    name = install_detector(install_dist, "blocks-found", WITH_SETTINGS)

    assert detectors()[name].value == "blocks_found:Detector"

    plugin = load_detector(name)

    assert isinstance(plugin, DetectorPlugin)
    assert plugin.name == "blocks_found"
    assert plugin.cls.__module__ == "blocks_found"
    assert plugin.source == "blocks-found 0.1.0 (blocks_found:Detector)"
    assert plugin.summary == "Sees one coloured block."


def test_names_are_compared_in_lower_case(install_dist):
    install_dist("loud-blocks", {GROUP: {"LoudBlocks": "loud_blocks:Detector"}},
                 {"loud_blocks": WITH_SETTINGS})

    assert "loudblocks" in detectors()
    assert load_detector("LOUDBLOCKS").name == "loudblocks"


def test_two_detectors_with_one_name_are_refused(install_dist):
    install_dist("blocks-one", {GROUP: {"blocks": "blocks_one:Detector"}})
    install_dist("blocks-two", {GROUP: {"blocks": "blocks_two:Detector"}})

    with pytest.raises(ValueError, match="Dwa detektory o nazwie 'blocks'"):
        detectors()


def test_an_unknown_detector_names_the_installed_ones(install_dist):
    install_detector(install_dist, "blocks-known", WITH_SETTINGS)

    with pytest.raises(KeyError, match="Nieznany detektor 'yolo'. Zainstalowane: .*blocks_known"):
        load_detector("yolo")


def test_a_detector_without_the_base_class_loads_by_its_attributes(install_dist):
    name = install_detector(install_dist, "base-free", BASE_FREE)
    plugin = load_detector(name)

    assert plugin.settings is None
    assert plugin.fields == ()

    detector = plugin.create({})

    assert detector.interval_s == 2.0
    assert detector.detect(None) == []


@pytest.mark.parametrize("dist, source, problem", [
    ("det-no-detect", NO_DETECT, "potrzeba klasy z `detect\\(request\\)` - patrz kontrakt Detector"),
    ("det-a-function", A_FUNCTION, "potrzeba klasy z `detect"),
    ("det-dict-settings", NOT_A_DATACLASS, "`settings` musi byc klasa dataclass"),
    ("det-list-setting", LIST_SETTING, "ustawienie colors ma typ"),
])
def test_a_detector_the_panel_cannot_build_is_refused_when_loaded(install_dist, dist, source, problem):
    name = install_detector(install_dist, dist, source)

    with pytest.raises(TypeError, match=problem):
        load_detector(name)


# --------------------------------------------------------------------------- strict settings


def test_its_settings_dataclass_becomes_the_form_fields(install_dist):
    name = install_detector(install_dist, "blocks-fields", WITH_SETTINGS)
    fields = {field.name: field for field in load_detector(name).fields}

    assert list(fields) == ["color", "size_m", "interval_s"]
    assert fields["color"].kind is str
    assert fields["color"].required
    assert fields["size_m"].kind is float
    assert fields["size_m"].default == 0.03


@pytest.mark.parametrize("dist, fields", [
    ("choices-number", "value: int = field(default=1, metadata={'choices': ('1', '2')})"),
    ("choices-string", "value: str = field(default='red', metadata={'choices': 'red'})"),
    ("choices-missing-selector", "value: str = field(default='', metadata={'choices_from': 'cli', "
     "'choices': {'codex': ('',)}})"),
    ("choices-missing-branch", "cli: str = field(default='codex', metadata={'choices': ('codex', 'claude')})\n"
     "    model: str = field(default='', metadata={'choices_from': 'cli', 'choices': {'codex': ('',)}})"),
])
def test_invalid_dropdown_metadata_is_refused_when_the_plugin_loads(install_dist, dist, fields):
    source = (
        "from dataclasses import dataclass, field\n\n@dataclass\nclass Settings:\n"
        f"    {fields}\n\nclass Detector:\n    settings = Settings\n"
        "    def detect(self, request):\n        return []\n")
    name = install_detector(install_dist, dist, source)

    with pytest.raises(TypeError, match="choices"):
        load_detector(name)


def test_the_detector_is_created_with_its_settings(install_dist):
    name = install_detector(install_dist, "blocks-created", WITH_SETTINGS)

    detector = load_detector(name).create({"color": "green", "interval_s": 2})

    assert type(detector).__name__ == "Detector"
    assert detector.config.color == "green"
    assert detector.config.size_m == 0.03
    assert detector.interval_s == 2.0
    assert isinstance(detector.interval_s, float)
    # The contract's defaults reach a subclass that does not set them.
    assert detector.pixel_error_px == 3.0


def test_an_unknown_key_is_refused(install_dist):
    name = install_detector(install_dist, "blocks-unknown", WITH_SETTINGS)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji blocks_unknown: \['colour'\]"):
        load_detector(name).create({"color": "red", "colour": "red"})


def test_a_missing_key_is_refused(install_dist):
    name = install_detector(install_dist, "blocks-missing", WITH_SETTINGS)

    with pytest.raises(ValueError, match=r"Brak kluczy w sekcji blocks_missing: \['color'\]"):
        load_detector(name).create({})


def test_a_detector_without_settings_refuses_keys_as_a_detector(install_dist):
    name = install_detector(install_dist, "base-free-keys", BASE_FREE)

    with pytest.raises(ValueError, match=r"^Detektor base_free_keys nie ma ustawien"):
        load_detector(name).create({"color": "red"})


def test_the_config_section_named_like_the_detector_prefills_the_settings(tmp_path, monkeypatch,
                                                                         install_dist):
    name = install_detector(install_dist, "blocks-config", WITH_SETTINGS)
    config = tmp_path / "local.yaml"
    config.write_text(f"{name}:\n  color: green\n  size_m: 0.04\n", encoding="utf-8")
    monkeypatch.setenv("MACHINEKIND_TWIN_CONFIG", str(config))

    raw = config_section(name)

    assert raw == {"color": "green", "size_m": 0.04}
    assert config_section("yolo") == {}
    assert load_detector(name).create(raw).config.size_m == 0.04


# --------------------------------------------------------------------------- one flat config namespace


def test_a_name_only_a_detector_registers_shares_no_section(install_dist):
    name = install_detector(install_dist, "blocks-alone", WITH_SETTINGS)

    assert shared_section(name) == ""


def test_a_strategy_with_the_detector_s_name_shares_its_section(install_dist):
    name = install_detector(install_dist, "both-kinds", WITH_SETTINGS)
    install_dist("both-kinds-strategy",
                 {"machinekind_twin.strategies": {"Both_Kinds": "both_kinds_strategy:Strategy"}})

    assert shared_section(name) == "strategia"


def test_a_backend_with_the_detector_s_name_shares_its_section(install_dist):
    """The test robot's package registers a backend named like the robot."""
    install_dist("testbot-detector", {GROUP: {BOT: "testbot_detector:Detector"}},
                 {"testbot_detector": BASE_FREE})

    assert shared_section(BOT) == "backend"


# --------------------------------------------------------------------------- the example cube detector


def test_the_example_cube_is_found_through_its_entry_point(cube_example):
    assert detectors()[cube_example].value == "cube_detector:ColourCube"

    plugin = load_detector(cube_example)

    assert plugin.name == "cube"
    assert plugin.cls.__name__ == "ColourCube"
    assert plugin.cls.__module__ == "cube_detector"
    assert plugin.source == "machinekind-twin-cube 0.1.0 (cube_detector:ColourCube)"
    assert [field.name for field in plugin.fields] == ["color", "size_m", "label", "min_area_px", "interval_s"]


def test_the_example_cube_is_created_with_its_settings(cube_example):
    from cube_detector import ColourCube, CubeSettings

    detector = load_detector(cube_example).create({"color": "green", "size_m": 0.04, "interval_s": 0.5})

    assert type(detector) is ColourCube
    assert detector.config == CubeSettings(color="green", size_m=0.04, interval_s=0.5)
    assert detector.interval_s == 0.5
    assert detector.pixel_error_px == 4.0


def test_the_example_cube_refuses_an_unknown_key(cube_example):
    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji cube: \['colour'\]"):
        load_detector(cube_example).create({"colour": "green"})


@pytest.mark.parametrize("raw, problem", [
    ({"color": "purple"}, "unknown color 'purple'"),
    ({"size_m": 0.0}, "size_m must be in"),
    ({"size_m": 0.5}, "size_m must be in"),
    ({"interval_s": 0.01}, "interval_s must be at least"),
])
def test_the_example_cube_refuses_settings_it_cannot_work_with(cube_example, raw, problem):
    with pytest.raises(ValueError, match=problem):
        load_detector(cube_example).create(raw)


def test_the_cube_config_section_prefills_the_example(tmp_path, monkeypatch, cube_example):
    config = tmp_path / "local.yaml"
    config.write_text("cube:\n  color: green\n  size_m: 0.04\n", encoding="utf-8")
    monkeypatch.setenv("MACHINEKIND_TWIN_CONFIG", str(config))

    raw = config_section(cube_example)

    assert raw == {"color": "green", "size_m": 0.04}
    assert load_detector(cube_example).create(raw).config.color == "green"


def test_the_example_cube_package_lives_outside_the_twin():
    assert "src" not in CUBE.parts
    assert (CUBE / "cube_detector" / "__init__.py").is_file()


def test_the_example_cube_imports_nothing_from_the_twin_at_runtime():
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(CUBE)!r})\n"
        "import cube_detector\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'machinekind_twin'))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "[]"


def test_the_example_cube_package_never_imports_the_twin():
    """Only the contracts (`machinekind_twin_api`), anywhere in the package, also inside a function."""
    for path in (CUBE / "cube_detector").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue

            for module in modules:
                assert module.split(".")[0] != "machinekind_twin", f"{path}: {module}"


# --------------------------------------------------------------------------- the LLM detector


def test_the_llm_detector_is_listed_through_its_entry_point(llm_package):
    assert llm_package in detectors()
    assert detectors()[llm_package].value == "machinekind_twin_llm:LlmDetector"

    plugin = load_detector(llm_package)

    assert plugin.name == "llm"
    assert plugin.cls.__name__ == "LlmDetector"
    assert plugin.source == "machinekind-twin-llm 0.1.0 (machinekind_twin_llm:LlmDetector)"


def test_the_llm_settings_draw_as_str_float_int_and_bool(llm_package):
    fields = {field.name: field for field in load_detector(llm_package).fields}

    assert {name: field.kind for name, field in fields.items()} == {
        "cli": str, "model": str, "effort": str, "objects": str, "cameras": str, "prompt": str,
        "interval_s": float, "timeout_s": float, "max_side_px": int, "pixel_error_px": float,
        "reuse_if_still": bool, "reuse_max_age_s": float, "max_calls_per_hour": int, "min_confidence": float, "executable": str,
        "claude_images": str,
    }
    assert not any(field.required for field in fields.values())
    assert fields["cli"].default == "codex"
    assert fields["interval_s"].default == 10.0


def test_the_llm_model_choices_follow_the_cli_without_limiting_config_models(llm_package):
    plugin = load_detector(llm_package)
    fields = {field.name: field for field in plugin.fields}

    assert fields["cli"].options({}) == ("codex", "claude")
    assert fields["model"].choices_from == "cli"
    codex = fields["model"].options({"cli": "codex"})
    claude = fields["model"].options({"cli": "claude"})
    assert "" in codex and "" in claude
    assert "gpt-6-astra" in codex and "gpt-6-astra" not in claude
    assert "sonnet" in claude and "sonnet" not in codex
    assert plugin.create({"cli": "codex", "model": "gpt-future"}).config.model == "gpt-future"
    assert plugin.create({"cli": "claude", "model": "claude-future"}).config.model == "claude-future"


def test_the_llm_config_section_prefills_the_detector(tmp_path, monkeypatch, llm_package):
    config = tmp_path / "local.yaml"
    config.write_text("llm:\n  cli: claude\n  model: sonnet\n  objects: red pen, cube\n  interval_s: 20\n",
                      encoding="utf-8")
    monkeypatch.setenv("MACHINEKIND_TWIN_CONFIG", str(config))

    raw = config_section(llm_package)
    detector = load_detector(llm_package).create(raw)

    assert raw == {"cli": "claude", "model": "sonnet", "objects": "red pen, cube", "interval_s": 20}
    assert detector.config.cli == "claude"
    assert detector.config.model == "sonnet"
    assert detector.interval_s == 20.0


def test_the_llm_detector_refuses_settings_it_cannot_work_with(llm_package):
    with pytest.raises(ValueError, match="is an OpenAI model and cli is claude"):
        load_detector(llm_package).create({"cli": "claude"})
