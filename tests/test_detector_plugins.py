"""Detectors from installed packages: found through entry points, built strictly from their settings.

The detectors here are inline fakes that `install_detector` installs the way pip would. Their
dist names start apart from the strategy tests' ones: a module once imported stays in
`sys.modules` for the whole run. The example cube detector (`examples/detectors/cube`) gets its own cases once it exists.
"""

from __future__ import annotations

import pytest
from conftest import BOT, install_detector
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
