"""Run geospatial-to-text conversion for fetched datasets."""

from importlib.util import module_from_spec, spec_from_file_location
import sys
from pathlib import Path


CONVERTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "agents"
    / "converter_agent.py"
    / "converter.py"
)


def _load_converter_agent_class():
    spec = spec_from_file_location("converter_module", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load converter module from {CONVERTER_PATH}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ConverterAgent


def run_conversion():
    print("=" * 60)
    print("GEOSPATIAL TO TEXT CONVERSION")
    print("=" * 60)
    converter_cls = _load_converter_agent_class()
    converter_cls().run()


if __name__ == "__main__":
    run_conversion()
