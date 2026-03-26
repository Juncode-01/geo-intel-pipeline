"""Run geospatial-to-text conversion for fetched datasets."""

from agents.converter_agent_impl import ConverterAgent


def run_conversion():
    print("=" * 60)
    print("GEOSPATIAL TO TEXT CONVERSION")
    print("=" * 60)
    ConverterAgent().run()


if __name__ == "__main__":
    run_conversion()
