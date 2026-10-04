"""Call the deployed Modal preprocessor and specialist from the local CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import modal
from dotenv import load_dotenv


APP_NAME = "week8-price-pipeline"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an item's whole USD price on Modal.")
    parser.add_argument("text", help="Raw item title and details")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    if project_root.name == "myfiles" and project_root.parent.name == "week8":
        load_dotenv(project_root.parent / ".env", override=False)

    preprocessor = modal.Cls.from_name(APP_NAME, "Preprocessor")()
    specialist = modal.Cls.from_name(APP_NAME, "Specialist")()
    description = preprocessor.normalize.remote(args.text)
    price = specialist.price.remote(description)
    print("Canonical item description:\n" + description)
    print(f"Predicted price: ${price}")


if __name__ == "__main__":
    main()
