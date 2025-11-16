#!/usr/bin/env python3
from __future__ import annotations
import subprocess

if __name__ == "__main__":
    # Simple wrapper to run predict pipeline; extend to fetch data hourly if needed
    subprocess.run(["python", "-m", "src.pipelines.predict_pipeline"], check=True)
