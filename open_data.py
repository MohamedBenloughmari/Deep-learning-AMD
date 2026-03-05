from pathlib import Path

parts = sorted(Path("data").glob("Task_2.zip.*"))  # .001, .002, ...
out_zip = Path("data") / "Task_2.zip"

with out_zip.open("wb") as w:
    for p in parts:
        with p.open("rb") as r:
            w.write(r.read())

print("Wrote:", out_zip)