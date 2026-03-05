import zipfile
from pathlib import Path

zip_path = Path("data") / "Task_2.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    print("Files inside zip:")
    for name in z.namelist()[:50]:
        print(" -", name)

    # extract everything
    z.extractall("data_out")