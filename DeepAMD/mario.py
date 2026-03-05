import zipfile
import sys
from pathlib import Path

'''
This script is used to extract the Task_2.zip file
'''

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python mario.py <path_to_Task_2.zip>")
 
    zip_path = Path(sys.argv[1])
 
    with zipfile.ZipFile(zip_path, "r") as z:
        print("Files inside zip:")
        for name in z.namelist()[:50]:
            print(" -", name)
 
        # extract everything to Task_2 folder
        z.extractall(".")
        print("\nExtracted to Task_2/")
 
 
 
    parts = sorted(zip_path.parent.glob(f"{zip_path.stem}.zip.*"))  # .001, .002, ...
    out_zip = zip_path.parent / f"{zip_path.stem}.zip"
 
    with out_zip.open("wb") as w:
        for p in parts:
            with p.open("rb") as r:
                w.write(r.read())
 
    print("Wrote:", out_zip)