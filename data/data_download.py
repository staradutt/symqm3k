"""Downloads SymQM-3k CSV files from HuggingFace."""
from huggingface_hub import hf_hub_download
import shutil

REPO_ID = "taradutt007/symqm3k"  
FILES   = [
    "target_summary_final.csv",
    "atom_details_final.csv",
    "orbital_profile_final.csv",
    "splits.csv",
]

for fname in FILES:
    path = hf_hub_download(repo_id=REPO_ID,
                           filename=fname,
                           repo_type="dataset")
    shutil.copy(path, fname)
    print(f"Downloaded {fname}")

print("Done. All files saved to current directory.")