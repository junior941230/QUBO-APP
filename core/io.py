from config import DESTINATION_DIR
from parser import parse_seizure_file
import re

def discover_subjects(base_dir=DESTINATION_DIR):
    if not base_dir.exists():
        return []
    pattern = re.compile(r"chb\d{2}")
    return sorted(
        item.name for item in base_dir.iterdir()
        if item.is_dir() and pattern.fullmatch(item.name)
    )


def collect_files_and_seizures(subjects, max_files_per_subject):
    file_paths = []
    seizure_times = {}
    notes = []

    for subject in subjects:
        subject_dir = DESTINATION_DIR / subject
        summary_path = subject_dir / f"{subject}-summary.txt"

        if not subject_dir.exists():
            notes.append(f"Skip {subject}: subject directory not found")
            continue
        if not summary_path.exists():
            notes.append(f"Skip {subject}: summary file not found")
            continue

        seizure_times.update(parse_seizure_file(str(summary_path)))
        edf_files = sorted(subject_dir.glob("*.edf"))
        if max_files_per_subject > 0:
            edf_files = edf_files[:max_files_per_subject]
        file_paths.extend(str(path) for path in edf_files)

    return file_paths, seizure_times, notes