import argparse
import csv
import re
import shutil
import zipfile
from pathlib import Path


TRANSCRIPT_PATTERNS = (
    "{participant_id}_TRANSCRIPT.csv",
    "{participant_id} TRANSCRIPT.csv",
    "{participant_id}_transcript.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract participant archives for a DAIC-WOZ split and collect "
            "transcript CSV files into a single directory."
        )
    )
    parser.add_argument(
        "--split-csv",
        required=True,
        help="CSV file containing a Participant_ID column or participant IDs in the first column.",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing participant zip archives.",
    )
    parser.add_argument(
        "--extract-dir",
        default="data/extracted_daic_woz",
        help="Directory where archives will be extracted.",
    )
    parser.add_argument(
        "--transcript-dir",
        default="data/daic_woz_transcripts",
        help="Directory where normalized transcript CSV files will be copied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract archives and overwrite copied transcript files.",
    )
    return parser.parse_args()


def load_participant_ids(csv_path):
    participant_ids = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "Participant_ID" in reader.fieldnames:
            for row in reader:
                if row.get("Participant_ID"):
                    participant_ids.append(int(float(row["Participant_ID"])))
            return participant_ids

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if row and row[0].strip():
                participant_ids.append(int(float(row[0])))
    return participant_ids


def candidate_archive_names(participant_id):
    return [
        f"{participant_id}_P.zip",
        f"{participant_id} P.zip",
        f"{participant_id}. P.zip",
        f"{participant_id}_p.zip",
        f"{participant_id}.zip",
    ]


def find_archive(source_dir, participant_id):
    source_path = Path(source_dir)
    for candidate in candidate_archive_names(participant_id):
        candidate_path = source_path / candidate
        if candidate_path.exists():
            return candidate_path

    pattern = re.compile(rf"(^|[^0-9]){participant_id}([^0-9]|$)")
    for archive_path in source_path.glob("*.zip"):
        if pattern.search(archive_path.name):
            return archive_path
    return None


def extract_archive(archive_path, extract_dir, participant_id, overwrite=False):
    participant_dir = Path(extract_dir) / str(participant_id)
    if participant_dir.exists() and not overwrite:
        print(f"skip  extract {archive_path.name}")
        return participant_dir

    if participant_dir.exists():
        shutil.rmtree(participant_dir)
    participant_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(participant_dir)
    print(f"done  extract {archive_path.name}")
    return participant_dir


def transcript_names(participant_id):
    return [pattern.format(participant_id=participant_id).lower() for pattern in TRANSCRIPT_PATTERNS]


def find_transcript_file(extracted_dir, participant_id):
    expected_names = set(transcript_names(participant_id))
    for file_path in Path(extracted_dir).rglob("*"):
        if file_path.is_file() and file_path.name.lower() in expected_names:
            return file_path
    return None


def copy_transcript(transcript_path, transcript_dir, participant_id, overwrite=False):
    destination = Path(transcript_dir) / f"{participant_id}_TRANSCRIPT.csv"
    if destination.exists() and not overwrite:
        print(f"skip  transcript {destination.name}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transcript_path, destination)
    print(f"done  transcript {destination.name}")
    return destination


def main():
    args = parse_args()
    participant_ids = load_participant_ids(args.split_csv)
    if not participant_ids:
        raise SystemExit("No participant IDs found in split CSV.")

    extract_dir = Path(args.extract_dir)
    transcript_dir = Path(args.transcript_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    transcript_count = 0
    missing_archives = []
    missing_transcripts = []

    for participant_id in participant_ids:
        archive_path = find_archive(args.source_dir, participant_id)
        if archive_path is None:
            print(f"miss  archive for participant {participant_id}")
            missing_archives.append(participant_id)
            continue

        participant_extract_dir = extract_archive(
            archive_path,
            extract_dir,
            participant_id,
            overwrite=args.overwrite,
        )
        extracted_count += 1

        transcript_path = find_transcript_file(participant_extract_dir, participant_id)
        if transcript_path is None:
            print(f"miss  transcript for participant {participant_id}")
            missing_transcripts.append(participant_id)
            continue

        copy_transcript(
            transcript_path,
            transcript_dir,
            participant_id,
            overwrite=args.overwrite,
        )
        transcript_count += 1

    print(f"Participants in split: {len(participant_ids)}")
    print(f"Archives extracted: {extracted_count}")
    print(f"Transcripts copied: {transcript_count}")
    print(f"Missing archives: {len(missing_archives)}")
    print(f"Missing transcripts: {len(missing_transcripts)}")
    print(f"Extract dir: {extract_dir.resolve()}")
    print(f"Transcript dir: {transcript_dir.resolve()}")


if __name__ == "__main__":
    main()
