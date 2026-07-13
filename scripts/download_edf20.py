"""Download preprocessed Sleep-EDF-20 NPZ files from NTU Dataverse."""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_API = (
    "https://researchdata.ntu.edu.sg/api/datasets/:persistentId/"
    "?persistentId=doi:10.21979/N9/MA1AVG"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_file_list():
    req = urllib.request.Request(DATASET_API, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    return payload["data"]["latestVersion"]["files"]


def download_one(file_meta, output_dir):
    data_file = file_meta["dataFile"]
    fname = data_file["filename"]
    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path):
        return fname, "skipped", os.path.getsize(out_path)

    url = f"https://researchdata.ntu.edu.sg/api/access/datafile/{data_file['id']}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as resp:
        content = resp.read()
    with open(out_path, "wb") as handle:
        handle.write(content)
    return fname, "downloaded", len(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/edf_20_npz")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Limit number of files (useful for quick smoke tests)")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = fetch_file_list()
    if args.max_files:
        files = files[: args.max_files]

    print(f"Fetching {len(files)} files into {args.output_dir} ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, f, args.output_dir) for f in files]
        for fut in as_completed(futures):
            fname, status, size = fut.result()
            print(f"  {fname}: {status} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
