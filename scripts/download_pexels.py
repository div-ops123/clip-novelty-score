"""Phase 1 of the validation-dataset pipeline: build the raw clip pool.

Searches Pexels' free video API for POV/first-person footage across the
task categories in `config.SEARCH_TERMS`, downloads a deduplicated pool of
clips up to `config.TARGET_POOL_SIZE`, and records their metadata in
`data/raw_manifest.csv` for the later partition/build phases.

Usage:
    python scripts/download_pexels.py discover
    python scripts/download_pexels.py fetch
"""

import argparse
import csv
import os
import sys
import time

import requests
from dotenv import load_dotenv

import config

FIELDNAMES = [
    "pexels_id",
    "search_term",
    "source_url",
    "rendition_link",
    "quality",
    "width",
    "height",
    "duration_sec",
    "local_path",
]


def _load_api_key() -> str:
    """Loads PEXELS_API_KEY from the repo's .env file.

    Returns:
        The Pexels API key.

    Raises:
        SystemExit: If PEXELS_API_KEY is not set.
    """
    load_dotenv(config.REPO_ROOT / ".env")
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        sys.exit("PEXELS_API_KEY not set. Put it in .env (see .env.example).")
    return key


def _read_existing_manifest() -> list[dict]:
    """Loads previously downloaded pool rows, if any, for resumable runs."""
    if not config.RAW_MANIFEST_PATH.exists():
        return []
    with open(config.RAW_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_manifest(rows: list[dict]) -> None:
    """Overwrites data/raw_manifest.csv with the given rows."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RAW_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _pick_rendition(video_files: list[dict]) -> dict | None:
    """Picks the smallest mp4 rendition, preferring `config.PREFERRED_QUALITY`.

    Args:
        video_files: The `video_files` list from a Pexels search/video result.

    Returns:
        The chosen rendition dict, or None if no mp4 rendition exists.
    """
    mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4"]
    if not mp4_files:
        return None
    preferred = [f for f in mp4_files if f.get("quality") == config.PREFERRED_QUALITY]
    pool = preferred if preferred else mp4_files
    return min(pool, key=lambda f: f.get("width") or 10**9)


def _get_with_backoff(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """Issues a GET request, retrying with backoff on HTTP 429 responses.

    Args:
        session: The requests session to issue the call on.
        url: Target URL.
        **kwargs: Forwarded to `session.get` (e.g. `params`).

    Returns:
        The final response, whether successful or still rate-limited after
        all retries.
    """
    for attempt in range(4):
        resp = session.get(url, **kwargs)
        if resp.status_code != 429:
            return resp
        wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
        print(f"  rate limited, waiting {wait}s...")
        time.sleep(wait)
    return resp


def _download_file(session: requests.Session, url: str, dest_path) -> None:
    """Streams a URL to a local file in chunks.

    Args:
        session: The requests session to issue the call on.
        url: Direct download URL for the video rendition.
        dest_path: Local filesystem path to write to.
    """
    with session.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def discover(api_key: str) -> None:
    """Builds the raw clip pool by searching and downloading from Pexels.

    Iterates over `config.SEARCH_TERMS`, paging through Pexels video search
    results per term, skipping clips already in the pool (by Pexels ID) or
    longer than `config.MAX_DURATION_SEC`, and downloading up to
    `config.MAX_PER_TERM` clips per term until `config.TARGET_POOL_SIZE`
    unique clips have been collected in total. The per-term cap keeps the
    pool spread across task categories rather than exhausting one search
    term first. Resumable: rows already in `raw_manifest.csv` are kept and
    not re-downloaded.

    Args:
        api_key: Pexels API key used for the Authorization header.
    """
    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_existing_manifest()
    seen_ids = {row["pexels_id"] for row in rows}

    per_term_count = {}
    for row in rows:
        per_term_count[row["search_term"]] = per_term_count.get(row["search_term"], 0) + 1

    for term in config.SEARCH_TERMS:
        if len(seen_ids) >= config.TARGET_POOL_SIZE:
            break
        page = 1
        while (
            len(seen_ids) < config.TARGET_POOL_SIZE
            and per_term_count.get(term, 0) < config.MAX_PER_TERM
        ):
            resp = _get_with_backoff(
                session,
                config.PEXELS_SEARCH_URL,
                params={"query": term, "per_page": config.RESULTS_PER_PAGE, "page": page},
            )
            resp.raise_for_status()
            data = resp.json()
            videos = data.get("videos", [])
            if not videos:
                break

            for video in videos:
                if len(seen_ids) >= config.TARGET_POOL_SIZE:
                    break
                if per_term_count.get(term, 0) >= config.MAX_PER_TERM:
                    break
                video_id = str(video["id"])
                if video_id in seen_ids:
                    continue
                duration = video.get("duration", 0)
                if duration > config.MAX_DURATION_SEC:
                    continue
                rendition = _pick_rendition(video.get("video_files", []))
                if rendition is None:
                    print(f"  skipping {video_id}: no mp4 rendition found")
                    continue

                local_path = config.RAW_DIR / f"{video_id}.mp4"
                print(f"[{term}] downloading {video_id} ({rendition.get('width')}px)...")
                try:
                    _download_file(session, rendition["link"], local_path)
                except requests.RequestException as e:
                    print(f"  failed to download {video_id}: {e}")
                    continue

                rows.append(
                    {
                        "pexels_id": video_id,
                        "search_term": term,
                        "source_url": video.get("url", ""),
                        "rendition_link": rendition["link"],
                        "quality": rendition.get("quality", ""),
                        "width": rendition.get("width", ""),
                        "height": rendition.get("height", ""),
                        "duration_sec": duration,
                        "local_path": str(local_path.relative_to(config.REPO_ROOT)).replace("\\", "/"),
                    }
                )
                seen_ids.add(video_id)
                per_term_count[term] = per_term_count.get(term, 0) + 1
                time.sleep(1)

            if not data.get("next_page"):
                break
            page += 1

    _write_manifest(rows)
    print(f"\nPool size: {len(rows)} clips -> {config.RAW_MANIFEST_PATH}")
    if len(rows) < config.TARGET_POOL_SIZE:
        print(f"WARNING: only found {len(rows)}/{config.TARGET_POOL_SIZE}; consider adding search terms.")


def fetch(api_key: str) -> None:
    """Re-downloads any pool clips missing from disk.

    Reads `raw_manifest.csv` and, for each row whose `local_path` no longer
    exists locally (e.g. after cloning the repo, since video files are
    gitignored), looks up a fresh download link via the Pexels video-by-id
    endpoint — the original search-result rendition link can expire — and
    re-downloads it to the recorded path. A row whose Pexels video has since
    been removed is skipped with a warning rather than aborting the run.

    Args:
        api_key: Pexels API key used for the Authorization header.

    Raises:
        SystemExit: If `raw_manifest.csv` does not exist yet.
    """
    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    rows = _read_existing_manifest()
    if not rows:
        sys.exit("No raw_manifest.csv found — run 'discover' first (or on the original machine).")

    for row in rows:
        local_path = config.REPO_ROOT / row["local_path"]
        if local_path.exists():
            continue
        video_id = row["pexels_id"]
        print(f"re-fetching {video_id}...")
        resp = _get_with_backoff(session, config.PEXELS_VIDEO_URL.format(video_id=video_id))
        if resp.status_code == 404:
            print(f"  WARNING: video {video_id} no longer available on Pexels, skipping")
            continue
        resp.raise_for_status()
        video = resp.json()
        rendition = _pick_rendition(video.get("video_files", []))
        if rendition is None:
            print(f"  WARNING: no mp4 rendition for {video_id} anymore, skipping")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_file(session, rendition["link"], local_path)
        except requests.RequestException as e:
            print(f"  failed to download {video_id}: {e}")
        time.sleep(1)


def main() -> None:
    """Parses CLI arguments and dispatches to `discover` or `fetch`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["discover", "fetch"])
    args = parser.parse_args()

    api_key = _load_api_key()
    if args.mode == "discover":
        discover(api_key)
    else:
        fetch(api_key)


if __name__ == "__main__":
    main()
