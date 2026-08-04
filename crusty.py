"""Client for the crusty camera dashboard API (crusty.myspys:9502).

Pulls real footage for face-recognition evaluation and enrolment.

Two ways to get pixels out, and the cheap one is usually right:

  frames(event_id)   -> JPEGs the dashboard extracts server-side with ffmpeg.
                        Its "detections" strategy returns the frames where something
                        was actually detected, which is exactly what we want and skips
                        transferring a whole video to look at three useful frames.

  download(path)     -> the full video file, when frame-level control is needed
                        (our own adaptive sampling, motion gating, tracking).

Auth: X-API-Key header, key checked against the dashboard's api_keys table. Set
CRUSTY_API_KEY in the environment. Basic auth also works but is the frontend's path,
not ours.
"""
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional

BASE_URL = os.environ.get("CRUSTY_URL", "http://crusty.myspys:9502")
API_KEY = os.environ.get("CRUSTY_API_KEY", "")
# Basic auth is the frontend's path; usable for us when no API key has been minted.
BASIC_USER = os.environ.get("CRUSTY_USER", "")
BASIC_PASS = os.environ.get("CRUSTY_PASS", "")


def _auth_header() -> Optional[tuple]:
    if API_KEY:
        return ("X-API-Key", API_KEY)
    if BASIC_USER:
        import base64
        token = base64.b64encode(f"{BASIC_USER}:{BASIC_PASS}".encode()).decode()
        return ("Authorization", f"Basic {token}")
    return None


class CrustyError(RuntimeError):
    pass


def _request(path: str, params: Optional[dict] = None, timeout: int = 60,
             raw: bool = False):
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None})

    request = urllib.request.Request(url)
    header = _auth_header()
    if header:
        request.add_header(*header)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise CrustyError(
                "401 from crusty. Set CRUSTY_API_KEY, or CRUSTY_USER/CRUSTY_PASS "
                "for basic auth."
            ) from exc
        raise CrustyError(f"{exc.code} {exc.reason} for {path}") from exc

    return payload if raw else json.loads(payload)


def health() -> dict:
    """Unauthenticated liveness check — useful to separate 'down' from 'no key'."""
    return _request("/health", timeout=15)


def cameras() -> List[dict]:
    result = _request("/api/v1/cameras")
    return result if isinstance(result, list) else result.get("cameras", result)


def events(camera_name: Optional[str] = None, detection_type: str = "person",
           since: Optional[str] = None, until: Optional[str] = None,
           limit: int = 100, offset: int = 0, order: str = "desc") -> List[dict]:
    """List detection events. Defaults to people, newest first."""
    result = _request("/api/v1/events", {
        "camera_name": camera_name, "detection_type": detection_type,
        "from": since, "to": until, "limit": limit, "offset": offset, "order": order,
    })
    return result if isinstance(result, list) else result.get("events", [])


def iter_events(page_size: int = 100, max_events: Optional[int] = None,
                **filters) -> Iterator[dict]:
    """Page through events so a large sweep doesn't need one giant request."""
    offset, yielded = 0, 0
    while True:
        batch = events(limit=page_size, offset=offset, **filters)
        if not batch:
            return
        for event in batch:
            yield event
            yielded += 1
            if max_events and yielded >= max_events:
                return
        offset += len(batch)
        if len(batch) < page_size:
            return


def event(event_id: str) -> dict:
    return _request(f"/api/v1/events/{event_id}")


def frames(event_id: str, count: int = 5, strategy: str = "detections") -> dict:
    """Server-side frame extraction.

    strategy: 'detections' (frames where something was detected - usually what you
    want), 'analyzed' (the exact frames camera-analyzer used), or 'uniform'.
    """
    return _request(f"/api/v1/events/{event_id}/frames",
                    {"count": count, "strategy": strategy}, timeout=180)


def download(video_path: str, destination: Path, chunk: int = 1 << 20) -> Path:
    """Stream a video file to disk. Skips the transfer if already present."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/api/video?" + urllib.parse.urlencode({"path": video_path})
    request = urllib.request.Request(url)
    header = _auth_header()
    if header:
        request.add_header(*header)

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=600) as response, \
                open(temporary, "wb") as handle:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                handle.write(block)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise CrustyError(f"{exc.code} downloading {video_path}") from exc

    # Rename only on success, so an interrupted transfer never looks like a good file.
    temporary.rename(destination)
    return destination


def frame_urls(event_id: str, count: int = 5, strategy: str = "detections") -> List[str]:
    """Absolute URLs for extracted frames (they expire ~10 minutes after extraction)."""
    payload = frames(event_id, count=count, strategy=strategy)
    return [f"{BASE_URL}{f['url']}" for f in payload.get("frames", [])]


def fetch(url: str, timeout: int = 120) -> bytes:
    """GET raw bytes from an absolute dashboard URL, applying auth."""
    request = urllib.request.Request(url)
    header = _auth_header()
    if header:
        request.add_header(*header)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def browse(camera: str, date: str) -> List[dict]:
    """List video files for a camera on a date (YYYY-MM-DD). Authenticated."""
    result = _request(f"/api/browse/{camera}/{date}")
    return result.get("files", result) if isinstance(result, dict) else result


def find_video(camera: str, timestamp: str) -> Optional[str]:
    """Resolve a video path from camera + 'YYYY-MM-DD HH:MM:SS'.

    Uses the authenticated browse listing rather than the frames endpoint. Filenames
    embed the capture time (…_YYYYMMDDHHMMSS.mp4), so an exact stamp match is reliable;
    we fall back to the nearest file within a minute for clock skew between the event
    record and the recorder.
    """
    date_part, time_part = timestamp.split(" ")
    stamp = date_part.replace("-", "") + time_part.replace(":", "")

    # The on-disk folder does not always match the display name: the dashboard shows
    # "reolink-yard-farm" while the directory is "reolinkyardfarm". Try the obvious
    # variants rather than hardcoding a mapping that will drift.
    files = []
    for candidate in (camera, camera.replace("-", ""), camera.replace("-", "_")):
        try:
            files = browse(candidate, date_part)
            if files:
                break
        except CrustyError:
            continue
    for entry in files:
        if stamp in entry.get("name", ""):
            return entry.get("path")

    # nearest within 60s
    best, best_delta = None, 61
    target = int(time_part[:2]) * 3600 + int(time_part[3:5]) * 60 + int(time_part[6:8])
    for entry in files:
        name = entry.get("name", "")
        digits = "".join(c for c in name if c.isdigit())
        if len(digits) < 14:
            continue
        hhmmss = digits[-6:]
        try:
            seconds = (int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60
                       + int(hhmmss[4:6]))
        except ValueError:
            continue
        delta = abs(seconds - target)
        if delta < best_delta:
            best, best_delta = entry.get("path"), delta
    return best


def video_path_of(event_record: dict) -> Optional[str]:
    """Find the video path in an event record, tolerating field-name drift."""
    for key in ("video_path", "videoPath", "file_path", "filePath", "path", "filename"):
        value = event_record.get(key)
        if value:
            return value
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="crusty dashboard client")
    parser.add_argument("command", choices=["health", "cameras", "events", "pull"])
    parser.add_argument("--camera")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--type", default="person")
    parser.add_argument("--out", default="/home/grazzy/media/crusty")
    args = parser.parse_args()

    if args.command == "health":
        print(json.dumps(health(), indent=2))
        print("API key set" if API_KEY else "NO API KEY — set CRUSTY_API_KEY")

    elif args.command == "cameras":
        for cam in cameras():
            print(f"  {cam.get('name', '?'):28} id={cam.get('id', '?')}")

    elif args.command == "events":
        for e in events(camera_name=args.camera, detection_type=args.type,
                        since=args.since, until=args.until, limit=args.limit):
            print(f"  {e.get('id')}  {e.get('timestamp', '')[:19]}  "
                  f"{str(e.get('camera_name')):24} {video_path_of(e)}")

    elif args.command == "pull":
        out = Path(args.out)
        pulled = 0
        for e in iter_events(camera_name=args.camera, detection_type=args.type,
                             since=args.since, until=args.until,
                             max_events=args.limit):
            path = video_path_of(e)
            if not path:
                continue
            target = out / str(e.get("camera_name", "unknown")) / Path(path).name
            try:
                download(path, target)
                pulled += 1
                print(f"  {target}")
            except CrustyError as exc:
                print(f"  FAILED {path}: {exc}")
        print(f"pulled {pulled} videos into {out}")
