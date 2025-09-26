#!/usr/bin/env python3
"""
Sync and manage evaluations stored on a deployed Railway app.

Features
- List users that have saved evaluations
- Download all or per-user evaluations to a local folder
- Delete a user's evaluations or a specific file on the server (admin)

Quick examples
- Sync everything to ./synced_evaluations:
    python scripts/sync_from_railway.py

- Sync a single user into local repo evaluations/ (merge into local data):
    python scripts/sync_from_railway.py --user test --merge-into-local-evals

- Delete one user remotely (requires ADMIN_TOKEN):
    ADMIN_TOKEN=xxxx python scripts/sync_from_railway.py --delete-user test --yes

- Delete a single file remotely (requires ADMIN_TOKEN):
    ADMIN_TOKEN=xxxx python scripts/sync_from_railway.py --delete-file test amateur_piano_1.json --yes

Requires: requests
  pip install requests
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
import os

import requests


DEFAULT_BASE = "https://piano-performance-marker-production.up.railway.app"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_EVALS = REPO_ROOT / "piano-performance-marker" / "evaluations"
DEFAULT_DOWNLOADS = REPO_ROOT / "piano-performance-marker" / "downloaded_evaluations"


def get_users(base: str) -> list[dict]:
    r = requests.get(f"{base}/api/users", timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("users", [])


def get_evals(base: str, username: str, with_data: bool = True) -> list[dict]:
    r = requests.get(f"{base}/api/evaluations/{username}", params={"withData": int(with_data)}, timeout=30)
    r.raise_for_status()
    return r.json().get("evaluations", [])


def ensure_out_dir(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_user(base: str, username: str, out_dir: Path) -> int:
    user_out = ensure_out_dir(out_dir / username)
    evals = get_evals(base, username, with_data=True)
    downloaded = 0
    for ev in evals:
        fn = ev.get("filename") or "unknown.json"
        data = ev.get("data")
        if data is None:
            q = requests.get(
                f"{base}/api/get-evaluation",
                params={"username": username, "filename": fn},
                timeout=20,
            )
            j = q.json()
            data = j.get("data")
        with open(user_out / fn, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        downloaded += 1
    print(f"synced {downloaded} files for {username} → {user_out}")
    return downloaded


def download_all(base: str, out_dir: Path) -> int:
    ensure_out_dir(out_dir)
    users = get_users(base)
    total = 0
    for u in users:
        total += download_user(base, u["username"], out_dir)
    print(f"synced total {total} files for {len(users)} users → {out_dir}")
    return total


def delete_user(base: str, username: str, admin_token: str | None):
    headers = {"x-admin-token": admin_token} if admin_token else {}
    r = requests.delete(f"{base}/api/evaluations/{username}", headers=headers, timeout=30)
    try:
        print(r.status_code, r.json())
    except Exception:
        print(r.status_code, r.text)


def delete_file(base: str, username: str, filename: str, admin_token: str | None):
    headers = {"x-admin-token": admin_token} if admin_token else {}
    r = requests.delete(
        f"{base}/api/evaluation/{username}/{filename}", headers=headers, timeout=30
    )
    try:
        print(r.status_code, r.json())
    except Exception:
        print(r.status_code, r.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"Base URL (default: {DEFAULT_BASE})",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_DOWNLOADS),
        help=f"Output directory to store JSON files (default: {DEFAULT_DOWNLOADS})",
    )
    ap.add_argument("--user", help="Sync only this username (default: all users)")
    ap.add_argument("--merge-into-local-evals", action="store_true", help=f"Write into repo evaluations directory: {DEFAULT_LOCAL_EVALS}")
    ap.add_argument("--delete-user", dest="del_user", help="Delete all evaluations for a username on server (admin)")
    ap.add_argument("--delete-file", nargs=2, metavar=("USERNAME", "FILENAME"), help="Delete one file for a user on server (admin)")
    ap.add_argument("--admin-token", help="Admin token for deletion endpoints (or set ADMIN_TOKEN env)")
    ap.add_argument("--yes", action="store_true", help="Do not prompt for confirmation on destructive ops")
    args = ap.parse_args()

    base = args.base.rstrip('/')

    # Resolve output
    out_dir = Path(args.out)
    if args.merge_into_local_evals:
        out_dir = DEFAULT_LOCAL_EVALS

    # Resolve admin token
    admin_token = args.admin_token or os.getenv("ADMIN_TOKEN")

    # Destructive ops first
    if args.del_user:
        if not admin_token:
            print("error: --admin-token or ADMIN_TOKEN env is required for deletion", file=sys.stderr)
            return 2
        if not args.yes:
            ans = input(f"Delete ALL files for user '{args.del_user}' on {base}? [y/N]: ")
            if ans.strip().lower() != 'y':
                print("aborted")
                return 1
        delete_user(base, args.del_user, admin_token)
        return 0

    if args.delete_file:
        if not admin_token:
            print("error: --admin-token or ADMIN_TOKEN env is required for deletion", file=sys.stderr)
            return 2
        u, fn = args.delete_file
        if not args.yes:
            ans = input(f"Delete file '{fn}' for user '{u}' on {base}? [y/N]: ")
            if ans.strip().lower() != 'y':
                print("aborted")
                return 1
        delete_file(base, u, fn, admin_token)
        return 0

    # Sync
    if args.user:
        download_user(base, args.user, out_dir)
    else:
        download_all(base, out_dir)


if __name__ == "__main__":
    sys.exit(main())
