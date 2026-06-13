"""
Mercurial リポジトリ情報を `hg` コマンド経由で取得するユーティリティ。

使用例（lume ディレクトリから実行する場合、リポジトリが src にある想定）::

    from utils.hg_utils import get_hg_info
    info = get_hg_info(repo_path="src")
    mlflow.set_tags({
        "hg.branch": info["branch"],
        "hg.revision": info["revision"],
        "hg.dirty": info["dirty"],
    })
    mlflow.log_text(info["diff"], "hg_diff.patch")
"""

from __future__ import annotations

import subprocess
from typing import Any


def _run_hg(
    args: list[str],
    repo_path: str | None = None,
    timeout: float = 10.0,
) -> str | None:
    cmd = ["hg"]
    if repo_path:
        cmd += ["--cwd", repo_path]
    cmd += args
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return out.stdout.rstrip("\n")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_hg_branch(repo_path: str | None = None) -> str | None:
    """現在のブランチ名（`hg branch` 相当）。"""
    return _run_hg(["branch"], repo_path=repo_path)


def get_hg_revision(repo_path: str | None = None, full: bool = False) -> str | None:
    """
    作業ディレクトリのリビジョン識別子（短い / 完全な node hash）。
    未コミット変更がある場合は末尾に ``+`` が付く。
    """
    args = ["id", "-i"]
    if full:
        args.append("--debug")
    return _run_hg(args, repo_path=repo_path)


def get_hg_rev_number(repo_path: str | None = None) -> str | None:
    """ローカルリビジョン番号（`hg log -r . --template "{rev}"`）。"""
    return _run_hg(
        ["log", "-r", ".", "--template", "{rev}"],
        repo_path=repo_path,
    )


def get_hg_tags(repo_path: str | None = None) -> list[str]:
    """現在のリビジョンに付いたタグのリスト。"""
    raw = _run_hg(
        ["log", "-r", ".", "--template", "{tags}"],
        repo_path=repo_path,
    )
    if raw is None or not raw.strip():
        return []
    return [t for t in raw.split() if t]


def get_hg_diff(repo_path: str | None = None) -> str:
    """ワーキングコピー vs 親リビジョンの差分（`hg diff`）。"""
    out = _run_hg(["diff"], repo_path=repo_path)
    return out if out is not None else ""


def get_hg_info(
    repo_path: str | None = None,
    include_diff: bool = True,
) -> dict[str, Any]:
    """
    ブランチ・リビジョン・タグ・dirty 状態・任意で diff をまとめて返す。

    戻り値のキー: ``branch``, ``revision``, ``revision_full``, ``rev``,
    ``tags``, ``dirty``, ``diff``
    """
    branch = get_hg_branch(repo_path=repo_path)
    revision = get_hg_revision(repo_path=repo_path, full=False)
    revision_full = get_hg_revision(repo_path=repo_path, full=True)
    rev = get_hg_rev_number(repo_path=repo_path)
    tags = get_hg_tags(repo_path=repo_path)
    dirty = bool(revision and revision.endswith("+"))
    diff = get_hg_diff(repo_path=repo_path) if include_diff else ""

    return {
        "branch": branch,
        "revision": revision,
        "revision_full": revision_full,
        "rev": rev,
        "tags": tags,
        "dirty": dirty,
        "diff": diff,
    }
