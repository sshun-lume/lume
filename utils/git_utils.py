"""
Utility functions for getting git repository information.
"""

import os
import subprocess
from typing import Optional, Tuple


def get_git_info() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get git branch name, commit hash, and diff output.

    Returns:
        Tuple of (branch_name, commit_hash, diff_output)
        Returns (None, None, None) if not in a git repository
    """
    try:
        # Change to the git repository directory
        repo_path = os.path.dirname(os.path.abspath(__file__))

        # Get branch name
        branch_process = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        branch_name = branch_process.stdout.strip()

        # Get commit hash
        hash_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hash = hash_process.stdout.strip()

        # Get diff output
        diff_process = subprocess.run(
            ["git", "diff"], cwd=repo_path, capture_output=True, text=True, check=True
        )
        diff_output = diff_process.stdout

        return branch_name, commit_hash, diff_output

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repository or git command failed
        return None, None, None


def get_git_branch() -> Optional[str]:
    """
    Get the current git branch name.

    Returns:
        Branch name or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_hash() -> Optional[str]:
    """
    Get the current git commit hash.

    Returns:
        Commit hash or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_diff() -> Optional[str]:
    """
    Get the git diff output.

    Returns:
        Diff output or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "diff"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


if __name__ == "__main__":
    # Example usage
    branch, commit_hash, diff = get_git_info()

    print(f"Branch: {branch}")
    print(f"Commit Hash: {commit_hash}")
    print(f"Diff:")
    print(diff)
