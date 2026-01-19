import subprocess
import datetime

def get_git_info():
    """Return a dict with branch name, commit hash, and commit date."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        commit_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        commit_date = subprocess.check_output(
            ["git", "show", "-s", "--format=%ci", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        commit_message = subprocess.check_output(
            ["git", "show", "-s", "--format=%s", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        return {
            "git_branch":       branch,
            "git_commit_id":    commit_id,
            "git_commit_date":  commit_date,
            "commit_message":   commit_message,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

    except Exception as e:
        # Happens if not in a git repo or on a detached head
        return {"git_branch": None, "git_commit": None, "git_commit_date": None, "error": str(e)}
