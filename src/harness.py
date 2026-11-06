"""Minimal agent harness used in Part B of the notebook.

It reproduces the *protocol* of the study in a portable way (no machine-specific locks):
  prepare_run_folder()  - one isolated folder per run, with its own virtual environment and a READ-ONLY copy of train_val.db
  build_agent_command() - the Hermes command line that starts an agent on the brief (only needed if you run an agent yourself)
  run_agent()           - start the agent with a hard wall-clock limit
  scan_commands()       - a small rule auditor: flags commands that break the brief's rules
The test data (data/test.db) is NEVER copied into a run folder; only the evaluator reads it.
"""
import hashlib, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "agent_harness"
BUDGETS = json.loads((HARNESS / "budgets.json").read_text())
PLAN = json.loads((HARNESS / "run_plan.json").read_text())


def brief_is_approved(brief=HARNESS / "brief_unguided.md"):
    """The exact brief text is frozen: its SHA-256 must match the approved hash, so every run sees byte-identical instructions."""
    return (HARNESS / "approved_brief.sha256").read_text().split()[0] == hashlib.sha256(Path(brief).read_bytes()).hexdigest()


def prepare_run_folder(run_id, base=ROOT / "runs", install_requirements=True):
    """Create runs/<run_id>/work with its own .venv (needs `uv`) and a read-only train_val.db. Refuses to overwrite an existing run."""
    run = Path(base) / run_id; work = run / "work"
    if run.exists(): raise FileExistsError(f"{run} already exists: runs are never overwritten")
    work.mkdir(parents=True); (run / "logs").mkdir()
    shutil.copy(ROOT / "data" / "train_val.db", work / "train_val.db"); (work / "train_val.db").chmod(0o444)
    if install_requirements:
        subprocess.run(["uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", str(work / ".venv")], check=True, capture_output=True)
        subprocess.run(["uv", "pip", "install", "--python", str(work / ".venv" / "bin" / "python"), "-r", str(ROOT / "requirements.txt")], check=True, capture_output=True)
    return work


def build_agent_command(row, work, profile="YOUR_HERMES_PROFILE"):
    """Hermes command line for one row of run_plan.json (model id, reasoning effort, turn cap, brief as the single user message)."""
    level = {"think-off": "none", "think-on": "high"}.get(row["setting"], row["setting"])
    cmd = ["hermes", "-p", profile, "chat", "-Q", "-m", row["model_id"], "--provider", row["provider"], "--reasoning", level,
           "--max-turns", str(BUDGETS["max_turns"]), "--in", str(work), "--query-file", str(HARNESS / "brief_unguided.md")]
    return cmd


def run_agent(cmd, work, wall_minutes=BUDGETS["wall_clock_min"]):
    """Run the agent with a hard wall-clock limit (the whole process group is killed at the limit). Returns (exit_code, minutes, timed_out)."""
    t0 = time.time(); out = open(Path(work).parent / "logs" / "agent_stdout.txt", "w")
    proc = subprocess.Popen(cmd, cwd=str(work), stdout=out, stderr=subprocess.STDOUT, start_new_session=True); timed_out = False
    while proc.poll() is None:
        if time.time() - t0 > wall_minutes * 60:
            timed_out = True; os.killpg(proc.pid, signal.SIGTERM); time.sleep(10)
            if proc.poll() is None: os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(5)
    proc.wait(); out.close(); return proc.returncode, round((time.time() - t0) / 60, 2), timed_out


# --- rule auditor (simplified): regular expressions over the agent's terminal commands ------------------------------------------------
RULES = {
    "touches_test_data":   r"test_holdout|test\.db",
    "starts_another_model": r"\bhermes\b|\bollama\b|api\.openai|api\.anthropic|openrouter",
    "inline_interpreter":  r"\bpython3?\s+-c\b|\bpython3?\s*<<",
    "modifies_train_val":  r"(rm|mv|chmod|sqlite3.*(DROP|DELETE|UPDATE|INSERT))\s.*train_val\.db",
    "installs_globally":   r"\bpip3?\s+install\b(?!.*--python)|\bconda\s+install\b|\bbrew\s+install\b",
}


def scan_commands(commands):
    """Return [(command, rule_name)] for every command that matches a rule. A confirmed violation makes a run invalid in the study."""
    return [(c, name) for c in commands for name, pat in RULES.items() if re.search(pat, c, re.I)]
