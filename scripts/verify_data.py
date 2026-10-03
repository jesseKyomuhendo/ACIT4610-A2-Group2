"""Check that data/orlib files match the official OR-Library hashes and parse correctly."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cflp_moea.data_loader import DATA_DIR, load_instance, load_optimal_values  # noqa: E402

ok = True
for line in (DATA_DIR / "SHA256SUMS").read_text().splitlines():
    digest, fname = line.split()
    actual = hashlib.sha256((DATA_DIR / fname).read_bytes()).hexdigest()
    status = "OK" if actual == digest else "MISMATCH"
    ok &= actual == digest
    print(f"{fname:12s} {status}")

opt = load_optimal_values()
for name in ["cap41", "cap42", "cap101", "cap102", "cap121", "cap122"]:
    inst = load_instance(name)
    print(f"{name:7s} m={inst.m:3d} n={inst.n:3d} total_demand={inst.total_demand:9.0f} "
          f"total_capacity={inst.capacity.sum():9.0f} known_opt(f1+f2)={opt[name]:.3f}")
sys.exit(0 if ok else 1)
