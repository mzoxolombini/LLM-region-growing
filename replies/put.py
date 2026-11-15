"""put.py RUN GEN K file.py -> writes the reply file for prompt gGG_kKK."""
import sys
run, g, k, f = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
open(f"{run}/manual/g{g:02d}_k{k:02d}.reply.md", "w").write("```python\n" + open(f).read().strip() + "\n```\n")
