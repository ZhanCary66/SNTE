"""Scheduling, shell safety, and collector preflight tests with synthetic JSON."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

torch.set_num_threads(1)

from dataset import DATASETS
from run_paper import (
    ROOT, HEADS, VARIANTS, Run, build_runs, canonical_run, logical_table_runs, run_aliases,
)
from test_results_io import result_for, write_run


def cli(script, *arguments, cwd=None):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                          cwd=cwd, env=environment, text=True, capture_output=True, timeout=30)


class SchedulingTests(unittest.TestCase):
    def test_logical_189_references_are_174_unique_runs(self):
        logical = logical_table_runs("all", list(DATASETS))
        runs = build_runs("all", list(DATASETS))
        self.assertEqual(len(logical), 189)
        self.assertEqual(len(runs), 174)
        self.assertEqual(len(set(runs)), 174)
        self.assertEqual(set(runs), {canonical_run(run) for run in logical})
        self.assertEqual(sum(run.table == "table1" for run in runs), 90)
        self.assertEqual(sum(run.table == "table2" for run in runs), 30)
        self.assertEqual(sum(run.table == "table3" for run in runs), 54)
        self.assertEqual(len(HEADS), 3)
        self.assertEqual(len(VARIANTS), 6)

    def test_table2_only_includes_15_reused_table1_runs(self):
        runs = build_runs("2", list(DATASETS))
        self.assertEqual(len(runs), 45)
        self.assertEqual(sum(run.table == "table1" for run in runs), 15)
        for reference in logical_table_runs("2", list(DATASETS)):
            self.assertIn(canonical_run(reference), runs)
        canonical = next(run for run in runs if run.table == "table1")
        alias = run_aliases(canonical)[0]
        self.assertEqual(canonical_run(alias), canonical)
        self.assertEqual(alias.command(), canonical.command())

    def test_invalid_or_duplicate_datasets_rejected_not_filtered(self):
        for datasets in ([], ["PKUEEG", "PKUEEG"], ["typo"], ["PKUEEG", "typo"], [""]):
            with self.assertRaises(ValueError):
                build_runs("all", datasets)
        with self.assertRaises(ValueError):
            build_runs("4", ["PKUEEG"])
        self.assertEqual({run.dataset for run in build_runs("1", ["SEM4Lang", "PKUEEG"])},
                         {"SEM4Lang", "PKUEEG"})

    def test_absolute_main_output_and_data_paths_and_shell_quoting(self):
        run = build_runs("2", ["PKUEEG"])[0]
        python = "/space dir/python's executable"
        output = Path("/result dir/with ' quote")
        data = Path("/data dir/$(not-a-command)")
        command = run.command(python, output, data)
        self.assertEqual(command[2], str(ROOT / "main.py"))
        self.assertEqual(command[command.index("--data-root") + 1], str(data))
        self.assertEqual(command[command.index("--output") + 1],
                         str(output / "table1/snte_split101/PKUEEG_seed2.json"))
        self.assertEqual(shlex.split(run.shell(python, output, data)), command)
        slurm = run.slurm(python, output, data, partition="custom-gpu")
        self.assertIn("#SBATCH --partition=custom-gpu", slurm)
        self.assertIn(run.shell(python, output, data), slurm)
        self.assertIn("set -euo pipefail", slurm)
        self.assertNotIn("sbatch ", slurm)
        for partition in ("gpu\n#SBATCH --time=1", "gpu;touch", ""):
            with self.assertRaises(ValueError):
                run.slurm(partition=partition)

    def test_cli_works_from_unrelated_cwd_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="cwd space ") as directory:
            base = Path(directory)
            emitted = cli("run_paper.py", "--table", "2", "--datasets", "PKUEEG",
                          "--python", "/my interpreter/python", "--output-root", "result dir",
                          "--data-root", "data dir", cwd=base)
            self.assertEqual(emitted.returncode, 0, emitted.stderr)
            commands = [shlex.split(line) for line in emitted.stdout.splitlines()]
            self.assertEqual(len(commands), 15)
            self.assertTrue(all(command[2] == str(ROOT / "main.py") for command in commands))
            self.assertEqual(commands[0][commands[0].index("--data-root") + 1], str(base / "data dir"))
            output = commands[0][commands[0].index("--output") + 1]
            self.assertTrue(output.startswith(str(base / "result dir")))
            for datasets in ("PKUEEG,PKUEEG", "PKUEEG,typo", ""):
                bad = cli("run_paper.py", "--datasets", datasets, cwd=base)
                self.assertNotEqual(bad.returncode, 0)
                self.assertEqual(bad.stdout, "")

    def test_slurm_never_overwrites_or_submits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ("--table", "3", "--datasets", "SEM4Lang", "--emit", "slurm",
                    "--slurm-dir", root, "--partition", "testing-gpu")
            first = cli("run_paper.py", *args)
            self.assertEqual(first.returncode, 0, first.stderr)
            scripts = list(root.rglob("*.slurm"))
            self.assertEqual(len(scripts), 18)
            self.assertTrue(all("#SBATCH --partition=testing-gpu" in path.read_text() for path in scripts))
            scripts[0].write_text("preserve this content")
            second = cli("run_paper.py", *args)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)
            self.assertEqual(scripts[0].read_text(), "preserve this content")
            self.assertEqual(len(list(root.rglob("*.slurm"))), 18)


class CollectorTests(unittest.TestCase):
    def test_missing_official_results_nonzero_no_tables_partial_coverage_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = build_runs("1", ["PKUEEG"])[0]
            write_run(root, run)
            official = cli("collect_results.py", "--results", root, "--table", "1", "--datasets", "PKUEEG")
            self.assertNotEqual(official.returncode, 0)
            self.assertEqual(official.stdout, "")
            self.assertIn("missing 29", official.stderr)
            partial = cli("collect_results.py", "--results", root, "--table", "1", "--datasets", "PKUEEG", "--partial")
            self.assertEqual(partial.returncode, 0, partial.stderr)
            self.assertIn("1/5", partial.stdout)
            self.assertIn("0/5", partial.stdout)
            self.assertNotIn("mean ±", partial.stdout)
            self.assertNotIn("±", partial.stdout)
            self.assertNotIn("0.30", partial.stdout)

    def test_partial_invalid_result_is_always_error_before_any_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = build_runs("1", ["PKUEEG"])[-1]
            result = result_for(run)
            result["integration"] = True
            write_run(root, run, result)
            output = cli("collect_results.py", "--results", root, "--table", "all", "--datasets", "PKUEEG", "--partial")
            self.assertNotEqual(output.returncode, 0)
            self.assertEqual(output.stdout, "")
            self.assertIn("integration", output.stderr)

    def test_full_tables_reuse_canonical_and_historical_alias_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, run in enumerate(build_runs("all", ["PKUEEG"])):
                aliases = run_aliases(run)
                write_run(root, aliases[0] if aliases and index % 2 else run, result_for(run))
            output = cli("collect_results.py", "--results", root, "--table", "all", "--datasets", "PKUEEG")
            self.assertEqual(output.returncode, 0, output.stderr)
            self.assertIn("Table 1", output.stdout)
            self.assertIn("Table 2", output.stdout)
            self.assertIn("Table 3", output.stdout)
            self.assertIn("mean ± std over 5", output.stdout)
            self.assertIn("±", output.stdout)
            self.assertNotIn("SparKULee", output.stdout)
            partial = cli("collect_results.py", "--results", root, "--table", "2", "--datasets", "PKUEEG", "--partial")
            self.assertEqual(partial.returncode, 0, partial.stderr)
            self.assertEqual(partial.stdout.count("5/5"), 3)
            self.assertNotIn("Table 1", partial.stdout)

    def test_all_selected_tables_preflight_and_alias_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = build_runs("all", ["SEM4Lang"])
            for run in runs:
                write_run(root, run)
            damaged = result_for(runs[-1])
            damaged["learning_rate"] = 0.8
            write_run(root, runs[-1], damaged)
            output = cli("collect_results.py", "--results", root, "--datasets", "SEM4Lang")
            self.assertNotEqual(output.returncode, 0)
            self.assertEqual(output.stdout, "")
            write_run(root, runs[-1])
            alias = run_aliases(runs[0])[0]
            conflict = result_for(runs[0])
            conflict["test"]["loss"] = 2.0
            write_run(root, alias, conflict)
            output = cli("collect_results.py", "--results", root, "--table", "2", "--datasets", "SEM4Lang", "--partial")
            self.assertNotEqual(output.returncode, 0)
            self.assertEqual(output.stdout, "")
            self.assertIn("conflicting", output.stderr)

    def test_collector_legacy_subject_map_is_explicit_and_external(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = {}
            for run in build_runs("2", ["SEM4Lang"]):
                result = result_for(run)
                result.pop("schema_version")
                result.pop("candidate_seeds")
                entry = {"split_unit_names": result.pop("split_unit_names"),
                         "architecture": result.pop("architecture")}
                path = write_run(root, run, result)
                mapping[path.relative_to(root).as_posix()] = entry
            path = root / "explicit-map.json"
            path.write_text(json.dumps(mapping))
            base = ("--results", root, "--table", "2", "--datasets", "SEM4Lang")
            missing_metadata = cli("collect_results.py", *base)
            self.assertNotEqual(missing_metadata.returncode, 0)
            self.assertEqual(missing_metadata.stdout, "")
            official = cli("collect_results.py", *base, "--subject-map", path)
            self.assertEqual(official.returncode, 0, official.stderr)
            for key in mapping:
                result = json.loads((root / key).read_text())
                self.assertNotIn("architecture", result)
                self.assertNotIn("split_unit_names", result)


if __name__ == "__main__":
    unittest.main()
