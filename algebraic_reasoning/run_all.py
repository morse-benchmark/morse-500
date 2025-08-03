import os
import subprocess
import sys


def run_all_sh_scripts(base_folder):
    subdirs = [
        os.path.join(base_folder, name)
        for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name))
    ]

    for subdir in subdirs:
        run_script = os.path.join(subdir, "run.sh")
        if os.path.isfile(run_script):
            print(f"Running {run_script}...")
            try:
                result = subprocess.run(
                    ["bash", run_script],
                    cwd=subdir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(result.stdout.decode())
            except subprocess.CalledProcessError as e:
                print(f"Error running {run_script}:\n{e.stderr.decode()}")
        else:
            print(f"No executable run.sh found in {subdir}")


if __name__ == "__main__":
    base_dir = os.path.abspath("./")
    run_all_sh_scripts(base_dir)
