import subprocess


def run_cg2all(cg2all_args):
    proc = subprocess.run(cg2all_args,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise subprocess.SubprocessError(
            f"Error when running cg2all:\n{proc.stderr.decode('utf-8')}")