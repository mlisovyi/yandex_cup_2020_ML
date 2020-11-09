# You can run interactor.py to test solution with this program
#
# Use this command to run testing:
# python3 testing_tool.py python3 interact.py input output -- <solution>
#
# * input - input file (`01` in the archive)
# * output - output file (choose any)
# * <solution> - your solution command line (e.g., `python3 solution.py`)
#
# For example:
# python3 testing_tool.py python3 interact.py 01 output -- python3 solution.py

import sys, subprocess, threading

class SubprocessThread(threading.Thread):
    def __init__(
        self,
        args,
        stdin_pipe=subprocess.PIPE,
        stdout_pipe=subprocess.PIPE,
        stderr_prefix=None
    ):
        threading.Thread.__init__(self)
        self.stderr_prefix = stderr_prefix
        self.p = subprocess.Popen(
            args,
            stdin=stdin_pipe,
            stdout=stdout_pipe,
            stderr=subprocess.PIPE
        )

    def run(self):
        self.make_stderr(self.p.stderr)
        self.return_code = self.p.wait()

    def make_stderr(self, stream):
        to_append = True
        while True:
            chunk = stream.readline(1024)
            if chunk:
                chunk = chunk.decode("UTF-8")
                if to_append and self.stderr_prefix:
                    chunk = self.stderr_prefix + chunk
                    to_append = False
                sys.stderr.write(chunk)
                if chunk.endswith("\n"):
                    to_append = True
                sys.stderr.flush()
            else:
                return

def argument_parser():
    sep_index = sys.argv.index("--")
    interact_args = sys.argv[1:sep_index]
    solution_args = sys.argv[sep_index + 1:]
    return sep_index, interact_args, solution_args

sep_index, interact_args, solution_args = argument_parser()

t_solution = SubprocessThread(solution_args, stderr_prefix="  solution: ")
t_interact = SubprocessThread(
    interact_args,
    stdin_pipe=t_solution.p.stdout,
    stdout_pipe=t_solution.p.stdin,
    stderr_prefix="interact: ")
t_solution.start()
t_interact.start()
t_solution.join()
t_interact.join()

print("interact return: {}".format(t_interact.return_code))
print("solution return: {}".format(t_solution.return_code))
