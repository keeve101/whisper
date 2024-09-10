import subprocess
from config import WHISPER_MAIN

def run(model, file):
    cmd = [WHISPER_MAIN]
    cmd += ["--model", model]
    cmd += ["--file", file]
    cmd += ["-nt"]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Return the output and error messages
    return result.stdout, result.stderr
