import subprocess

def run(model, file):
    cmd = ["C:\\Users\\keith\\Desktop\\repos\\whisper.cpp\\main.exe"]
    cmd += ["--model", model]
    cmd += ["--file", file]
    cmd += ["-nt"]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Return the output and error messages
    return result.stdout, result.stderr