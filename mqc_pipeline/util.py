import subprocess

def check_gpu():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        print("GPU is available:\n", output)
    except subprocess.CalledProcessError:
        print("No GPU available or NVIDIA drivers not installed")


