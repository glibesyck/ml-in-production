import subprocess

def run_nni_experiment():
    subprocess.run(["nnictl", "create", "--config", "nni_config.yml"], check=True)
    return True

if __name__ == "__main__":   
    run_nni_experiment()