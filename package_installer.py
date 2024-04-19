import subprocess

def install_requirements():
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt", "--user"])
        print("Requirements are Installed")
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing Requirements: {e}")   

if __name__ == "__main__":
    install_requirements()
    