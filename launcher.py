import os, subprocess, sys, webbrowser, time, venv

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(APP_DIR, "auto_env")

def ensure_venv():
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)

def run(cmd):
    subprocess.check_call(cmd, shell=True)

def main():
    ensure_venv()

    # Path to python inside the auto venv
    py = os.path.join(VENV_DIR, "Scripts", "python.exe")

    # Install deps if missing
    run(f'"{py}" -m pip install streamlit sounddevice tensorflow numpy librosa soundfile')

    # Launch Streamlit app
    print("Starting Streamlit...")
    subprocess.Popen(
        f'"{py}" -m streamlit run "{APP_DIR}/app.py"',
        shell=True
    )

    # Wait and open browser
    time.sleep(5)
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    main()
