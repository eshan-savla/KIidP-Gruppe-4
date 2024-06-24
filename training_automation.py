import os
import subprocess

def run_git_commands(commit_message):
    try:
        # Änderungen zu Git hinzufügen
        git_add_command = "git add ."
        execute_command(git_add_command)
        
        # Änderungen committen
        git_commit_command = f"git commit -m '{commit_message}'"
        execute_command(git_commit_command)
        
        # Änderungen pushen
        # Hole den Token aus den Umgebungsvariablen
        username = os.getenv('GIT_USERNAME')
        token = os.getenv('GIT_TOKEN')
        
        git_push_command = f"git push https://{token}:@github.com/eshan-savla/KIidP-Gruppe-4 master"
        execute_command(git_push_command)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
               
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")


# Liste der Python-Befehle
python_commands = [                                                                        
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_1mdaf_5residual --epochs 50 --description training_GRConvNet_single_mdaf_and_single_rfb",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_rfb --epochs 50 --description  changed rfb from 1 to 5 residual blocks (as orginal in gr convnet) ",



    # Weitere Befehle können hier hinzugefügt werden
    # "python another_script.py --option value"
]

commit_messages = [
    "Commit zu: changed grconvnet3_mdaf_single_single_rfb.py to grconvnet3_1rfb_1mdaf_5residual.py",
    "Commit zu:  changed rfb from 1 to 5 residual blocks (as orginal in gr convnet) ",

    

    # Weitere Commit-Nachrichten können hier hinzugefügt werden
    # "Automated commit: another_script"

]

# Iteriere durch die Liste der Python-Befehle und führe jeden aus
for python_command in python_commands:
    for commit_message in commit_messages:
        execute_command(python_command)
        run_git_commands(commit_message)
        print(f"Executed command: Befehl ausgeführt")
        continue


