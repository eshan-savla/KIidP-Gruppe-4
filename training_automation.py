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
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight_without_rfb --epochs 50 --description lightweight_basis_without_rfb",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight_without_maxpooling --epochs 50 --description lightweight_basis_without_maxpooling",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet2_1resblock --epochs 50 --description training_GRConvNet_Basic_with_only_1_ResBlock",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_mdaf_single_single_rfb --epochs 50 --description training_GRConvNet_single_mdaf_and_single_rfb",
    # Weitere Befehle können hier hinzugefügt werden
    # "python another_script.py --option value"
]

commit_messages = [
    "Lightweight without MaxPooling 4.4MIo Conv8 channel 128 ResBlock 3 Zeile 22",
    "Lightweight without RFB 4.4MIo Conv8 channel 128 ResBlock 3 Zeile 22",
    "training_GRConvNet_Basic_with_only_1_ResBlock",
    "training_GRConvNet_Basic_1_RFB_1_MDFA_1",
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


