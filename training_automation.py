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
        
        git_push_command = f"git push https://{token}:@github.com/MauriceDroll/AIPisAwesome master"
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
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet2_1ResBlock --epochs 1 --description training_GRConvNet_Basic_with_only_1_ResBlock",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet2_1ResBlock --epochs 1 --description just_a_test",
    # Weitere Befehle können hier hinzugefügt werden
    # "python another_script.py --option value"
]

commit_messages = [
    "TEST FOR AUTOMATION PURPOSES",
    "TEST FOR TWO AUTO COMMITS",
    # Weitere Commit-Nachrichten können hier hinzugefügt werden
    # "Automated commit: another_script"

]

# Iteriere durch die Liste der Python-Befehle und führe jeden aus
for python_command in python_commands:
    for commit_message in commit_messages:
        execute_command(python_command)
        run_git_commands(commit_message)
        print(f"Executed command: Befehl 1 ausgeführt")
        continue


