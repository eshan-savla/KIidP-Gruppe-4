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
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet --epochs 50 --description training_grovnnet0_no_channelsize",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet2 --epochs 50 --description training_grovnnet_with_Parameter",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet2_MaxPooling --epochs 50 --description training_grovnnet_with_MaxPooling",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet2_1ResBlock --epochs 50 --description training_grovnnet_with_1ResBlock",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1mdaf --epochs 50 --description training_grovnnet3_with_1MDAF",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb --epochs 50 --description training_grovnnet3_with_1RFB",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_1mdaf_5residual --epochs 50 --description training_grovnnet3_with_5RFB",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_1mdaf_1residual --epochs 50 --description training_cornell_1rfb_1mdfa_1residual",


]

commit_messages = [
    "Commit zu: training_grovnnet0_no_channelsize",
    "Commit zu: training_grovnnet_with_Parameter ",
    "Commit zu: training_grovnnet_with_MaxPooling ",
    "Commit zu: training_grovnnet_with_1ResBlock ",
    "Commit zu: training_grovnnet3_with_1MDAF ",
    "Commit zu: training_grovnnet3_with_1RFB ",
   

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


