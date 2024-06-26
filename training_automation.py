import os
import subprocess
import datetime

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
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_with_MaxPooling --epochs 50 --description training_grovnnet3_with_MaxPooling",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight --epochs 50 --description training_lightweight_channel_32",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight_channelsize_128 --epochs 50 --description training_lightweight_channel_128",

]

commit_messages = [
    "Commit zu: training_grovnnet3_with_MaxPooling ",
    "training_lightweight_channel_32",
    "training_lightweight_channel_128"
    # Weitere Commit-Nachrichten können hier hinzugefügt werden
    # "Automated commit: another_script"

]

i =0
# Iteriere durch die Liste der Python-Befehle und führe jeden aus
for python_command in python_commands:
    print("Startzeit:", datetime.datetime.now())
    print(f"Execute command: {python_command}")
    execute_command(python_command)
    print(f"ExecuteD command: {python_command}")
    print("Endzeit:", datetime.datetime.now())

    print(f"Started command: " + commit_messages[i])
    run_git_commands(commit_messages[i])
    print(f"Executed command: " + commit_messages[i])
    i= i+1
    
    
    #for commit_message in commit_messages:
    #    run_git_commands(commit_message)
    #    print(f"Executed command: " + commit_message)
    #continue


