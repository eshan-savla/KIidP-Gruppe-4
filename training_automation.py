import os
import subprocess

def run_command(command,commit_message):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Änderungen zu Git hinzufügen
        git_add_command = "git add ."
        run_command(git_add_command)

        # Änderungen committen
        git_commit_command = "git commit -m 'Automated commit'"
        run_command(git_commit_command)

        # Änderungen pushen
        # Hole den Token aus den Umgebungsvariablen
        username = os.getenv('GIT_USERNAME')
        token = os.getenv('GIT_TOKEN')

        git_push_command = f"git push https://{username}:{token}@github.com/eshan-savla/KIidP-Gruppe-4.git main"
        
        run_command(git_push_command)
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

# Liste der Python-Befehle
python_commands = [
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet2_1ResBlock --epochs 50 --description training_GRConvNet_Basic_with_only_1_ResBlock",
    # Weitere Befehle können hier hinzugefügt werden
    # "python another_script.py --option value"
]

commit_messages = [
    "Automated commit: training_GRConvNet_Basic_with_only_1_ResBlock",
    # Weitere Commit-Nachrichten können hier hinzugefügt werden
    # "Automated commit: another_script"

]

# Iteriere durch die Liste der Python-Befehle und führe jeden aus
for python_command in python_commands:
    run_command(python_command)



