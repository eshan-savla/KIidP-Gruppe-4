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
    # DONE:   
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3 --epochs 50 --description 2_Orginal_GR-Convnet3_als_Referenz",                                                                  
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1residual --epochs 50 --description 2_Orginal_GR_Convnet3_mit_nur_einem_Residual_Block",
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1mdaf --epochs 50 --description Orginal_GR_Convnet3_zus_MDAF_Block_im_bottleneck_Einfluss_durch_MDAF_zu_testen ",
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb --epochs 50 --description 2_Orginal_GR_Convnet3_zus_RFB_im_bottleneck_um_Einfluss_durch_RFB_zu_testen  ",
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_1mdaf_5residual --epochs 50 --description 2_Orginal_GR_Convnet3_mit_allen_5_residual_Blocks_und_jeweils_einem_RFB_u_MDAF  ",
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_1mdaf_1residual --epochs 50 --description 2_Orginal_GR_Convnet3_mit_nur_einem_residual_Block_und_jeweils_einem_RFB_u_MDAF  ",
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_2maxpool --epochs 50 --description 2_Orginal_GR_Convnet3_mit_2_MaxPooling",
    #"python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_2mdaf --epochs 50 --description 2_Orginal_GR-Convnet3_mit_3_residual_Blocks_einem_RFB_2_MDAF_und_nur_2_ConfT2D",


    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight --epochs 50 --description Original_training_lightweight",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight_channelsize_128 --epochs 50 --description Original_training_lightweight_channelsize_128",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight_without_maxpooling --epochs 50 --description Original_training_lightweight_without_pooling",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight_without_rfb --epochs 50 --description Original_training_lightweight_without_rfb",
    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network lightweight_only_1_mdaf --epochs 50 --description Original_training_lightweight_with_only_1_mdaf",





]

#    "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3 --epochs 1 --description Orginal_GR-Convnet3_als_Referenz",
#     "python train_network.py --dataset cornell --dataset-path data/ --network grconvnet3_1mdaf --epochs 1 --description Orginal_GR_Convnet3_mit_2_MaxPooling",
# "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1mdaf --epochs 1 --description Orginal_GR_Convnet3_zus_MDAF_Block_im_bottleneck_Einfluss_durch_MDAF_zu_testen ",
# "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb --epochs 1 --description Orginal_GR_Convnet3_zus_RFB_im_bottleneck_um_Einfluss_durch_RFB_zu_testen  ",
# "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_1mdaf_5residual --epochs 1 --description Orginal_GR_Convnet3_mit_allen_5_residual_Blocks_und_jeweils_einem_RFB_u_MDAF  ",
# "python train_network.py --dataset cornell --dataset-path utils/data/cornell --network grconvnet3_1rfb_1mdaf_1residual --epochs 1 --description Orginal_GR_Convnet3_mit_nur_einem_residual_Block_und_jeweils_einem_RFB_u_MDAF  ",





commit_messages = [
    # DONE:
    #"Commit zu: RUN_2_Orginal_GR-Convnet3_als_Referenz",
    #"Commit zu: RUN_2_Orginal_GR_Convnet3_mit_nur_einem_Residual_Block",
    #"Commit zu: RUN_2_Orginal_GR_Convnet3_zus_MDAF_Block_im_bottleneck_Einfluss_durch_MDAF_zu_testen",
    #"Commit zu: RUN_2_Orginal_GR_Convnet3_zus_RFB_im_bottleneck_um_Einfluss_durch_RFB_zu_testen",
    #"Commit zu: RUN_2_Orginal_GR_Convnet3_mit_allen_5_residual_Blocks_und_jeweils_einem_RFB_u_MDAF",
    #"Commit zu: RUN_2_Orginal_GR_Convnet3_mit_nur_einem_residual_Block_und_jeweils_einem_RFB_u_MDAF",
    #"Commit zu: RUN_2_Orginal_GR_Convnet3_mit_2_MaxPooling",
    #"Commit zu: RUN_2_Convnet3_mit_3_residual_Blocks_einem_RFB_2_MDAF_und_nur_2_ConfT2D",

    "Commit zu: RUN_2_Original_training_lightweight",
    "Commit zu: RUN_2_Original_training_lightweight_channelsize_128",
    "Commit zu: RUN_2_Original_training_lightweight_without_pooling",
    "Commit zu: RUN_2_Original_training_lightweight_without_rfb",
    "Commit zu: RUN_2_Original_training_lightweight_with_only_1_mdaf",
    
    
    # Weitere Commit-Nachrichten können hier hinzugefügt werden
    # "Automated commit: another_script"

]

#     "Commit zu: Orginal_GR_Convnet3_mit_2_MaxPooling",
#     "Commit zu: Convnet3_mit_3_residual_Blocks_einem_RFB_2_MDAF_und_nur_2_ConfT2D",


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


