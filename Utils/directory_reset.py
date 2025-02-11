import os 
import shutil

if os.path.isdir("Data/Negatives"): 
    print("Resetting Data Folders!")
    shutil.rmtree("Data/Negatives")
    shutil.rmtree("Data/Parts")
    shutil.rmtree("Data/Positives")

    os.mkdir("Data/Negatives")
    os.mkdir("Data/Parts")
    os.mkdir("Data/Positives")

    open('Data/filtered_annotations.txt', 'w').close()

elif not os.path.isdir("Data/Negatives"): 
    print("Creating Data Folders!")
    os.mkdir("Data/Negatives")
    os.mkdir("Data/Parts")
    os.mkdir("Data/Positives")