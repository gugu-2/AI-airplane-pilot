import zipfile
import os

def create_linux_compatible_zip():
    folders_to_zip = ['src', 'models']
    
    with zipfile.ZipFile('colab_training_package.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in folders_to_zip:
            for root, _, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Convert Windows backslashes to Linux forward slashes inside the ZIP metadata
                    arcname = file_path.replace('\\', '/')
                    zipf.write(file_path, arcname)
                    
create_linux_compatible_zip()
print("Created cross-platform zip successfully.")
