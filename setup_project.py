import os

# Define the project structure
structure = {
    "folders": [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "models",
    ],
    "files": [
        "src/__init__.py",
        "src/preprocessing.py",
        "src/augmentation.py",
        "src/model.py",
        "src/inference.py",
        "app.py",
        "README.md",
        "requirements.txt",
        ".gitignore"
    ]
}

def create_structure():
    # Create Folders
    for folder in structure["folders"]:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created folder: {folder}")

    # Create Files
    for file in structure["files"]:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                pass # Create empty file
            print(f"✅ Created file: {file}")
        else:
            print(f"⚠️ File already exists: {file}")

if __name__ == "__main__":
    create_structure()
    print("\n🚀 Project structure is ready!")