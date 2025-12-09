import os

def list_files(startpath):
    # Lista folderów do ignorowania (żeby nie robić bałaganu)
    ignore = {'.git', '.idea', '__pycache__', 'venv', '.venv', 'env', '.vscode'}
    
    for root, dirs, files in os.walk(startpath):
        # Filtrowanie folderów ignorowanych
        dirs[:] = [d for d in dirs if d not in ignore]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            # Ignoruj pliki .pyc i inne śmieci
            if not f.endswith(('.pyc', '.pyd')):
                print(f'{subindent}{f}')

if __name__ == '__main__':
    print("STRUKTURA PROJEKTU:\n")
    list_files('.')