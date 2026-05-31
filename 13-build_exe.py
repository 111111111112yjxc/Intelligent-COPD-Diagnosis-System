import PyInstaller.__main__
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
main_script = os.path.join(base_dir, "12-app.py")

if not os.path.exists(main_script):
    print(f"错误：找不到 {main_script}")
    sys.exit(1)

PyInstaller.__main__.run([
    main_script,
    '--onefile',
    '--console',
    '--name', 'COPD_Diagnosis',
    '--hidden-import', 'pydicom',
    '--hidden-import', 'scipy.ndimage',
    '--hidden-import', 'sklearn.metrics',
    '--hidden-import', 'tqdm',

    '--add-data', f'{os.path.join(base_dir, "best_resnet18_binary.pth")};.',
    '--add-data', f'{os.path.join(base_dir, "best_resnet18_multi.pth")};.',
])