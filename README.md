# vgg_related

This folder contains VGG-based training, unlearning, and visualization scripts.

## Path Dependency Audit (2026-03-26)

Main findings:

- No hardcoded Windows absolute paths (for example, `D:\\...`) were found.
- Most scripts use `SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))`, which is portable across machines when run as files.
- Several scripts use `root='./data'` for CIFAR-10, which depends on the current working directory and may break when run from a different folder.
- `train_vgg_small.py` and `train_vgg_112.py` use multi-candidate dataset paths (`105_classes_pins_dataset`), so they rely on specific local folder layouts.

Risk levels:

- High: dataset candidate search tied to local directory structure.
- Medium: `root='./data'` CWD dependency.
- Low: output paths based on `SCRIPT_DIR/checkpoint` and `SCRIPT_DIR/figs`.

## Suggested Usage

Run scripts from this directory to avoid CWD issues:

```powershell
cd vgg_related
python train_cifar10.py
```

For stronger portability, replace `root='./data'` with:

```python
root=os.path.join(SCRIPT_DIR, "data")
```

## Notes

- `.gitignore` excludes model checkpoints and local datasets by default.
- If you need to share model files, prefer release assets or Git LFS.
