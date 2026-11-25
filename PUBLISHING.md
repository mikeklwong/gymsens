# Publishing and maintaining gymsens

Repository: https://github.com/mikeklwong/gymsens

The project is intended to stay **private** during its research/pilot stage. Review model and dependency license obligations, camera privacy policies, and system validation before making it public or deploying commercially.

## Clone and develop

```bash
git clone https://github.com/mikeklwong/gymsens.git
cd gymsens
python -m venv .venv
# activate the virtual environment
pip install -r requirements-dev.txt
pytest -q
```

Do not commit `.env`, camera passwords, recordings, real-world calibration data, SQLite observations, or trained model weights. Rotate secrets promptly if one is accidentally exposed.

## Push updates

```bash
git add .
git diff --cached --check
git diff --cached
git commit -m "Describe your change"
git push origin main
```

The included GitHub Actions CI validates syntax and runs unit tests on push and pull requests. A passing CI run is **not** proof of field accuracy; real-camera validation and human-reviewed ground truth are still required.
