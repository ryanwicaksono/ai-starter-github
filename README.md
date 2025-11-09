# AI Starter (github.dev + GitHub Actions)

This is a minimal, **tablet-friendly** starter repo to learn AI/ML entirely from the browser:
- Edit code in **github.dev** (VS Code in the browser).
- Let **GitHub Actions** run tests + execute a notebook and upload an **HTML report** as an artifact.
- No local install required.

## What's inside
- `src/ai_starter/logistic.py`: Logistic Regression **from scratch** (NumPy).
- `tests/test_logistic.py`: sanity tests using synthetic data.
- `notebooks/EDA.ipynb`: a small notebook using the **breast cancer** dataset (from scikit-learn) to compare our scratch model vs scikit-learn.
- CI pipeline (in `.github/workflows/ci.yml`) that:
  - lints (ruff, black)
  - runs tests (pytest)
  - executes the notebook with **papermill**
  - converts it to HTML and uploads as an artifact

## How to use (from your tablet)
1. Create a new empty repo on GitHub (public or private).
2. Upload these files (or just upload the ZIP) and commit.
3. Open the repo in the browser editor: press `.` in the repo page (github.dev).
4. Go to the **Actions** tab and enable GitHub Actions if prompted.
5. Push a change (e.g., edit `README.md`) to trigger the pipeline.
6. After the run finishes, open the **Artifacts** of the workflow and download `EDA_out.html` to view the executed notebook.

### Optional (Codespaces)
If you use GitHub **Codespaces**, a dev container config is included so you can run code interactively with a terminal.

## Makefile (optional when using Codespaces)
- `make format` — format with black
- `make lint` — ruff + black check
- `make test` — run pytest
- `make nb` — execute the notebook locally (requires papermill + jupyter)

Happy building!
