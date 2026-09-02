# Installation

Use Python 3.11. From the repository root, install the project environment:

```bash
poetry install
poetry run python -m ipykernel install --user \
  --name weeviltrak-py311 \
  --display-name "WeevilTrak (.venv Python 3.11)"
```

For a minimal local inference environment, install the packages listed in
`requirements/requirements-client.txt`. The full repository is required for
the supplied helper because it uses the reviewed canonical event and learned
termination implementations under `app/`.

Do not put AWS, Redshift, or other credentials in notebooks or configuration
files. Local pickle inference does not require those credentials.
