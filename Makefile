
.PHONY: format lint test nb

format:
	black .

lint:
	ruff check .
	black --check .

test:
	pytest -q

nb:
	papermill notebooks/EDA.ipynb notebooks/EDA_out.ipynb
	jupyter nbconvert --to html notebooks/EDA_out.ipynb
