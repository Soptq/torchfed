install:
	pip install -r requirements.txt

lint-fix:
	autopep8 --in-place --aggressive --aggressive --recursive ./torchfed
	autopep8 --in-place --aggressive --aggressive --recursive ./example

freeze:
	pip freeze > requirements.txt