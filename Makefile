install:
	pip install -r requirements.txt

fmt:
	autopep8 --in-place --aggressive --aggressive --recursive ./torchfed
	autopep8 --in-place --aggressive --aggressive --recursive ./example
	autopep8 --in-place --aggressive --aggressive --recursive ./tests

freeze:
	pip freeze > requirements.txt

clean:
	find ./example -name ".aim" -type d -prune -exec rm -rf '{}' +
	find ./example -name "logs" -type d -prune -exec rm -rf '{}' +
