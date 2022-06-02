install:
	git submodule update --init
	pip install -e deps/visdom
	pip install -r requirements.txt

lint-fix:
	autopep8 --in-place --aggressive --aggressive --recursive ./torchfed
	autopep8 --in-place --aggressive --aggressive --recursive ./example
	autopep8 --in-place --aggressive --aggressive --recursive ./tests

freeze:
	pip freeze > requirements.txt