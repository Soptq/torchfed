install:
	pip install -r requirements.txt

lint-fix:
	autopep8 --in-place --aggressive --aggressive --recursive ./torchfed
	autopep8 --in-place --aggressive --aggressive --recursive ./example
	autopep8 --in-place --aggressive --aggressive --recursive ./tests

freeze:
	pip freeze > requirements.txt

clean:
	rm -rf ./example/centralized/fedavg/.aim
	rm -rf ./example/centralized/fedavg/logs
	rm -rf ./example/decentralized/fedavg/.aim
	rm -rf ./example/decentralized/fedavg/logs