.PHONY: typehint
typehint:
	mypy --ignore-missing-imports src

.PHONY: ptest
ptest:
	pytest -n auto -vv --cov=src\
		tests --cov-report term-missing -s

.PHONY: test
test:
	pytest -vv --cov=src\
		tests --cov-report term-missing -s

.PHONY: nrtest
nrtest:
	pytest -vv --cov=src\
		tests --cov-report term-missing -s -m "not regression"

.PHONY: format
format:
	black -l 79 src/*.py tests/*.py

.PHONY: lint
lint:
	pylint --disable=R,C src/*.py tests/*.py

.PHONY: qtconsole
qtconsole:
	jupyter qtconsole &

.PHONY: jupyter
jupyter:
	jupyter notebook &

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -rf
	find . -type d -name __pycache__ | xargs rm -rf

.PHONY: checklist
checklist: lint typehint test

.PHONY: all
all: install lint typehint test format
