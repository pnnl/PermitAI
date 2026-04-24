SHELL := /bin/bash
.PHONY: all clean python-venv clean-venv


all: python-venv
	@echo "Creating the python virtual environment"
	module purge ;\
	module load gcc/12.2.0 ;\
	module load python/3.11.5 ;\
	module load sqlite/3.36 ;\
	python3 -m venv maple_venv ;\
	source maple_venv/bin/activate ;\
	pip3 install -r requirements.txt

clean: clean-venv
	@echo "Removing the python virtual environment"
	rm -rf maple_venv
	@echo "Finished removing the python virtual environment"
