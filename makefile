setup:
	python3 -m venv ./venv && \
	. ./venv/bin/activate && \
	pip install -r requirements.txt \
	mkdir -p ./output \
	mkdir -p ./results \
	&& echo "Setup complete. Run 'make run' to start the program."

run: 
	. ./venv/bin/activate && \
	python3 main.py