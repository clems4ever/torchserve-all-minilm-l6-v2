init:
	pip install -r requirements.txt

start-server:
	bash ./scripts/start-torchserve.sh

stop-server:
	bash ./scripts/stop-torchserve.sh

.PHONY: init test