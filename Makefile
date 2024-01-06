.PHONY: help build start stop clean

# Display this help text
help:
	@echo "Available commands:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' |  sed -e 's/^/ /'

## build: Build the application
build:
	@docker compose build
	@echo "Application built"

## start: Start the application
start:
	@docker compose up --detach
	@echo "Application started"

## stop: Stop the application
stop:
	@docker compose down
	@echo "Application stopped"

## clean: Stop the application and remove all volumes
clean:
	@docker compose down -v --rmi all --remove-orphans
	@docker volume prune -f
