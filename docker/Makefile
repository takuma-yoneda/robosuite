SHELL := /bin/bash
# .PHONY: all clean build test mount_shell shell upload check-env

DOCKER_NAME := takumaynd/robosuite
DOCKER := $(shell type -p nvidia-docker || echo docker)

default: build
build:
	docker build -t $(DOCKER_NAME) -f Dockerfile ..

push:
	docker push $(DOCKER_NAME)