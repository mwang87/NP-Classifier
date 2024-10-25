build:
	docker build --platform linux/amd64 -t npclassifier . --no-cache

bash:
	docker run --platform=linux/amd64 -it --rm npclassifier /bin/bash

server-compose-build-nocache:
	docker-compose --compatibility build --no-cache

server-compose-interactive:
	docker-compose --compatibility build
	docker-compose --compatibility up

server-compose:
	docker-compose --compatibility build
	docker-compose --compatibility up -d

server-compose-production:
	docker-compose --compatibility build
	docker-compose --compatibility up -d

attach:
	docker exec -i -t npclassifier-dash /bin/bash
