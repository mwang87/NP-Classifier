build:
	docker build -t npclassifier . 

bash:
	docker run -it --rm npclassifier /bin/bash

server-compose-build-nocache:
	docker-compose build --no-cache

server-compose-interactive:
	docker-compose build
	docker-compose up

server-compose:
	docker-compose build
	docker-compose up -d

server-compose-production:
	docker-compose build
	docker-compose up -d

attach:
	docker exec -i -t npclassifier-dash /bin/bash
