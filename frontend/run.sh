docker run -p 8000:80 -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro -d frontend:latest
