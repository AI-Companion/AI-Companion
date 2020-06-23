FROM nginx:alpine
RUN ["apk", "add", "--update", "nodejs", "nodejs-npm"]
COPY . /srv
WORKDIR /srv
RUN ["npm", "install"]
RUN ["npx", "webpack"]
RUN ["cp", "dist/index.html", "/usr/share/nginx/html"]
RUN ["cp", "dist/main.js", "/usr/share/nginx/html"]

