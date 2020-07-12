FROM nginx:alpine
RUN ["apk", "add", "--update", "nodejs", "nodejs-npm"]
COPY . /srv
WORKDIR /srv
RUN ["npm", "install"]
RUN ["npm", "run", "build"]
CMD ["npm", "run", "start"]
