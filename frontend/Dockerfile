FROM nginx:alpine
RUN ["apk", "add", "--update", "nodejs", "nodejs-npm"]
COPY . /srv
WORKDIR /srv
EXPOSE 3000
RUN ["npm", "install"]
RUN ["npm", "run", "build"]
CMD ["npm", "run", "start"]
