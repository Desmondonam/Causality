# Only used for local `docker compose up` full-stack dev. Production
# deployment target is Vercel (see README.md), which does not use this file.
FROM node:22-slim AS base

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
