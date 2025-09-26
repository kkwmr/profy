FROM node:18-bullseye AS builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --legacy-peer-deps

# Copy sources and build React app
COPY . .
RUN CI=false npm run build

# Runtime image
FROM node:18-bullseye AS runner
WORKDIR /app

# Install only production deps for the server
COPY package*.json ./
RUN npm ci --omit=dev --legacy-peer-deps && npm cache clean --force

# Copy server and production build output from builder stage
COPY server.js ./server.js
COPY --from=builder /app/build ./build

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

CMD ["node", "server.js"]
