#!/bin/bash
#
# Deploy Zen AI Models
#

set -e

echo "🚀 Deploying Zen AI Models..."

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Remove old traefik container if it exists
docker stop traefik 2>/dev/null || true
docker rm traefik 2>/dev/null || true

# Build images
echo "🏗️ Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 10

# Check status
echo ""
echo "📊 Service Status:"
docker-compose ps

# Test API
echo ""
echo "🧪 Testing API..."
curl -s http://localhost:8000/health | jq . || echo "API not ready yet"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📍 Access points:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Traefik Dashboard: http://localhost:8080"
echo "  - Ollama: http://localhost:11434"
echo ""
echo "📝 Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - Scale API: docker-compose up -d --scale zen-api=3"