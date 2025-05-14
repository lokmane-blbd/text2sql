#!/bin/bash


# Step 1: Start uvicorn in the background (silent)
echo "🚀 Starting FastAPI server on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
UVICORN_PID=$!


# Step 2: Start ngrok in the background
echo "🌐 Launching ngrok..."
ngrok http 8000 > ngrok.log &
NGROK_PID=$!

# Step 3: Wait for ngrok to initialize
echo "⏳ Waiting for ngrok to initialize..."
sleep 5

# Step 4: Get public ngrok URL
NGROK_URL=$(curl --silent http://localhost:4040/api/tunnels \
  | grep -oE 'https://[a-z0-9\-]+\.ngrok-free\.app' \
  | head -n 1)

if [ -z "$NGROK_URL" ]; then
  echo "❌ Failed to get ngrok URL"
  kill $UVICORN_PID
  kill $NGROK_PID
  exit 1
fi

echo "✅ ngrok tunnel is up: $NGROK_URL"

# Step 5: Start question loop
while true; do
  echo ""
  read -p "❓ Enter your question (or CTRL+C to exit): " USER_QUESTION

  if [ -z "$USER_QUESTION" ]; then
    echo "⚠️  Empty question. Try again."
    continue
  fi

  curl -s -X POST "$NGROK_URL/query" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$USER_QUESTION\"}"
  echo ""
done

# Cleanup (runs only on manual stop)
trap "echo '🧹 Shutting down...'; kill $UVICORN_PID; kill $NGROK_PID; exit" INT
