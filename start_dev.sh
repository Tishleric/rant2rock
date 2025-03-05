#!/bin/bash

# Start the backend server
echo "Starting backend server..."
export API_MODE=true
python3 main.py &
BACKEND_PID=$!

# Wait for the backend to start
echo "Waiting for backend to start..."
sleep 3

# Start the frontend server
echo "Starting frontend server..."
cd ui
npm run dev &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

# Register the cleanup function for script termination
trap cleanup SIGINT SIGTERM

# Keep the script running
echo "Development environment is running."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:8080"
echo "Press Ctrl+C to stop both servers."
wait 