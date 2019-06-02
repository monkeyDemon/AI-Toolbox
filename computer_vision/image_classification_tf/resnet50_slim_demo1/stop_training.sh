ps aux | grep "python src/train.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9
echo "training stopped..."

