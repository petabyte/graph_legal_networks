# Graphiti Ingest — How to Restart

## Quick start

```bash
# 1. Keep the model loaded (prevents Ollama from unloading after idle — run once per session)
curl -s http://localhost:11434/api/generate -d '{"model": "qwen3.5:latest", "keep_alive": -1}' > /dev/null

# 2. Start the ingest (resume-safe — picks up where it left off)
nohup python3.12 scripts/graphiti_parallel.py > logs/graphiti_parallel.log 2>&1 &
```

## Check status

```bash
python3.12 scripts/graphiti_parallel.py --status
```

While a shard is actively writing, Kuzu holds an exclusive lock and the status
will show 0 for that shard — this is normal. Check the lock:

```bash
lsof data/shards/shard_1/kuzu.db
```

Check which case is currently being processed:

```bash
cat data/ingest_last_case.txt
```

## Stop safely

```bash
# Find the PID
pgrep -f graphiti_parallel

# Kill it — all written data is safe, resumes on next run
kill <PID>
```

## Handle segfaults (Kuzu crash)

If the process dies with a segfault, the offending case ID is in the sentinel file:

```bash
cat data/ingest_last_case.txt
# e.g. shard=1 case_id=109836
```

Add it to the skip list and restart:

```bash
echo "109836" >> data/ingest_skip.txt
nohup python3.12 scripts/graphiti_parallel.py > logs/graphiti_parallel.log 2>&1 &
```

Known bad cases so far: `109836`

## Monitor logs

```bash
tail -f logs/graphiti_parallel.log
```
