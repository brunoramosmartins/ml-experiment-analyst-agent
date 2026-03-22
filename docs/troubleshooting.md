# Troubleshooting

Common errors and their solutions when running the ML Experiment Analyst Agent.

---

## MLflow Connection Errors

**Symptom:** `ERROR loading experiment: Could not connect to MLflow at http://localhost:5000`

**Solutions:**

1. Make sure Docker is running:
   ```bash
   docker compose up -d
   ```

2. Wait 10-15 seconds for the containers to initialize, then check health:
   ```bash
   curl http://localhost:5000/health
   ```

3. Check container logs:
   ```bash
   docker compose logs mlflow
   ```

4. If port 5000 is in use by another service, update `MLFLOW_TRACKING_URI` in `.env`.

---

## Ollama Not Responding

**Symptom:** `Connection refused` or timeout when calling the agent with `LLM_PROVIDER=ollama`.

**Solutions:**

1. Check that Ollama is running:
   ```bash
   ollama list
   ```

2. If not running, start it:
   ```bash
   ollama serve
   ```

3. Verify the model is pulled:
   ```bash
   ollama pull llama3.1:8b
   ```

4. Check the base URL matches your config:
   ```bash
   curl http://localhost:11434/api/tags
   ```

---

## Tool Calling Failures with llama3

**Symptom:** Agent calls tools incorrectly or fails to use tools at all.

**Cause:** `llama3:latest` (Llama 3.0) does **not** support tool calling. This is a known limitation of the base Llama 3 model.

**Solution:** Use a model that supports tool calling:

```env
LLM_MODEL=llama3.1:8b     # recommended
# or
LLM_MODEL=llama3.2:3b     # lighter alternative
```

Then pull the model:
```bash
ollama pull llama3.1:8b
```

---

## Tavily Web Search Not Loading

**Symptom:** Only 6 tools loaded instead of 7. The `web_search` tool is missing.

**Cause:** The `TAVILY_API_KEY` environment variable was not set when the tools module was imported. The key check happens at import time.

**Solutions:**

1. Ensure `TAVILY_API_KEY` is set in your `.env` file:
   ```
   TAVILY_API_KEY=tvly-your-key-here
   ```

2. Make sure `python-dotenv` is installed and `.env` is loaded before tool imports. The tool registry calls `load_dotenv()` automatically, but if you import tools before the `.env` file exists, the key will not be found.

3. Verify in your script:
   ```python
   from src.tools import ALL_TOOLS
   print(f"Tools loaded: {len(ALL_TOOLS)}")
   # Should show 7 if Tavily is configured
   ```

---

## Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'deepagents'` or similar.

**Solutions:**

1. Make sure you are in the virtual environment:
   ```bash
   source .venv/bin/activate        # Linux / macOS
   # .venv\Scripts\activate         # Windows
   ```

2. Install the project in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Verify the install:
   ```bash
   python -c "from deepagents import create_deep_agent; print('OK')"
   ```

---

## MinIO Bucket Not Found

**Symptom:** MLflow artifact storage errors referencing S3 or MinIO buckets.

**Cause:** The `minio-init` container may have failed to create the default bucket.

**Solutions:**

1. Check the init container logs:
   ```bash
   docker compose logs minio-init
   ```

2. Restart the stack:
   ```bash
   docker compose down
   docker compose up -d
   ```

3. Manually access MinIO console at `http://localhost:9001` (default credentials: `minioadmin` / `minioadmin`) and verify the bucket exists.

---

## GovernanceLimitError

**Symptom:** `GovernanceLimitError: Token budget exceeded` or `GovernanceLimitError: Too many consecutive failures`.

**Cause:** The agent hit one of the governance safety limits.

**Solutions:**

- **Token budget exceeded:** Increase the limit in `.env`:
  ```
  AGENT_MAX_TOKENS=100000
  ```

- **Consecutive failures:** The agent failed 3+ tool calls in a row. Check the JSONL trace log for details:
  ```bash
  ls data/logs/agent_traces/
  ```
  Increase the threshold if needed:
  ```
  AGENT_MAX_FAILURES=5
  ```

- **Root cause:** Review the trace logs to understand why tools are failing. Common causes include MLflow being down or invalid experiment names.

---

## TimeoutError

**Symptom:** `TimeoutError` after the agent runs for the configured timeout period.

**Cause:** The agent took longer than `AGENT_TIMEOUT_SECONDS` (default: 300 seconds).

**Solutions:**

1. Increase the timeout in `.env`:
   ```
   AGENT_TIMEOUT_SECONDS=600
   ```

2. Use a faster LLM model or switch to Anthropic for faster inference:
   ```
   LLM_PROVIDER=anthropic
   ```

3. Simplify your query. Complex queries that require many tool calls take longer.

---

## Dashboard Shows No Data

**Symptom:** The Streamlit dashboard opens but shows empty tables and charts.

**Cause:** No JSONL trace logs have been generated yet.

**Solution:** Run the demo to generate trace data, then open the dashboard:

```bash
python scripts/run_demo.py
streamlit run src/dashboard/app.py
```

The dashboard reads from the directory configured in `AGENT_TRACE_LOG_DIR` (default: `data/logs/agent_traces/`).

---

## Tests Failing

**Symptom:** `pytest` fails with import errors or unexpected failures.

**Solutions:**

1. Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run only unit tests (no MLflow required):
   ```bash
   python -m pytest tests/unit/ tests/edge_cases/ -v
   ```

3. Integration tests require a running MLflow server:
   ```bash
   docker compose up -d
   python -m pytest tests/integration/ -v -m integration
   ```

4. Check environment variables are not interfering:
   ```bash
   LANGCHAIN_TRACING_V2=false python -m pytest tests/ -v
   ```
