# Text-to-SQL RAG

**File:** `07_advanced/text_to_sql.ipynb`

## Core Concept
Three-stage pipeline: (1) LLM translates a natural language question into SQL, (2) the query executes against a real database, (3) LLM synthesizes the raw rows back into a human-readable answer.

## What You Learn
- Inject the database schema into the prompt so the LLM generates valid column names
- Strip markdown code block wrappers that local models add around SQL
- Execute generated SQL with error handling at the database boundary
- Stream the final synthesis for real-time user feedback

## Key Constructs
```python
import sqlite3

schema = """
Table: system_alerts
Columns: alert_id, cluster_name, port_number, alert_type, severity, incident_date
severity examples: 'Critical', 'Urgent', 'Warning'
"""

def generate_sql(question: str) -> str:
    prompt = [
        {"role": "system", "content": f"Convert to SQL. Schema:\n{schema}\nReturn ONLY the SQL."},
        {"role": "user",   "content": question}
    ]
    raw = llm_call(prompt).strip()

    # Strip markdown wrappers (common with local models)
    if raw.startswith("```"):
        raw = raw.replace("```sql", "").replace("```", "").strip()
    return raw

def run_pipeline(question: str):
    sql = generate_sql(question)

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"SQL error: {e}")
        return

    # Synthesize rows → natural language (streamed)
    for chunk in llm_stream([
        {"role": "system", "content": "Convert raw DB rows to a clear sentence."},
        {"role": "user",   "content": f"Question: {question}\nRows: {rows}"}
    ]):
        print(chunk, end="", flush=True)
```

## Three Failure Points
| Stage | Common Failure | Fix |
|---|---|---|
| SQL generation | Markdown wrappers, wrong column names | Schema injection + markdown strip |
| SQL execution | Syntax error, missing table | `try/except` with clear error message |
| Synthesis | Loses numeric precision | Instruct LLM to preserve exact numbers |

## Pitfalls
- Schema must include example values — LLM won't guess `'Critical'` vs `'critical'` correctly
- Never interpolate user input directly into SQL — use parameterized queries for production
- Synthesis can silently round or summarize numbers — validate against raw rows for accuracy-critical use cases
