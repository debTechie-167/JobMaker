# Resume vs Job Matcher Backend

## Run locally

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Environment variables

Copy `.env.example` to `.env` or set these in your shell:

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (optional)
- `CORS_ORIGINS` (optional, comma-separated)

## API

`POST /api/analyze`

Form fields:

- `resume`: PDF, DOCX, or TXT file
- `job_description`: plain text job description

Example frontend request:

```js
const formData = new FormData();
formData.append("resume", file);
formData.append("job_description", jobDescription);

const response = await fetch("http://localhost:8000/api/analyze", {
  method: "POST",
  body: formData,
});

const data = await response.json();
```
