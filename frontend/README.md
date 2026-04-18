# AlignMetric Frontend

Open [index.html](C:/Users/Admin/Documents/Playground/frontend/index.html) in a browser after the backend is running.

## Backend URL

By default, the page sends requests to:

```text
http://localhost:8000
```

If your backend is running somewhere else, define this before the page script runs:

```html
<script>
  window.ALIGNMETRIC_API_BASE_URL = "http://localhost:8000";
</script>
```
