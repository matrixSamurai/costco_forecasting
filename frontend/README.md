# Frontend

Web UI for the Costco Route Delay Forecasting demo.

## Structure

| Item | Description |
|------|-------------|
| **index.html** | Single page: scenarios, weather form, results cards, bar chart. |
| **static/css/style.css** | Costco Wholesale styling and layout. |
| **static/js/app.js** | Form logic, presets, `POST /api/predict`, Chart.js. |

## Run

The frontend is **served by the backend**. From the project root or from `backend/`:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**. The Flask app serves `../frontend` as the UI and exposes `/api/predict` for delay predictions.

No build step; static HTML/CSS/JS only.
