# NN Analyzer

## Project Structure

```
project/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── requirements.txt
│   ├── model_processor/
│   │   └── model_processor.py   # Your existing Model_Processor class
│   └── vector_analyzer.py       # Your existing Vector_Analyzer class
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   └── main.tsx
    ├── index.html
    ├── package.json
    └── vite.config.ts
```

## Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: http://localhost:5173

## API Endpoints

| Method | Path                      | Description                          |
|--------|---------------------------|--------------------------------------|
| POST   | /upload/model             | Upload a model file, returns session |
| POST   | /upload/dataset           | Upload a dataset for a session       |
| POST   | /inference/run            | Run inference on the loaded dataset  |
| GET    | /inference/results        | Paginated inference results          |
| POST   | /analysis/similar-pairs   | Find similar activation vector pairs |
| GET    | /model/data               | Get model architecture data          |
| DELETE | /session/{id}             | Clean up a session                   |

## Usage Flow

1. Upload a `.keras`, `.onnx`, `.pt`, or `.pth` model file
2. Upload a `.csv` or `.npz` test dataset
3. Click **Run Inference**
4. Adjust the cosine distance threshold and click **Find Similar Pairs**