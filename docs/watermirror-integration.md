# WaterMirror Integration Guide

This document describes the API contract between **WaterMirror** (frontend) and **WQSurrogateModels** (backend).

## Overview

WaterMirror communicates with this backend using two patterns:

1. **Modern JSON API** (recommended for new features)
2. **Legacy CSV Upload API** (still actively used by WaterMirror for batch CSV analysis)

---

## 1. Modern Endpoints (JSON)

### `POST /predict`

**Request Body (JSON):**

```json
{
  "DO": 7.2,
  "BOD": 2.1,
  "NH3N": 0.3,
  "EC": 450,
  "SS": 12,
  "model_type": "lightgbm"
}
```

- `model_type`: One of `direct_wqi5`, `lr`, `mpr`, `svm`, `rf`, `xgboost`, `lightgbm`. Defaults to `direct_wqi5` if omitted.

**Response:**

Uses `PredictionResponse` schema (see below).

---

## 2. Legacy CSV Endpoints (Used by WaterMirror)

These endpoints are called via `multipart/form-data` when users upload CSV files in WaterMirror.

### `POST /score/total/`

Calculates the **mean WQI5 score** across all rows in the uploaded CSV.

**Form Data:**
- `file`: CSV file (must contain columns: `DO,BOD,NH3N,EC,SS`)
- `model_type` (optional): string, e.g. `direct_wqi5` or `lightgbm`

**Response:** `PredictionResponse`

### `POST /score/all/`

Calculates WQI5 score **for every row** in the uploaded CSV.

**Form Data:**
- `file`: CSV file
- `model_type` (optional)

**Response:**

```json
{
  "scores": [82.5, 75.3, ...],
  "model_type": "lightgbm",
  "latency_ms": 23.4
}
```

> **Important:** `model_type` is always returned as a plain string.

---

## 3. Shared Response Types

### PredictionResponse

```json
{
  "score": 82.5,
  "category": "Good",
  "rating_range": "70 < WQI5 ≤ 85",
  "model_type": "lightgbm",
  "latency_ms": 12.4,
  "assessment": {
    "DO": "Good",
    "BOD": "Fair",
    ...
  },
  "warnings": []
}
```

### Other Utility Endpoints Used by WaterMirror

- `GET /status`
- `GET /percentile?score=82.5`
- `GET /categories`

---

## 4. Model Types

Supported values for `model_type`:

| Value        | Description                     | Requires Pre-trained Model |
|--------------|----------------------------------|------------------------------|
| `direct_wqi5` | Formula-based baseline          | No                           |
| `lr`          | Linear Regression               | Yes                          |
| `mpr`         | Multiple Polynomial Regression  | Yes                          |
| `svm`         | Support Vector Machine          | Yes                          |
| `rf`          | Random Forest                   | Yes                          |
| `xgboost`     | XGBoost                         | Yes                          |
| `lightgbm`    | LightGBM                        | Yes                          |

---

## 5. CORS Configuration

When running WaterMirror (especially in development or web builds), the backend must allow requests from the frontend origin.

**Environment Variable:**

```env
CORS_ALLOW_ORIGINS=http://localhost:8081,https://your-watermirror-domain.com
```

- Use `*` only for local development.
- Separate multiple origins with commas.

---

## 6. Backward Compatibility Notes

- The backend guarantees that `model_type` is always returned as a **plain string** in all responses.
- The legacy endpoints (`/score/total/`, `/score/all/`) must remain stable as WaterMirror still relies on them for CSV batch processing.
- When `model_type` is not provided, the backend uses the value from `DEFAULT_MODEL` environment variable (default: `direct_wqi5`).

---

## 7. Recommended Environment Variables for WaterMirror Integration

```env
# Backend
DEFAULT_MODEL=direct_wqi5
CORS_ALLOW_ORIGINS=*

# WaterMirror (client)
EXPO_PUBLIC_API_BASE_URL=http://localhost:8001
EXPO_PUBLIC_DEFAULT_MODEL=direct_wqi5
EXPO_PUBLIC_REQUEST_TIMEOUT_MS=10000
```

---

**Maintained for compatibility between WaterMirror and WQSurrogateModels.**

Last updated: 2026
