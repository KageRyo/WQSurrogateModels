# MPR_Model

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org)

Project Summary
---------------
MPR_Model provides a model training and prediction framework for water quality evaluation based on multivariate polynomial regression and machine learning approaches. The solution supports standard indicators and generates Water Quality Index (WQI/WQI5) estimates.

Key Features
------------
- Support for science-backed indicators: DO, BOD, NH3-N, EC, SS
- Model variants maintained in `models/` (e.g., LR, RF, SVM, XGBoost, LightGBM, MPR)
- REST API implementation under `src/api.py` for inference and data ingestion
- Training scripts in `src/training` for reproducibility on various datasets
- Example data in `data/` for quick evaluation and comparison

Getting Started
---------------
### Prerequisites
- Python 3.8 or later
- Virtual environment manager (recommended): `venv` or `conda`

### Install
```bash
git clone https://github.com/KageRyo/MPR_Model.git
cd MPR_Model
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Configuration
-------------
- `main.py` defaults to listening on `0.0.0.0:8000`.
- Update host/port in `main.py` or extend configuration handling in `src/api.py` as needed.

Usage
-----
#### Start the API server
```bash
python3 main.py
```

#### Train a model (example)
```bash
python3 src/training/mainLGBMVer.1.0.py
```

API Endpoints
-------------
- `POST /predict` : accepts JSON or CSV payload for one-to-many inference
- `GET /status` : health check endpoint

Refer to `src/api.py` for exact endpoint paths and data schemas.

Data Format
-----------
Accepted CSV shape:
| DO | BOD | NH3N | EC | SS |
Rows should contain numeric values.

Project Structure
-----------------
- `data/` : training and validation datasets
- `models/` : persisted model artifacts
- `src/` : API and training implementations
- `main.py` : service entrypoint

Code Quality
------------
- Include linting (`flake8`, `pylint`) in your development process.
- Add tests for new behavior under `src/test`.

Contributing
------------
1. Fork the repository
2. Create a feature branch
3. Add tests and documentation updates
4. Submit a pull request with verification steps

License
-------
Apache License Version 2.0. See `LICENSE`.

Maintainers
-----------
- Chien-Hsun Chang (KageRyo)
- Kuo-Wei Wu (RRAaru)

