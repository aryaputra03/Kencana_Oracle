Kencana Oracle: End-to-End Gold Price Forecasting System V2 🪙

Kencana Oracle is an advanced, highly scalable microservices architecture designed for real-time Gold Price (XAU/IDR) forecasting. Version 2.0 bridges the gap between data science experimentation and production-grade inference by introducing machine learning algorithms, high-speed caching mechanisms, and a comprehensive monitoring suite.


KEY FEATURES


- Advanced Multi-Model Forecasting:
  Supports machine learning (XGBoost) and additive models (Prophet), alongside fine-tuned statistical models (ARIMA & SARIMA) for comprehensive multi-model comparisons.

- Optimized API with Redis Caching:
  Low-latency FastAPI REST backend integrated with Upstash Redis caching. It significantly reduces inference times and server computational load.

- On-the-Fly Feature Engineering:
  Robust automated preprocessing pipeline that dynamically generates complex time-series features (lags, rolling stats, differencing) during real-time inference.

- Multipage Dashboard & Model Comparison:
  Native multipage Streamlit application featuring a "Compare Models" tool for simultaneous multi-line forecasting visualization and 1-click CSV data export.

- Cloud DB & Admin Monitoring System:
  Persistent audit trails using PostgreSQL (Supabase) and SQLAlchemy ORM, complemented by a secure Admin Monitoring UI to track API logs and latency.

- Enterprise-Grade Architecture:
  Enforces Separation of Concerns (SoC) by isolating data loading, feature engineering, and inference routing layers.


TECH STACK


Core / Data Science:
- Python
- Pandas
- Numpy
- Statsmodels
- XGBoost
- Prophet

Backend / API:
- FastAPI
- Pydantic
- Uvicorn

Frontend:
- Streamlit
- Plotly

Database & Caching:
- PostgreSQL (Supabase)
- SQLAlchemy ORM
- Redis (Upstash)

Testing & DevOps:
- Pytest
- Docker
- Git


PROJECT STRUCTURE


Kencana_Oracle/
├── artifacts/                 
│   ├── arima_model.pkl         
│   ├── sarima_model.pkl        
│   ├── prophet_model.json     
│   └──  xgboost_model.json      
│   
│
├── config/                     
│   ├── __init__.py
│   └── setting.py              
│
├── src/                        
│   ├── __init__.py
│   ├── preprocessor.py          
│   ├── cache_manager.py        
│   ├── loader.py               
│   ├── predictor.py            
│   ├── utils.py                
│   ├── database.py             
│   ├── models.py               
│   └── crud.py                  
│
├── scripts/
│   └── seed_data.py            
│
├── api/                       
│   ├── __init__.py
│   ├── main.py                 
│   └── schemas.py              
│
├── dashboard/                 
│   ├── app.py                  
│   └── pages/                  
│       └── 1_monitoring.py     
│
├── tests/                      
│   ├── __init__.py
│   ├── test_predictor.py       
│   └── test_ml_inference.py     
│
├── notebook/
│   └── main.ipynb
│
├── .env                        
├── .gitignore                  
├── requirements.txt            


INSTALLATION & SETUP


1. Clone the Repository

git clone https://github.com/yourusername/kencana-oracle.git
cd kencana-oracle

2. Create Virtual Environment & Install Dependencies

python -m venv venv

On Linux / Mac:
source venv/bin/activate

On Windows:
venv\Scripts\activate

pip install -r requirements.txt

3. Environment Variables (.env)

Create a .env file in the root directory and configure:

# Database Configuration (Supabase/PostgreSQL)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Redis Cache Configuration (Upstash)
REDIS_URL=rediss://default:password@host:port
CACHE_TTL=3600

4. Run the Application

Terminal 1: Start FastAPI Backend
uvicorn api.main:app --reload --port 8000

API Docs:
http://localhost:8000/docs

Terminal 2: Start Streamlit Dashboard
streamlit run dashboard/app.py

Dashboard:
http://localhost:8501


API ENDPOINTS SUMMARY


GET    /                     -> API Health Check
GET    /prices/history       -> Retrieve historical gold price data
POST   /predict/future       -> Predict future prices using a specific model
POST   /predict/compare      -> Compare predictions across multiple AI models
GET    /predictions/history  -> Admin endpoint to view prediction logs


TESTING


Run automated test suite:

pytest tests/ -v


AUTHOR


Developed by R.Ganendra Geanza Aryaputra

LinkedIn:
www.linkedin.com/in/ganendra-geanza-aryaputra-b8071a194

GitHub:
https://github.com/aryaputra03/Kencana_Oracle