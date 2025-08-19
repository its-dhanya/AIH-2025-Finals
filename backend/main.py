from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router

# 1. Create the main FastAPI application instance
app = FastAPI(title="weavedocs API")

# 2. Configure CORS to allow your frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Your React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Include the router from your api.py file
app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the weavedocs API"}