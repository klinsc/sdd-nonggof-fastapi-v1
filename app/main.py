import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

from app.dependencies import get_query_token
from app.routers import chat

app = FastAPI(dependencies=[Depends(get_query_token)])

# --- CORS Configuration ---
origins = [
    # "http://localhost",
    "http://localhost:3000",  # Your frontend's origin
    "https://sdd.chatbordin.com",
    "http://sdd.chatbordin.com",
    "https://ssd-web-beta.vercel.app",
    "https://sdd-nonggof-reverse.chatbordin.com",
    # You can add more origins if your frontend is deployed elsewhere, e.g.,
    # "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins, or specify `origins` list to restrict
    allow_credentials=True,  # Allow cookies/authentication headers to be sent
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers (e.g., Content-Type, Authorization)
)
# --- End CORS Configuration ---

app.include_router(
    chat.router,
)
# app.include_router(users.router)
# app.include_router(items.router)
# app.include_router(
#     admin.router,
#     prefix="/admin",
#     tags=["admin"],
#     dependencies=[Depends(get_token_header)],
#     responses={418: {"description": "I'm a teapot"}},
# )


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
