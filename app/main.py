from fastapi import FastAPI

app = FastAPI(
    title="Industrial Knowledge-Base RAG Assistant",
    description="A RAG assistant for industrial documentation.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "industrial-knowledge-rag"
    }