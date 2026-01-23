from locust import HttpUser, task, between
import random

class LegalRAGUser(HttpUser):
    """
    Load test for Legal RAG API
    Simulates multiple users querying the system
    """
    wait_time = between(1, 3)
    
    # Sample queries representing different query types
    exact_queries = [
        "What is Article 15?",
        "Show me Article 6.1",
        "What is Article 9.2.a?",
        "Display Article 7",
    ]
    
    conceptual_queries = [
        "What are the consent requirements?",
        "What rights do data subjects have?",
        "How does GDPR define personal data?",
        "What are the principles of data processing?",
    ]
    
    comparison_queries = [
        "What's the difference between Article 6 and Article 7?",
        "Compare consent and legitimate interest",
    ]
    
    @task(3)
    def query_exact_reference(self):
        """Test exact article reference queries (most common)"""
        query = random.choice(self.exact_queries)
        self.client.post("/chat", json={
            "query": query,
            "session_id": f"load_test_{self.environment.runner.user_count}"
        })
    
    @task(2)
    def query_conceptual(self):
        """Test conceptual queries"""
        query = random.choice(self.conceptual_queries)
        self.client.post("/chat", json={
            "query": query,
            "session_id": f"load_test_{self.environment.runner.user_count}"
        })
    
    @task(1)
    def query_comparison(self):
        """Test comparison queries (less common)"""
        query = random.choice(self.comparison_queries)
        self.client.post("/chat", json={
            "query": query,
            "session_id": f"load_test_{self.environment.runner.user_count}"
        })
    
    @task(1)
    def check_health(self):
        """Periodically check health endpoint"""
        self.client.get("/health")