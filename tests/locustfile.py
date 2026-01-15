from locust import HttpUser, task, between

class APIUser(HttpUser):
    # Wait between 1 and 3 seconds between tasks (simulates reading)
    wait_time = between(1, 3)

    @task
    def chat_test(self):
        # We send a standard query to see how the RAG engine handles it
        payload = {
            "query": "What are the administrative fines?",
            "session_id": "stress_test_user"
        }
        
        # The client automatically tracks if this returns 200 (Success) or 500 (Fail)
        self.client.post("/chat", json=payload)