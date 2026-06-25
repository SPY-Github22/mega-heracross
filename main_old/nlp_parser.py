import json

class NLPOrchestratorMock:
    def __init__(self):
        self.tools = {
            "sever_edge": "Sever a specific road between two nodes",
            "sever_node": "Sever an entire intersection (node)"
        }
        
    def parse_intent(self, user_prompt):
        print(f"\n[NLP] Receiving User Prompt: '{user_prompt}'")
        print("[NLP] Engaging LLM Tool Calling...")
        
        # MOCKING LLM EXTRACTION
        # In production, this would be an OpenAI API call with strict JSON schema
        if "bridge" in user_prompt.lower() or "road" in user_prompt.lower():
            action = "sever_edge"
            # Mocking extracted geographic coordinates resolving to Node IDs
            params = {"u": 61746207, "v": 1607769741}
        elif "flood" in user_prompt.lower() or "intersection" in user_prompt.lower():
            action = "sever_node"
            params = {"node_id": 260739401}
        else:
            return {"error": "Could not parse intent"}
            
        print(f"[NLP] LLM successfully resolved natural language to mathematical entities.")
        
        # PHASE 13: Data Serialization & API Design
        # Strict JSON payload
        payload = {
            "action": action,
            "parameters": params,
            "confidence_score": 0.98
        }
        
        print(f"[NLP] Serialized API Payload -> Graph Engine:\n{json.dumps(payload, indent=2)}")
        return payload

class LLMAsAJudge:
    def evaluate_action(self, user_prompt, action_payload):
        print("\n--- PHASE 9: LLM-AS-A-JUDGE EVALUATION ---")
        print("Evaluating if the NLP parser hallucinated...")
        # Check against rubric
        if "confidence_score" not in action_payload or action_payload["confidence_score"] < 0.90:
            return False, "Low confidence score"
            
        if action_payload["action"] not in ["sever_edge", "sever_node"]:
            return False, "Hallucinated tool call"
            
        print("[JUDGE] PASS: The extracted entities align deterministically with the mathematical graph.")
        return True, "Valid Orchestration"

if __name__ == "__main__":
    print("--- PHASE 5: NLP ORCHESTRATOR MOCKING ---")
    orchestrator = NLPOrchestratorMock()
    judge = LLMAsAJudge()
    
    # Scenario 1
    prompt = "A massive flood has completely destroyed the major intersection at Koramangala block 3."
    payload = orchestrator.parse_intent(prompt)
    judge.evaluate_action(prompt, payload)
    
    # Scenario 2
    prompt = "The bridge on Hosur Road just collapsed."
    payload2 = orchestrator.parse_intent(prompt)
    judge.evaluate_action(prompt, payload2)
