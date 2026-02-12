from openai import OpenAI
import re
import time
import json, random
import os

class ChatGPT:
    def __init__(self, model_version):
        self.OPENAI_API_KEY = ''
        self.model = model_version
        self.messages = []
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    def add_role(self, role):
        self.messages.append({"role": "system", "content": role})

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def check_answer(self, answer):
        # First attempt: Match JSON that includes "Reasoning"
        match = re.search(r'({.*"Political":.*?"Reasoning":.*?})', answer, re.DOTALL)
        
        if match:
            json_part = match.group(1)  # Extract the JSON part including "Reasoning"
        else:
            # Second attempt: Match JSON that includes up to the second closing brace after "Bias"
            match = re.search(r'({.*"Political":.*?"Bias":.*?}\s*})', answer, re.DOTALL)
            if match:
                json_part = match.group(1)  # Extract the JSON part up to the second closing brace
            else:
                json_part = None  # Return None if no match is found
        
        return json_part
    
    def run(self, prompt):
        self.add_message("user", prompt)
        
        for attempt in range(self.max_retries):
            try:
                self.client = OpenAI(api_key=self.OPENAI_API_KEY)
                
                if self.model == 'o3':
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=0.2,
                    )
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=0.2,
                    )

                # print("completion: ", completion)
                answer = completion.choices[0].message.content
                
                # Convert the completion object to a dictionary and print it as JSON
                completion_dict = completion.to_dict()  # Convert the response to a dictionary
                completion_json = json.dumps(completion_dict, indent=4)
                print(f"Completion in JSON format (Attempt {attempt + 1}):")

                checked_answer = self.check_answer(answer)
                if checked_answer:
                    self.add_message("assistant", answer)
                    return checked_answer
                else:
                    print(f"Attempt {attempt + 1} failed to produce a valid answer. Retrying...")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
            
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                time.sleep(60)

        print("Max retries reached. Unable to get a valid answer.")
        return None

# Usage example:
# if __name__ == "__main__":
#     chatgpt = ChatGPT('gpt-4o')
#     role_prompt_path = os.path.join(current_dir, '../prompt_fewshot_4dim_perspective', 'prompt_role_opposed_left.txt')
#     with open(role_prompt_path, 'r', encoding='utf-8') as f:
#         prompt_template = f.read()
#     role_prompt =prompt_template.format(query="immigration")
#     chatgpt.add_role(role_prompt)
#     result = chatgpt.run("Analyze the political bias in this statement: [Your statement here]")
#     print("Final result:", result)
