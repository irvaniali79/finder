import sys
from dotenv import load_dotenv
load_dotenv()
from src.pipeline import build_pipeline
from src.agent import QAAgent

def main():
    print("Company Knowledge Bot - Ask me anything about our policies and products!")
    print("Type 'exit' or 'quit' to stop.\n")

    pipeline = build_pipeline(variant="V5")
    agent = QAAgent()
    print("Ready! Knowledge base loaded.\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        chunks = pipeline.retrieve(query)
        answer = agent.answer(query, chunks)
        
        print("\nAnswer: {}\n".format(answer))

if __name__ == "__main__":
    main()