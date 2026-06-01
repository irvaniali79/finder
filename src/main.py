import sys
from src.retrieval import DocumentRetriever
from src.agent import QAAgent

def main():
    print("Company Knowledge Bot - Ask me anything about our policies and products!")
    print("Type 'exit' or 'quit' to stop.\n")
    
    retriever = DocumentRetriever()
    agent = QAAgent()
    
    print("Loading knowledge base...")
    retriever.setup()
    print("Ready! Knowledge base loaded.\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        chunks = retriever.retrieve(query, top_k=3)
        answer = agent.answer(query, chunks)
        
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()