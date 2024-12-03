import os
from openai import OpenAI, completions
import json
from dotenv import load_dotenv
from vector_database_setter import query_documents

def create_prompt(retrieved_texts, user_query):
    context = "\n\n---\n\n".join(retrieved_texts)
    prompt = f"""
    You are an expert assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {user_query}

    Answer:
    """
    return prompt

def generate_answer(prompt):
    response = completions.create(
        model="gpt-3.5-turbo-instruct",  # Use a valid chat model name
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

def rag_pipeline(query, k=3):
    results = query_documents(query, k)
    print("Results 0:")
    retrieved_texts = [doc.page_content for doc in results]
    prompt = create_prompt(retrieved_texts, query)
    print("Prompt:\n", prompt)  # Printing the entire prompt
    answer = generate_answer(prompt)
    return answer

user_query = input("Enter your query: ")
final_output = rag_pipeline(user_query, k=3)
print("Final Output:\n", final_output)
