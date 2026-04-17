import json

def chat_with_report(question: str, context: str, llm_client=None, model: str = None) -> str:
    """Answers a question based on the provided report context using an LLM."""
    if not llm_client:
        # Fallback if no LLM is provided
        return f"I see you're asking about '{question}'. Since I don't have an active LLM connection right now to process the context, please refer to the downloaded report."
        
    system_prompt = (
        "You are an expert AI consulting assistant. Your task is to answer "
        "the user's question based strictly on the provided report context. "
        "Be concise, professional, and act like a McKinsey consultant. "
        "If the answer is not in the context, say so clearly."
    )
    
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    
    try:
        response = llm_client.chat.completions.create(
            model=model or "meta-llama/llama-3.2-1b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error while trying to answer: {e}"
