from openai import OpenAI
import pandas as pd

client = OpenAI()

def generate_research_area():
    prompt = "Generate a random academic research area."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

def generate_research_method(area):
    prompt = f"Suggest a research method commonly used in {area}."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

def generate_research_abstract(area, method):
    prompt = f"""
    Generate a research abstract for a study in {area}.
    The research primarily uses {method} methodology.
    The abstract should be structured and sound scientifically valid.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

def generate_synthetic_paper():
    area = generate_research_area()
    method = generate_research_method(area)
    abstract = generate_research_abstract(area, method)

    return {
        "Abstract": abstract,
        "Area of Study": area,
        "Methodology": method
    }

synthetic_data = [generate_synthetic_paper() for _ in range(50)]

df = pd.DataFrame(synthetic_data)

df.to_csv("synthetic_research_data.csv", index=False)
