
---

## 📘 CADL4 – Topic Modeling with LDA

```markdown
# Topic Modeling with LDA (Latent Dirichlet Allocation)

## 📌 CADL Activity 4

### 🔹 Task
- Preprocess a text corpus (tokenization, stop word removal).  
- Apply **LDA (Latent Dirichlet Allocation)** for topic modeling.  
- Visualize topics with **pyLDAvis**.  

---

### 🛠️ Requirements
- Python 3.x  
- Libraries:  
  ```bash
  pip install nltk gensim pyLDAvis

Example Corpus
"The economy is recovering after the pandemic with growth in the IT sector."
"Artificial Intelligence and Machine Learning are transforming industries."
"Climate change is a pressing global issue that needs immediate attention."
"Social media platforms are influencing the opinions of young people."
"Healthcare systems are adopting digital technologies for better patient care."
"New advancements in space exploration are capturing the world's attention."

Run the Program
python lda_topic_modeling.py

Expected Output
Example Topics (LDA)
Topic 0: "ai", "machine", "learning", "industries"
Topic 1: "economy", "growth", "pandemic", "it"
Topic 2: "climate", "change", "global", "attention"

Visualization
The program generates an interactive visualization file:
lda_vis.html


Open it in your browser to explore topics and keyword distributions.
