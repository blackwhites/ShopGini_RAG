
from flask import Flask, render_template, request
from llama_parse import LlamaParse
from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

import os
import nest_asyncio

app = Flask(__name__)

vectara_customer_id = "1462565482"
vectara_corpus_id1 = "4" #Corpus for processed products from LlamaIndex
vectara_corpus_id2 = "3" #Corpus for original products pdf
vectara_api_key = "zut_Vyz6aoLQDc3em061SmzHAGBIcQry2_FdgqXs5A"


nest_asyncio.apply()

# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-012ehqGlaumwNHfTCXma3lFFifYd4K6qdaO1aK800cIVVAAf"

# Using OpenAI API for embeddings/llms
os.environ["OPENAI_API_KEY"] = "sk-proj-wL2UU3nfX6O3RDfmlEh8T3BlbkFJgoix4eOKkeJfhgdZRAsO"

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-3.5-turbo-0125")

Settings.llm = llm
Settings.embed_model = embed_model

fname =  "./docs/AProducts.pdf"
documents = LlamaParse(result_type="markdown", verbose=True, language="en").load_data(fname)

# sync
#documents = parser.load_data("./docs/NCEC.pdf")
#print(documents)
# with vectara1, we upload the markdown file that is the output of the LlamaParse processing to Vectara
temp_fname = 'LlamaProducts.md'
with open(temp_fname, 'w', encoding='utf-8') as f:
    f.write(documents[0].text)
vectara1 = VectaraIndex(vectara_customer_id=vectara_customer_id, 
                        vectara_corpus_id=vectara_corpus_id1, 
                        vectara_api_key=vectara_api_key)
vectara1.insert_file(temp_fname)
qe1 = vectara1.as_query_engine(similarity_top_k=10)
    
# with vectara2, we upload the PDF directly to Vectara

vectara2 = VectaraIndex(vectara_customer_id=vectara_customer_id, 
                        vectara_corpus_id=vectara_corpus_id2, 
                        vectara_api_key=vectara_api_key)
vectara2.insert_file(fname)
qe2 = vectara2.as_query_engine(similarity_top_k=10)

query = input("\n\n\nEnter your Query: ")

#query = "Provides information about Apple financials for year 2021?"
#query = "Tell me the top three product name that has big discount and show me the result in percentage? Format the response in list manner."

response_1 = qe1.query(query)
print("\n*** Vectara + LlamaParse ***")
print(response_1)

response_2 = qe2.query(query)
print("\n*** Vectara Native ***")
print(response_2)

# if __name__ == '__main__':
#   app.run(debug=True)