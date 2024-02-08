from langchain.prompts import PromptTemplate

string_q2d_zs = """Write a passage that answers the following query: {query}"""

string_cot = """Question: {query}

Answer:
"""

string_conv = """This is a conversation, Always generate grammatically correct sentences that no repeat. The user is a very understanding. Respond to recent discussion between user and you
{conversation}"""

string_cot_prf = """Answer the following query based on the context:
Context: {context}

Question: {query}

Answer:
"""

string_cot_prf_mod = """Create a final answer to the given question using the provided context, NO GENERATE MORE QUESTIONS. If you are unable to answer the question, simply state that you do not have enough information to answer the question. If the information is not useful generate the better answer.

==============
Context
{context}
==============

Question: {query}


"""

prompt_q2d_zs = PromptTemplate.from_template(string_q2d_zs)
prompt_cot = PromptTemplate.from_template(string_cot)
prompt_conv = PromptTemplate.from_template(string_conv)
prompt_cot_prf = PromptTemplate.from_template(string_cot_prf)
prompt_cot_prf_mod = PromptTemplate.from_template(string_cot_prf_mod)