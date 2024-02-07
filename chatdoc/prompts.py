from langchain.prompts import PromptTemplate

string_q2d_zs = """Write a passage that answers the following query: {query}"""

string_cot = """Question: {query}

Answer:
"""

string_conv = """This is a conversation, Always generate grammatically correct sentences that no repeat. The user is a very understanding. Respond to recent discussion between user and you
{conversation}"""


prompt_q2d_zs = PromptTemplate.from_template(string_q2d_zs)
prompt_cot = PromptTemplate.from_template(string_cot)
prompt_conv = PromptTemplate.from_template(string_conv)