# %%
import pandas as pd
from calc import cost, tokens_counter


# %%

entry = open(r'C:\Users\JUAN\Desktop\.begin\tudo\vscode\APIOpenAI\data\input.txt', 'r')
gpt_input = entry.read()

result = open(r'C:\Users\JUAN\Desktop\.begin\tudo\vscode\APIOpenAI\data\output.txt', 'r')
gpt_output = result.read()


# %%
token_input = tokens_counter(gpt_input)
token_output = tokens_counter(gpt_output)
report_cost = cost(token_input, token_output)


# %%
num_tokens = {
    'Número de tokens (input)': token_input,
    'Número de tokens (output)': token_output
}

df_tokens = pd.DataFrame(list(num_tokens.items()),
                         columns=['item', 'Quantidade'])
df_tokens

# %%
final_report = pd.DataFrame(list(report_cost.items()), columns=[
                            'Versão GPT', 'Custo'])
final_report
