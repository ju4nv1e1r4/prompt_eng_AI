# %%
import tiktoken


# %%


def tokens_counter(text):
    enconder = tiktoken.get_encoding('cl100k_base')
    tokens = enconder.encode(text)
    token_count = len(tokens)

    return token_count


# %%
def cost(x, y):
    gpt_35_turbo_0613_input = x * 0.0000015
    gpt_35_turbo_0613_output = y * 0.000002
    gpt_35_turbo_16k_0613_input = x * 0.000003
    gpt_35_turbo_16k_0613_output = y * 0.000004
    total_0613 = gpt_35_turbo_0613_input + gpt_35_turbo_0613_output
    total_16k_0613 = gpt_35_turbo_16k_0613_input + gpt_35_turbo_16k_0613_output

    cost_table = {
        'GPT 3.5 Turbo 0613 (Input)': gpt_35_turbo_0613_input,
        'GPT 3.5 Turbo 0613 (Output)': gpt_35_turbo_0613_output,
        'GPT 3.5 Turbo 16k 0613 (Input)': gpt_35_turbo_16k_0613_input,
        'GPT 3.5 Turbo 16k 0613 (Output)': gpt_35_turbo_16k_0613_output,
        'Total GPT 3.5 Turbo 0613': total_0613,
        'Total GPT 3.5 Turbo 16k 0613': total_16k_0613
    }

    return cost_table
