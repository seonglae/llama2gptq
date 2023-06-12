from re import split

import torch
import streamlit as st
from streamlit_chat import message

from src.qa import ask


TITLE = 'Angry Face'


st.set_page_config(page_title=TITLE)
st.header(TITLE)
st.markdown('''
### Ask anythig to [Texonom](https://texonom.com).
Question for recently learned
''', unsafe_allow_html=True)

styl = """
<style>
    .stTextInput {
      position: fixed;
      bottom: 3rem;
      z-index: 1;
    }
    .StatusWidget-enter-done{
      position: fixed;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }
    .StatusWidget-enter-done button{
      display: none;
    }
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

if 'generated' not in st.session_state:
  st.session_state['generated'] = []

if 'past' not in st.session_state:
  st.session_state['past'] = []


def query(query):
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  _, answer_refs, answer, output_refs = ask(
      query, device, None, None, None)

  # Append references
  refs = answer_refs + output_refs
  ref_set = {split(r"\\|/", ref.metadata["source"])[-1] for ref in refs}
  for ref in ref_set:
    slug = split(r" |.md", ref)[-2]
    answer += f"\n> https://texonom.com/{slug}"
  return answer


def get_text():
  input_text = st.text_input("You: ", key="input")
  return input_text


user_input = get_text()


if user_input:
  output = query(user_input)
  st.session_state.past.append(user_input)
  st.session_state.generated.append(output)


if st.session_state['generated']:
  for i, _ in enumerate(st.session_state['generated']):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    message(st.session_state["generated"][i], key=str(i), seed=13)
