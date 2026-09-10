"""User-message templates extracted unchanged from the reference implementations."""
import textwrap

def clean(given_sentence, given_full_response, given_question):
    return textwrap.dedent(f'    §GIVEN QUESTION§: {given_question}\n\n    §GIVEN FULL RESPONSE§: {given_full_response}\n\n    §GIVEN SENTENCE§: {given_sentence}\n\n    ')

def judge(scoring_point, matched_sentences_list):
    return textwrap.dedent(f'    Now process the below information:\n    ==========================\n    §Scoring point§:\n    {scoring_point}\n\n    §Matched sentences§:\n    {matched_sentences_list}\n    ==========================\n    ')

def fact_decompose(given_sentence):
    return textwrap.dedent(f'§Given Sentence§: {given_sentence}')

def fact_clarify(given_unit_facts_list, given_question, given_full_response):
    return textwrap.dedent(f'\n    §Given Unit Facts List§: {given_unit_facts_list}\n    §Given Question§: {given_question}\n    §Given Full Response§: {given_full_response}\n    ')

def fact_check(given_sentence, given_fact):
    return textwrap.dedent(f'\n                                Now process the below information:\n                                ==========================\n                                §Given Unit Fact§: {given_fact}\n                                §Given Sentence§: {given_sentence}\n                                ==========================\n                                ')

def fact_judge(scoring_point, matched_sentences_list, all_fact_check_results):
    return textwrap.dedent(f'    Now process the below information:\n    ==========================\n    §All fact check results you could refer to§:\n    {all_fact_check_results}\n    \n    §Scoring point§: \n    {scoring_point}\n    \n    §Matched sentences§: \n    {matched_sentences_list}\n    ==========================\n    ')

