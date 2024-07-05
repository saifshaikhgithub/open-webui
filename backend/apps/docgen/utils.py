import regex as re
import json

def clean_output(model_output):

    output = model_output

    section_title_pattern = r'^.*Section Title: '
    section_content_pattern = r'^^.*Section Content: '
    summary_pattern = r'^.*Summary:.*$'

    output = re.sub(summary_pattern, '', output, flags=re.MULTILINE)
    output = re.sub(section_content_pattern, '', output, flags=re.MULTILINE)
    output = re.sub(section_title_pattern, '', output, flags=re.MULTILINE)

    return output



def get_summaries(query):

    summary_pattern = r'^.*Summary:(.+)\s+.+'

    summaries = re.findall(summary_pattern, query, re.MULTILINE)

    return summaries

def get_summary(prompt):

    summary_pattern = r'^.*Summary:\s+(.+)'
    summary = re.findall(summary_pattern, prompt,re.MULTILINE)

    return summary


def get_titles(query):
    

    pattern = r'^\d+\.\s*(.+?):'

    groups = re.findall(pattern, query, re.MULTILINE)

    return groups






