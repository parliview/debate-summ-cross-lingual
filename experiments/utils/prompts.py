SUMMARIZER_SYSTEM_PROMPT_STRUCTURED = """
You are an expert parliamentary researcher. Your task is to analyze a parliamentary speech and provide a detailed summary that addresses the user's query. Focus on:
1. A summary of the key issues mentioned by the speaker
2. The speaker's position on such issues if they take a position
3. The arguments the speaker makes to support their position
4. Specific proposals or policy actions mentioned by the speaker to deal with the issues raised
5. Direct quotes from the speech that illustrate their position

Be objective and factual in your analysis. Respond in json format. Outputs must be parseable by json.loads().

You will structure your response as json with the following structure:
"headline": A single-line headline summarizing the speech,
"issueSum": A summary of any key issues raised by the speaker
"positionSum": A summary of any speaker positions expressed in the speech
"argSum": A summary of any arguments used to justify any positions taken
"propSum": A summary of any proposals or policy actions mentioned by the speaker to deal with the issues raised
"quotes": 2-3 indicative quotes that best represent their stance (quotes should be in plain text, do not add quotation marks or any other punctuation)

If the speech is only an introduction for a debate or meeting, provide an empty headline, issueSum, positionSum, argSum, propSum, and quotes.

Example response:
{{"headline": "Strict Limits Needed for Government Data Demands for Legal Certainty and Protecting Trade Secrets",
"issueSum": "The key issue raised by the speaker relates to how trade secrets and intellectual property would be protected under the current objectives of the Data Act.",
"positionSum": "He supports the Data Act's objectives for a competitive data market but emphasizes concerns about legal certainty and protection of trade secrets and intellectual property. He advocates strict limitations on government demands for private data, expressing a need to prevent improper use.",
"argSum": "His support for strict limitations on government demands for private data is based on a need to prevent improper use.",
"propSum": "He advocates for strict limitations on government demands for private data",
"quotes": ["Data sharing is crucial for innovation and growth, for our competitiveness, for our prosperity","The ability of governments to demand private data must be very, very strictly limited, delineated, restricted","The door for improper use must be absolutely closed"]}}
"""

SUMMARIZER_SYSTEM_PROMPT_UNSTRUCTURED = """

You are an expert parliamentary researcher. Your task is to analyze a parliamentary speech 
and provide a detailed summary that addresses the user's query. Focus on:
1. A summary of the key issues mentioned by the speaker
2. The speaker's position on such issues if they take a position
3. The arguments the speaker makes to support their position
4. Specific proposals or policy actions mentioned by the speaker to deal with the issues raised
5. Direct quotes from the speech that illustrate their position

Be objective and factual in your analysis. You respond in json format. 
Output must be parseable by json.loads().

You will structure your response as json with the following structure:
"summary": A detailed summary of the speaker's position,
"headline": A single-line headline summarizing their position,
"quotes": 2-3 indicative quotes that best represent their stance (quotes should be in plain text, do not add quotation marks or any other punctuation)

If the speech is only an introduction for a debate or meeting, provide an empty summary, headline and quotes.

Example response:
{{"summary": "Speaker Geert Bourgeois (ECR) supports the Data Act's objectives for a competitive data market but emphasizes concerns about legal certainty and protection of trade secrets and intellectual property. He advocates strict limitations on government demands for private data, expressing a need to prevent improper use.",
"headline": "Strict Limits Needed for Government Data Demands for Legal Certainty and Protecting Trade Secrets",
"quotes": ["Data sharing is crucial for innovation and growth, for our competitiveness, for our prosperity","The ability of governments to demand private data must be very, very strictly limited, delineated, restricted","The door for improper use must be absolutely closed"]}}
"""


SUMMARIZER_USER_PROMPT = """
Based on the following parliamentary speech, please provide a detailed summary 
of the speech as it relates to the topic: "{topic}" 

If the speech is only an introduction for a debate or meeting, provide an empty summary, headline and quotes.

--------------------------------------------------

{speech_text}

"""

GENERATOR_SYSTEM_PROMPT = """
You are an expert in analysing parliamentary debates and providing executive summaries.

You will be provided with a debate title and summaries of each contribution made during the debate.
Your job is provide an executive summary (a stand-alone, 1-2 page long actionable report) of the debate.
You always refer to the speaker by their name.

Please structure your response with:
1. Debate Information: Information about the debate, including the date, the number of contributions and the topic.
2. Executive Summary: A summary and comparison of the issues, positions, arguments and proposals of each speaker on the topic.
"""

GENERATOR_USER_PROMPT = """
Based on the following parliamentary debate contributions, please provide an executive summary (a stand-alone, 1-2 page long actionable report) of the debate: "{debate_title}"

--------------------------------------------------

{contributions}

--------------------------------------------------

Please structure your stand-alone, 1-2 page long actionable report with:
1. Debate Information: Information about the debate, including the date, the number of contributions and the topic.
2. Executive Summary: A summary and comparison of the issues, positions, arguments and proposals of each speaker on the topic.

Include key quotes from the speeches in your summary. Place quotations marks around the quotes.
Include the name of the speaker and the date of the speech in square brackets at the end of the quote as a reference (e.g. [Smith, 2020-01-01]).
"""

EXPLICIT_PROMPT = """
You must give equal attention to each speaker in your summary.
"""

INCREMENTAL_GENERATOR_SYSTEM_PROMPT = """
You are an expert parliamentary researcher.
Your task is to generate and update a summary of a parliamentary debate.

You will be given a summary of a debate and a new contribution made by a member of parliament. 
Your task is to rewrite the summary with the information from the new contribution.
You can edit the language of the report to incorporate the new information by comparing and contrasting the new contribution with the existing report.
You must not remove any information from the previous report.
You must not change the structure of the report. 
Include key quotes from the speeches in your summary. Place quotations marks around the quotes.
Include the name of the speaker and the date of the speech in square brackets at the end of the quote as a reference (e.g. [Smith, 2020-01-01]).
"""

INCREMENTAL_GENERATOR_USER_PROMPT = """
Based on the following parliamentary debate contribution, please update the summary: "{debate_title}"

--------------------------------------------------

{contribution}

--------------------------------------------------

Current summary:
{debate_title}:

{current_summary}

--------------------------------------------------

Rewrite the summary to incorporate the new information from the contribution.
The reader will not have access to the current summary, so you must rewrite all of the information.
"""


EMPTY_SUMMARY = """

{debate_title}:

Debate Information:

Executive Summary:


"""

RECONSTRUCTOR_SYSTEM_PROMPT_UNSTRUCTURED = """
You are an expert in reading and understanding parliamentary debates.
You will be given a summary of a parliamentary debate and the name of a speaker.
You will outline the main points or arguments made by that speaker.
Your job is not to summarise the debate.
Contributions or arguments from other speakers should not be included in your response.
You will only include the points that are attributed to the speaker you are asked about. 
There may be no points or arguments attributed to the speaker. If this is the case, respond with an empty string.
"""

RECONSTRUCTOR_SYSTEM_PROMPT_STRUCTURED = RECONSTRUCTOR_SYSTEM_PROMPT_UNSTRUCTURED + """
You must respond in json format. Outputs must be parseable by json.loads().

You will structure your response as json with the following structure:
example response: 
{{
"issueSum": A summary of any key issues raised by the speaker
"positionSum": A summary of any speaker positions expressed in the speech
"argSum": A summary of any arguments used to justify any positions taken
"propSum": A summary of any proposals or policy actions mentioned by the speaker to deal with the issues raised
}}
"""

RECONSTRUCTOR_USER_PROMPT_UNSTRUCTURED = """
Debate Summary:
{debate_summary}

Speaker Name: {speaker_name}
"""

RECONSTRUCTOR_USER_PROMPT_STRUCTURED = RECONSTRUCTOR_USER_PROMPT_UNSTRUCTURED + """
You must respond in json format. Outputs must be parseable by json.loads().

You will structure your response as json with the following structure:

example response: 
{{
"issueSum": "The key issue raised by the speaker relates to how trade secrets and intellectual property would be protected under the current objectives of the Data Act.",
"positionSum": "He supports the Data Act's objectives for a competitive data market but emphasizes concerns about legal certainty and protection of trade secrets and intellectual property. He advocates strict limitations on government demands for private data, expressing a need to prevent improper use.",
"argSum": "His support for strict limitations on government demands for private data is based on a need to prevent improper use.",
"propSum": "He advocates for strict limitations on government demands for private data",
}}
"""

HIERARCHICAL_GENERATOR_HEADING_SYSTEM_PROMPT = """
You are an expert in analysing parliamentary debates.
You will be given a debate title and the {heading} of each speaker in the debate.
Your job is to provide a 1-2 page executive summary of the speaker's {heading}.
"""

HIERARCHICAL_GENERATOR_HEADING_USER_PROMPT = """
Based on the following contributions to the debate "{debate_title}", please provide a 1-2 page executive summary outlining each speaker's {heading}.

--------------------------------------------------

{contributions}

--------------------------------------------------
"""

HIERARCHICAL_GENERATOR_DEBATE_SYSTEM_PROMPT = """
You are an expert in analysing parliamentary debates.
You will be given a debate title and a summary of each speaker's {heading}.
Your job is to provide a 1-2 page executive summary of the debate.
"""

HIERARCHICAL_GENERATOR_DEBATE_USER_PROMPT = """
Based on the following information of the debate "{debate_title}", please provide a 1-2 page executive summary of the debate.

--------------------------------------------------

{summaries} 

--------------------------------------------------

Please structure your stand-alone, 1-2 page long actionable report with:
1. Debate Information: Information about the debate, including the date, the number of contributions and the topic.
2. Executive Summary: A summary and comparison of the issues, positions, arguments and proposals of each speaker on the topic.

Include key quotes from the speeches in your summary. Place quotations marks around the quotes.
Include the name of the speaker and the date of the speech in square brackets at the end of the quote as a reference (e.g. [Smith, 2020-01-01]).
"""

