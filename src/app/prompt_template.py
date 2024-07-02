from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

key_points_template = ChatPromptTemplate.from_messages([
    ("system",  "You are a proficient AI with a specialty in distilling information \
        into key points. Based on the following text, identify and list the main\
        points that were discussed or brought up. These should be the most \
        important ideas, findings, or topics that are crucial to the essence \
        of the discussion. Your goal is to provide a list that someone\
        could read to quickly understand what was talked about.\
        Each response key point MUST be INSIDE <kps> </kps> tag."),
    ("human", "{dumps}"),
])

action_items_template = ChatPromptTemplate.from_messages([
    ("system",  "You are an AI expert in analyzing conversations and extracting action items. \
                Please review the text and identify any tasks, assignments, or actions that were agreed \
                upon or mentioned as needing to be done. These could be tasks assigned to specific individuals,\
                or general actions that the group has decided to take. Please list these action items clearly and concisely."),
    ("human", "{dumps}"),
    # Means the template will receive an optional list of messages under
    # the "conversation" key
    # ("placeholder", "{conversation}")
    # Equivalently:
    # MessagesPlaceholder(variable_name="conversation", optional=True)
])

key_points_template = PromptTemplate