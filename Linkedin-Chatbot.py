"""
You want to send a response for referral request from people. Some people can try to ask you for a 
a call to understand more about role and some people just want a referral. 
Sometimes a recruiter might want to reach out.
or a person may just want to ask you to join their entreprenueurship or ask you for sensitive details: 
do nothing for that!
"""


"""
Flows:

1. Referral --> send an email and ask them for job_id and title .
2. Meeting setup --> send an email with availability for next saturday between 2:30pm to 3:30pm 
3. Recruiter --> Send an email telling how much you liked that tehey reached out and send for connect the following monday.
4. Scammer--> Send an email, dude you are scammy!

"""

import datetime
from pydantic import BaseModel,Field
from openai import OpenAI
from dotenv import load_dotenv
import json

### load your API keys here ##
load_dotenv('.env')
your_name="XYZ"


## create a client
client = OpenAI()
model="gpt-4o"
############### DATA MODELS ###################

class EmailInfo(BaseModel):
    description:str = Field(description="Fetch the summary of message without sender's name")
    sender_name: str= Field(description="Fetch the name of sender")

class EmailReferral(BaseModel):
    is_type:bool = Field(description="check if the text is asking for a referral")
    confidence_score:float = Field(description="confidence score between 0 and 1")

class EmailRecruiter(BaseModel):
    is_type:bool = Field(description="check if the text is froma recruiter who is sending an email regarding job oppportunity at their company.")
    confidence_score:float = Field(description="confidence score between 0 and 1")

class EmailMeetingSetup(BaseModel):
    is_type:bool = Field(description="check if the text is asking for a meeting setup for general connect and not a job opportunity")
    confidence_score:float = Field(description="confidence score between 0 and 1")

class EmailScammer(BaseModel):
    is_type:bool = Field(description="check if the text is asking for joining an entrepreneurship opportunity or is asking for sensitive details")
    confidence_score:float = Field(description="confidence score between 0 and 1")


class EmailCreation(BaseModel):
    email_description:str = Field(description=f"draft an mail below 50 words for sender with given details and send the email from {your_name} with regards.")


############## HELPER FUNCTIONS ###############
def enrich_initial_context(type_of_model:str,user_message:str):
    today_date_context = f"Today the date is {datetime.datetime.now()}."
    ## extract the message description and sender_name
    email_information=get_email_information(user_message)
    enrich_context_mapping={
        "EmailReferral": f"Send an email to {email_information.sender_name} asking the person the job_id they are interested in along with your email address to refer them.",
        "EmailMeetingSetup": f"{today_date_context}.Send an email to {email_information.sender_name}  by telling you are happy to connect and can meet next coming saturday date at 12pm Noon EST.",
        "EmailRecruiter": f"{today_date_context}. Send an email to {email_information.sender_name} by telling you are very excited to know more about the opportunity and can meet next coming tuesday date at 2pm EST.",
        "EmailScammer": f"Send an email to {email_information.sender_name} telling how bad it is to scam people on linkedin and you will report them for this behavior."
    }
    return enrich_context_mapping[type_of_model]

def get_email_information(user_message:str)->EmailInfo:
    response = client.beta.chat.completions.parse(
            model=model,
            messages=[{
                "role":"system",
                "content":"Fetch the details from user message"
            },
            {
                "role":"user",
                "content":user_message
            },
            ],
            response_format=EmailInfo
        )
    return response.choices[0].message.parsed

def get_email(enriched_context:str) -> EmailCreation:
    response=client.beta.chat.completions.parse(
            model=model,
            messages=[{
                "role":"system",
                "content":"Draft an email based on user message"
            },
            {
                "role":"user",
                "content":enriched_context
            },
            ],
            response_format=EmailCreation
        )
    return response.choices[0].message.parsed
    


def process_linkedin_inbound_message(user_message:str):
    kinds_of_message_types = [EmailMeetingSetup,EmailRecruiter,EmailReferral,EmailScammer]
    max_confidence_score=0.0
    max_message_type=None
    for message_kind in kinds_of_message_types:
        current_parse=client.beta.chat.completions.parse(
            model=model,
            messages=[{
                "role":"system",
                "content":"analyze the message"
            },
            {
                "role":"user",
                "content":user_message
            },
            ],
            response_format=message_kind
        )
        current_response = current_parse.choices[0].message.parsed
        if current_response.is_type and current_response.confidence_score>max_confidence_score:
            max_confidence_score=current_response.confidence_score
            max_message_type=message_kind.__name__
    print(f"The input was identified as {max_message_type} with confidence_score of {max_confidence_score}!")
    #### now call the function to enrich the user prompt
    enriched_context=enrich_initial_context(max_message_type,user_message)

    ### get an email design
    created_email = get_email(enriched_context=enriched_context)
    
    print(created_email.email_description)



############# EXAMPLES ########################
message_1 = """
Hi XYZ,
Hope you're doing well. This is ABC, I just graduated with a MS in Data Science, I've 2 years of experience in business intelligence & data analytics. I recently came across a "junior Data Analyst" position at XYZ and am reaching out to get a referral for my application.
"""

message_2="""
I work as a QA Manager, managing the team business integration. At the same time, my partner and I are working on online entrepreneurship opportunities. We are expanding our team and selecting candidates with an entrepreneurial spirit for partnership. Are you considering the possibility of using your skills in your free time to become an entrepreneur and diversify your income?
"""


message_3="""
Hi XYZ
Are you available in job market? I have a below job please let me know if you anyone you know is available

Job Title: Data Scientist-Artificial Intelligence
Location: Middletown, NJ
Duration: 03 Mar 2025 - 05 Sep 2025
Pay Rate : $Please let me know your best rate on W2 
Travel Type: On-site (no expenses)

Please share your updated resume. I will call you to discuss more

Thank you.
"""

message_4="""
I have recently graduated and I'm looking for quant finance roles. I see that you're working on the data side of the quant finance group at UBS. I would be interested in knowing more about your role, and your suggestions on job search.
Can we connect over Zoom, sometime this week?
👏
👍
😊



Thanks,
ryan
"""

message_5="""
Hi XYZ,

I hope you are doing well.

My name is Tom, and I am a Data Scientist at Bank of America. I am exploring new Data Scientist opportunities and came across your Linkedin post and would love to explore any suitable openings at UBS.

A brief overview of my background:
*Education*: B.Tech. from 4d Roorkee (2017-2021).
*Experience*: 4 years in d Science, with expertisf4e in Fraud D4fetection, Model Monitoring, machine learning, deep learning, natural language processing (NLP), GenAI, time series analysis, and statistical modelling.
*Technical Skills*: Proficient in Python, Py4fSpark, 4, SQL, and experience with fine tuning of LLMs, GenAI, and various machine learning algorithms (e.g., SVM, Random Forest, ANN, CNN, RNN, Unsupervised ML, PCA). I have exposure to Keras, TensorFlow, SAS, and Scikit-learn.

I would be happy to provide any additional information or answer any questions.

Thanks and Regards,
Tom
"""

message_6="""
Hi XYZ,

Thank you for applying for our Risk Developer role at 4d Capital. We were impressed with your skillsets and experience and would like to schedule an initial screening interview to get to know you better and to share more about Balbec. The call will last approximately 30-minutes and will be conducted as a video call via MS Teams. Below is my calendar availability, please take a look and select a time that works best for you. If there are no available times aligned, please respond with your availability and I will do my best to work with you. 

https://calendly.com/de4d4ed/risk-developer-interview-4-4d

I look forward to connecting with you soon.

Best,
Tom Doe, Recruiting Coordinator

Tom Doe
Recruiting Coordinator at d3 Capital LP | Campus Recruiter | Passionate People Person
"""
process_linkedin_inbound_message(message_6)