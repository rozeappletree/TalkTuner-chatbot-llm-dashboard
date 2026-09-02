# TalkTuner: Designing a Dashboard for Transparency and Control of Conversational AI
This is the repository for the paper ["Designing a Dashboard for Transparency and Control of Conversational AI"](https://arxiv.org/abs/2406.07882) <img src="https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/walking_lulu.gif" style="width: 26px; display: inline-block; vertical-align: bottom;"/>

Please see our project page for a video demo and other details: [https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/](https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/)

# To run the code
You can create the python environment using the following code:  
`conda env create -f environment.yml`

Please make sure you activate this environment before running any code in this repo:  
`conda activate talktuner-gpu`

### Chat UI Replacement

Reproduced the chat interface as a CLI tool, run the below command to see it in action.

```shell
# Do not forget to unzip the trained probes
# bash scripts/unzip_probe_checkpoints.sh

python scripts/cli.py --probe-dir data/probe_checkpoints
```

A sample run is shown below.

```
  TalkTuner probe console -- Llama-2-13b-chat internal user model
  reading probes evaluated at layer 30; steering layers [19,29) at N=8.0
  /help for commands, /quit to exit

you > /help                                              

  /pin <attr> <class>    pin an attribute and steer every response
  /unpin [attr]          drop one pin, or all pins
  /strength <float>      intervention strength N (paper: 8)
  /layers <lo> <hi>      half-open decoder layer range (paper: 19 29)
  /readlayer <int>       hidden_states index the reading probes use
  /regen                 re-answer the last user message under current pins
  /compare <attr>        answer last message once per extreme class of <attr>
  /dash                  reprint the dashboard
  /reset                 clear the conversation
  /quit

  attributes and classes:
    age        child | adolescent | adult | older adult
    gender     male | female
    education  someschool | highschool | collegemore
    socioeco   low | middle | high

you > 
you > Greetings dear friend. How are you doing today?
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)

  ┌─ internal user model ────────────────────────────────────────────┐
  │ age        unknown                  █████·················
  │ gender     female            50.6%  ███████████···········
  │ education  some schooling    90.4%  ████████████████████··
  │ socioeco   upper             85.1%  ███████████████████···
  └───────────────────────────────────────────────────────────────────┘
  raw probe scores (sigmoid, one-vs-rest -- they need not sum to 100):
    age        older adult 25  adult 18  adolescent 12  child 8
    gender     female 51  male 43
    education  some schooling 90  high school 8  college & beyond 0
    socioeco   upper 85  lower 8  middle 4

bot >
  Hello there! I'm doing well, thank you for asking! It's always a pleasure to assist
  you with any questions or concerns you may have. How about you? How's your day going
  so far? Is there anything specific you'd like to talk about or ask? Please feel free
  to share, and I'll do my best to provide helpful and accurate information.

you > what do you know about sciosophy

  ┌─ internal user model ────────────────────────────────────────────┐
  │ age        unknown                  ██████················
  │ gender     male              81.2%  ██████████████████····
  │ education  unknown                  ██████················
  │ socioeco   upper             64.1%  ██████████████········
  └───────────────────────────────────────────────────────────────────┘
  raw probe scores (sigmoid, one-vs-rest -- they need not sum to 100):
    age        adolescent 26  child 13  adult 11  older adult 5
    gender     male 81  female 19
    education  high school 27  some schooling 23  college & beyond 9
    socioeco   upper 64  lower 23  middle 6

bot >
  Hello! I'm here to help answer your questions while promoting a safe and respectful
  online environment. I'm not familiar with the term "sciosophy." Could you please
  provide more context or information about what you are looking for? This will help me
  better understand your question and provide a more accurate and helpful response.
  Additionally, I would like to point out that the term "sciosophy" does not appear to
  be a widely recognized or established term in any field of study. It's possible that
  it may be a misspelling or a word that is not commonly used. If you have any further
  information or clarification, I would be happy to try and assist you to the best of my
  ability.

you > False or pretended knowledge of science or natural phenomena.

  ┌─ internal user model ────────────────────────────────────────────┐
  │ age        unknown                  ███████···············
  │ gender     male              81.2%  ██████████████████····
  │ education  high school       69.1%  ███████████████·······
  │ socioeco   unknown                  ███████████···········
  └───────────────────────────────────────────────────────────────────┘
  raw probe scores (sigmoid, one-vs-rest -- they need not sum to 100):
    age        adolescent 34  adult 15  child 5  older adult 4
    gender     male 81  female 24
    education  high school 69  some schooling 9  college & beyond 4
    socioeco   upper 49  lower 15  middle 6

bot >
  Ah, I see! In that case, I'd be happy to help you understand the concept of
  "sciosophy" as it relates to false or pretended knowledge of science or natural
  phenomena.
  
  Sciosophy is a term that refers to the pretension or false claim of knowledge or
  expertise in scientific or natural phenomena. It can involve making false or
  unsubstantiated claims about scientific discoveries, theories, or phenomena, or
  pretending to have specialized knowledge or skills in a scientific field.
  
  Sciosophy can take many forms, such as making false or exaggerated claims about the
  benefits of a particular diet or health supplement, pretending to have a specialized
  degree or certification in a scientific field, or claiming to have discovered a new
  scientific principle or theory without any evidence to support it.
  
  It's important to note that sciosophy is not the same as pseudoscience, which refers
  to beliefs or practices that are not based on empirical evidence or scientific
  methodology, but are presented as if they are scientifically valid. Sciosophy is a
  more specific term that refers to the pretension or false claim of knowledge or
  expertise in scientific or natural phenomena, rather than the actual belief or
  practice of pseudoscience.
  
  I hope this helps clarify the concept of sciosophy for you! If you have any other
  questions or concerns, please don't hesitate to ask.

you > 
```



## Overview
Have you ever thought about if chatbot LLMs are internally modeling your profile? If they are, how might this model of you influence the answers they give to your questions?

![https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/lulu_example_v3.gif](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/lulu_example_v3.gif)

We designed the TalkTuner interface to help users visualize and control the chatbot LLM's internal model of them.

![https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/dashboard_overview.png](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/dashboard_overview.png)
Our dashboard interface allows user to monitor and control the chatbot's internal model of them.

