# A-Feasibility-Aware-LLM-Framework-for-Navigation
To use the code, download Carla UE4 at https://carla.readthedocs.io/en/latest/download/

Then replace PythonAPI folder with the contents in the repository

#
The main contents in under PythonAPI/examples, the rest of files are all under this directory

Check llmutil/llmutil.py first and set API key in system environment (or directly type in)

manual_control.py is used for planning for 1 agent in manual control scenario, this file is modifed based on the one in carla example

multiagent.py is used for running the test in the observer view, this contains the main test scenarios including multi-agent communication and multi-task planning. Task configuration is at the start of function main(), task and input settings can be changed here.
