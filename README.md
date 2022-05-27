# Introduction 
This repository contains the content for the "Practical aspects of reinforcement learning" workshop at MLPrague 2022.  

# Content and Authors

Practical aspects of reinforcement learning: Build your own elements of contextual bandits in TF-Agents

Michal Kubišta, Dataclair.ai; Petr Stanislav, Dataclair.ai

Reinforcement learning (RL) models are a new type of intelligent machine that can help you drive your car or beat you in Starcraft. There are many RL libraries and they implement a wide range of policies (models) environments and other elements of RL. However they can never cover all use cases and you might quickly find out you need to build your pieces to make the package work for your project. This decision likely leads to scarcely documented protocols and interfaces you need to fulfil and this is where we want to help. Since implementing the full reinforcement learning solutions in a business setup (outside of the typical use cases with simulated environments) leads to additional complexities this workshop will focus on contextual multi-armed bandits (CMAB) a middle step between supervised and reinforcement learning.&nbsp;We will first review all building blocks of the RL / CMAB framework and then walk you through building a custom implementation of those elements which will include a lot of code running on tf.Graph. After this session you should understand the (dis)advantages of using CMAB and be ready to start using TF-Agents in your projects.

# Getting Started
To ensure you are ready for the workshop kindly follow these steps:
1.	Setup your environment for Python 3.8 (newer versions should be OK as well)
    - it generally shouldn't matter which tool you use
      - we have tested virtualenv and conda
    - based on your tool of choice, use **pip** `requirements.txt` or **conda** `environment.yml` and install the required packages
1. Prepare the data
   - either transform them on your own using command - `make dataset`
   - or download preprocessed data using command - `make blob`

> On M1 Macs you must replace `tensorflow` with `tensorflow-macos` and remove `tensorflow-io-gcs-filesystem` from `requirements.txt`. More info [here](https://developer.apple.com/metal/tensorflow-plugin/).

> For running in Google Colab replace first cell in notebooks with content from `src/practice/colab_settings.ipynb`.

# Contribute
Should you encounter any problem, or find any errors (typos, bugs), feel free to open an issue or PR directly.
We use [black](https://black.readthedocs.io/en/stable/) for code formatting (including notebooks), so please ensure your code was formatted the same way before submitting it.


While we haven't prepared any code of conduct for this project, we expect everybody can understand and behave according to a general set of rules, such as:
- DO be respectul and show empathy
- DO gracefully accept constructive feedback
- DON'T harrass anyone
- ...
