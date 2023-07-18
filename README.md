# 🦜️🔗 Marvin Chat Agent

This repository contains PlatoAI implementations of various LangChain solutions as Streamlit apps including:

- `marvin_chat_agent.py`: A chat agent with search (requires setting `OPENAI_API_KEY` env to run)
- `workflow_manager.py`: A manager for data workflow management (require setting [passwords] and username = "password" in secrets.toml) #WIP

Apps feature LangChain 🤝 Streamlit integrations such as the
[Callback integration](https://python.langchain.com/docs/modules/callbacks/integrations/streamlit).

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```shell
# Create Python environment
$ poetry install

# Install git pre-commit hooks
$ poetry shell
$ pre-commit install
```

## Running

```shell
# Run mrkl_demo.py or another app the same way
$ streamlit run streamlit_agent/mrkl_demo.py
```

## Contributing

We plan to add more agent examples over time - PRs welcome

- [ ] Chat QA over docs
- [ ] SQL agent
