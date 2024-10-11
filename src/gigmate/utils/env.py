from dotenv import dotenv_values

config = dotenv_values()


def get_env_var(name: str):
    return config.get(name)
