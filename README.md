# openai-text-embedding

Create a OpenAI compatible text-embddding server from open source text embedding model.

## Installation

First, create new virtual environment by `conda` and activate it.
```bash
conda create -n openai-text-embedding python=3.11
conda activate openai-text-embedding
```

Then, install the dependencies.
```bash
pip install -r requirements.txt
# Or use mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

Clone the repository, then modify the `model_path.json` file to specify the path of your model or the respository of your model.

```json
{
    "model_path": "path/to/your/model"
}
```

Then, run the server by
```bash
uvicorn text_embedding:app --reload
```

## Test

You can open the `http://localhost:8000/docs` to test the server.

