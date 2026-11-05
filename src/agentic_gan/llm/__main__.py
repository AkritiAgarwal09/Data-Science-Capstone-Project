"""Make one structured Vertex call to check ADC and model access."""

from pydantic import BaseModel

from .vertex import VertexClient


class AccessCheck(BaseModel):
    message: str


def main():
    response = VertexClient().complete(
        "Return a short confirmation of access as JSON.",
        "Reply with a message confirming this structured call succeeded.",
        AccessCheck,
    )
    print(AccessCheck.model_validate(response).message)


if __name__ == "__main__":
    main()
