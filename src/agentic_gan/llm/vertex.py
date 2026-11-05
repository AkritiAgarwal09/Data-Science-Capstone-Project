"""Gemini on Vertex through its OpenAI-compatible endpoint."""

import json
import os

import google.auth
from google.auth.transport.requests import Request
from openai import OpenAI
from pydantic import BaseModel


class VertexClient:
    def __init__(self):
        self.project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        self.model = os.environ.get("VERTEX_MODEL")
        if not all((self.project, self.location, self.model)):
            raise ValueError("Set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and VERTEX_MODEL")
        self.credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def complete(self, system_prompt: str, context: str, output_model: type[BaseModel]):
        self.credentials.refresh(Request())
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{self.location}-aiplatform.googleapis.com"
        )
        with OpenAI(
            base_url=(
                f"https://{host}/v1/projects/{self.project}/locations/"
                f"{self.location}/endpoints/openapi"
            ),
            api_key=self.credentials.token,
            max_retries=0,
        ) as client:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_model.__name__,
                        "schema": output_model.model_json_schema(),
                        "strict": True,
                    },
                },
                temperature=0,
            )
        return json.loads(response.choices[0].message.content)
