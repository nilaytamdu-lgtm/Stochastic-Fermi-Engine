import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from .engine import FermiComponent

class ComponentSchema(BaseModel):
    name: str = Field(description="Short name of the factor (e.g., 'Total Population' or 'Regulatory Approval')")
    factor_type: str = Field(description="Strictly output either 'continuous' (for amounts/numbers) or 'binary' (for yes/no risks)")
    p10: float = Field(description="The 10th percentile estimate (pessimistic)")
    p90: float = Field(description="The 90th percentile estimate (optimistic)")
    deterministic_point: float = Field(description="The point estimate or probability (0.0 to 1.0 if binary)")
    explanation: str = Field(description="Brief justification of these bounds")
    linked_factors: List[str] = Field(description="List of exact names of other factors that would crash if this factor crashes (leave empty if independent)")

class FermiEstimationSchema(BaseModel):
    components: List[ComponentSchema] = Field(description="The multiplicative chain of factors")

class ElicitorAgent:
    def __init__(self):
        self.client = genai.Client()
        
    def elicit(self, prompt: str) -> List[FermiComponent]:
        system_instruction = (
            "You are an expert management consultant. You decompose market sizing problems into a multiplicative Fermi tree. "
            "For each factor, provide bounds (p10 and p90). "
            "CRITICAL INSTRUCTION: Before you estimate the pessimistic (p10) bound, explicitly imagine the absolute worst-case apocalyptic scenario to reset your anchoring bias. "
            "If a factor is a yes/no risk (like winning a contract), mark it 'binary' and set p10=0, p90=1, and deterministic_point to the success probability (e.g. 0.5). "
            "List any dependencies in 'linked_factors'."
        )
        
        response = self.client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': FermiEstimationSchema,
                'system_instruction': system_instruction,
                'temperature': 0.3
            }
        )
        
        data = json.loads(response.text)
        
        fermi_components = []
        for comp in data.get('components', []):
            fermi_components.append(
                FermiComponent(
                    name=comp['name'],
                    factor_type=comp.get('factor_type', 'continuous'),
                    p10=float(comp['p10']),
                    p90=float(comp['p90']),
                    deterministic_point=float(comp['deterministic_point']),
                    explanation=comp['explanation'],
                    linked_factors=comp.get('linked_factors', [])
                )
            )
        
        return fermi_components
