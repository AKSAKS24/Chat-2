from fastapi import APIRouter
from ..llm.provider_registry import list_providers_and_models
from ..services.agent_registry import list_agents
from ..schemas import ProviderModelsResponse, AgentsListResponse, AgentInfo

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/models", response_model=ProviderModelsResponse)
async def get_models():
    return {"data": list_providers_and_models()}


@router.get("/agents", response_model=AgentsListResponse)
async def get_agents():
    agents = list_agents()
    return {"data": [AgentInfo(**a) for a in agents]}
