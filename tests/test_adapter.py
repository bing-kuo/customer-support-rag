import asyncio
import pytest
from app.ollama import Adapter,BudgetError,CapacityError,DependencyError


def test_context_budget_rejects_before_upstream():
    a=Adapter(historical_test=True)
    with pytest.raises(BudgetError):
        asyncio.run(a.generate('問題',[{'source_id':'test','text':'字'*1000}],[]))
    with pytest.raises(BudgetError):
        asyncio.run(a.embed(['字'*1000]))


def test_capacity_and_open_circuit():
    a=Adapter(historical_test=True);a.pending=3
    with pytest.raises(CapacityError):
        asyncio.run(a.generate('question',[],[]))
    import time
    a.open_until=time.monotonic()+30
    with pytest.raises(DependencyError):
        asyncio.run(a.generate('question',[],[]))


def test_total_budget_includes_system_schema_output_and_template():
    async def scenario():
        with pytest.raises(BudgetError):
            await Adapter(historical_test=True).generate('a'*2000,[{'source_id':'one','text':'b'*1300}],[])
    asyncio.run(scenario())
