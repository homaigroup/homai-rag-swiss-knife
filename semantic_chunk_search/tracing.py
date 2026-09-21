from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass,field
import time
from typing import Any,Callable,Iterator,Protocol

@dataclass(slots=True)
class TraceSpan:
    name:str;duration_ms:float;attributes:dict[str,Any]=field(default_factory=dict)

@dataclass(slots=True)
class SearchTrace:
    spans:list[TraceSpan]=field(default_factory=list);attributes:dict[str,Any]=field(default_factory=dict)
    @contextmanager
    def span(self,name:str,**attributes:Any)->Iterator[dict[str,Any]]:
        state=dict(attributes);start=time.perf_counter()
        try:yield state
        finally:self.spans.append(TraceSpan(name,(time.perf_counter()-start)*1000.0,state))
    def as_dict(self)->dict[str,Any]:return {"attributes":dict(self.attributes),"spans":[{"name":s.name,"duration_ms":s.duration_ms,"attributes":s.attributes} for s in self.spans],"total_ms":sum(s.duration_ms for s in self.spans)}

class TraceSink(Protocol):
    def __call__(self,trace:SearchTrace)->None:...

class OpenTelemetryTraceSink:
    """Optional bridge to an application's existing OpenTelemetry provider.

    Stored stage timings are emitted as span events/attributes; the package does not
    install or mutate the global tracer provider.
    """
    def __init__(self,tracer_name:str="semantic_chunk_search")->None:
        try:from opentelemetry import trace
        except ImportError as exc:raise ImportError("Install opentelemetry-api to use OpenTelemetryTraceSink") from exc
        self.tracer=trace.get_tracer(tracer_name)
    def __call__(self,record:SearchTrace)->None:
        with self.tracer.start_as_current_span("semantic_chunk_search.search") as span:
            for key,value in record.attributes.items():
                if isinstance(value,(str,bool,int,float)):span.set_attribute(f"semantic_chunk_search.{key}",value)
            for stage in record.spans:
                attrs={"duration_ms":float(stage.duration_ms)}
                attrs.update({k:v for k,v in stage.attributes.items() if isinstance(v,(str,bool,int,float))})
                span.add_event(stage.name,attrs)
            span.set_attribute("semantic_chunk_search.total_ms",sum(s.duration_ms for s in record.spans))
