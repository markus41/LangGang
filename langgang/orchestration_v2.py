"""
LangGraph Advanced Orchestration Module v2.0

This module implements 10 additional high-impact orchestration patterns:

1. Agent Health Monitoring & Circuit Breaker
2. Dynamic Agent Scaling
3. Distributed Tracing with OpenTelemetry
4. Semantic Caching for LLM Calls
5. Priority-Based Task Scheduling
6. Event Sourcing for State Changes
7. Workflow Versioning and Migration
8. Retry Policies with Exponential Backoff
9. Workflow Composition and Templates
10. Multi-Tenancy and Isolation

References:
- https://martinfowler.com/bliki/CircuitBreaker.html
- https://opentelemetry.io/docs/languages/python/
- https://martinfowler.com/eaaDev/EventSourcing.html
"""

import asyncio
import hashlib
import heapq
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. AGENT HEALTH MONITORING & CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """States for circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for agent health management.

    Implements the circuit breaker pattern to detect and isolate
    failing agents, preventing cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered
    """
    agent_id: str
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=30)
    half_open_max_calls: int = 3

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    half_open_calls: int = field(default=0)

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif self.failure_count >= self.failure_threshold:
            self._open()

    def can_execute(self) -> bool:
        """Check if circuit allows execution."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._half_open()
                return True
            return False

        # HALF_OPEN: allow limited calls
        if self.half_open_calls < self.half_open_max_calls:
            self.half_open_calls += 1
            return True
        return False

    def _open(self) -> None:
        """Open the circuit (reject all requests)."""
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        logger.warning(f"Circuit breaker OPENED for agent {self.agent_id}")

    def _close(self) -> None:
        """Close the circuit (normal operation)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker CLOSED for agent {self.agent_id}")

    def _half_open(self) -> None:
        """Set circuit to half-open (testing recovery)."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.half_open_calls = 0
        logger.info(f"Circuit breaker HALF-OPEN for agent {self.agent_id}")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time > self.recovery_timeout

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


class AgentHealthMonitor:
    """Monitors agent health and manages circuit breakers.

    Provides:
    - Circuit breaker management per agent
    - Health check endpoints
    - Automatic fallback to backup agents
    - Health status reporting
    """

    def __init__(self):
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._health_checks: Dict[str, Callable] = {}
        self._fallback_agents: Dict[str, List[str]] = {}
        self._health_history: Dict[str, List[Dict]] = {}

    def register_agent(
        self,
        agent_id: str,
        health_check: Optional[Callable] = None,
        fallbacks: Optional[List[str]] = None,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(seconds=30)
    ) -> None:
        """Register an agent with health monitoring.

        Args:
            agent_id: Unique identifier for the agent
            health_check: Optional async function to check health
            fallbacks: List of fallback agent IDs
            failure_threshold: Failures before opening circuit
            recovery_timeout: Time before attempting recovery
        """
        self._circuits[agent_id] = CircuitBreaker(
            agent_id=agent_id,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        if health_check:
            self._health_checks[agent_id] = health_check
        if fallbacks:
            self._fallback_agents[agent_id] = fallbacks
        self._health_history[agent_id] = []

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from health monitoring."""
        self._circuits.pop(agent_id, None)
        self._health_checks.pop(agent_id, None)
        self._fallback_agents.pop(agent_id, None)
        self._health_history.pop(agent_id, None)

    def get_circuit(self, agent_id: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for an agent."""
        return self._circuits.get(agent_id)

    def is_healthy(self, agent_id: str) -> bool:
        """Check if an agent is healthy (circuit closed or half-open)."""
        circuit = self._circuits.get(agent_id)
        if not circuit:
            return False
        return circuit.state != CircuitState.OPEN

    async def execute_with_fallback(
        self,
        agent_id: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with circuit breaker and fallback.

        Args:
            agent_id: Primary agent to use
            operation: Async operation to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Operation result

        Raises:
            RuntimeError: If all agents (primary + fallbacks) fail
        """
        agents_to_try = [agent_id] + self._fallback_agents.get(agent_id, [])

        last_error = None
        for aid in agents_to_try:
            circuit = self._circuits.get(aid)
            if not circuit:
                continue

            if not circuit.can_execute():
                logger.debug(f"Circuit open for {aid}, trying next")
                continue

            try:
                result = await operation(aid, *args, **kwargs)
                circuit.record_success()
                self._record_health(aid, True)
                return result
            except Exception as e:
                circuit.record_failure()
                self._record_health(aid, False, str(e))
                last_error = e
                logger.warning(f"Agent {aid} failed: {e}")

        raise RuntimeError(
            f"All agents failed for {agent_id}. Last error: {last_error}"
        )

    def _record_health(self, agent_id: str, healthy: bool, error: str = None) -> None:
        """Record health check result."""
        if agent_id not in self._health_history:
            self._health_history[agent_id] = []

        self._health_history[agent_id].append({
            "timestamp": datetime.now().isoformat(),
            "healthy": healthy,
            "error": error
        })

        # Keep only last 100 entries
        if len(self._health_history[agent_id]) > 100:
            self._health_history[agent_id] = self._health_history[agent_id][-100:]

    async def run_health_checks(self) -> Dict[str, bool]:
        """Run health checks on all registered agents.

        Returns:
            Dict mapping agent_id to health status
        """
        results = {}
        for agent_id, check in self._health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check):
                    result = await check()
                else:
                    result = check()
                results[agent_id] = bool(result)
                self._record_health(agent_id, results[agent_id])
            except Exception as e:
                results[agent_id] = False
                self._record_health(agent_id, False, str(e))
        return results

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers."""
        return {
            agent_id: circuit.get_status()
            for agent_id, circuit in self._circuits.items()
        }

    def get_health_history(self, agent_id: str, limit: int = 10) -> List[Dict]:
        """Get recent health history for an agent."""
        history = self._health_history.get(agent_id, [])
        return history[-limit:]


# =============================================================================
# 2. DYNAMIC AGENT SCALING
# =============================================================================

@dataclass
class ScalingConfig:
    """Configuration for dynamic scaling."""
    min_agents: int = 1
    max_agents: int = 10
    scale_up_threshold: int = 10      # Queue depth
    scale_down_threshold: int = 2
    processing_time_threshold: float = 5.0  # seconds
    error_rate_threshold: float = 0.1
    cooldown_period: timedelta = timedelta(seconds=60)


@dataclass
class AgentMetrics:
    """Metrics for scaling decisions."""
    queue_depth: int = 0
    avg_processing_time: float = 0.0
    error_rate: float = 0.0
    active_agents: int = 0
    total_processed: int = 0
    total_errors: int = 0
    last_scale_time: Optional[datetime] = None


class DynamicScaler:
    """Manages dynamic agent scaling based on workload.

    Monitors:
    - Queue depth
    - Processing time
    - Error rate

    Actions:
    - Scale up when load increases
    - Scale down when load decreases
    - Respects cooldown between scaling operations
    """

    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.metrics = AgentMetrics()
        self._agent_pool: List[str] = []
        self._agent_factory: Optional[Callable] = None
        self._agent_destroyer: Optional[Callable] = None
        self._scaling_history: List[Dict] = []

    def set_agent_factory(self, factory: Callable[[str], Any]) -> None:
        """Set factory function for creating new agents.

        Args:
            factory: Async function that takes agent_id and creates an agent
        """
        self._agent_factory = factory

    def set_agent_destroyer(self, destroyer: Callable[[str], Any]) -> None:
        """Set destroyer function for removing agents.

        Args:
            destroyer: Async function that takes agent_id and removes an agent
        """
        self._agent_destroyer = destroyer

    def update_metrics(
        self,
        queue_depth: int,
        processing_time: float,
        error_count: int,
        total_count: int
    ) -> None:
        """Update metrics for scaling decisions.

        Args:
            queue_depth: Current number of pending tasks
            processing_time: Average processing time in seconds
            error_count: Number of errors
            total_count: Total operations processed
        """
        self.metrics.queue_depth = queue_depth
        self.metrics.avg_processing_time = processing_time
        self.metrics.total_errors = error_count
        self.metrics.total_processed = total_count
        self.metrics.error_rate = error_count / max(total_count, 1)
        self.metrics.active_agents = len(self._agent_pool)

    def should_scale_up(self) -> bool:
        """Determine if we should add agents."""
        if len(self._agent_pool) >= self.config.max_agents:
            return False

        if not self._cooldown_elapsed():
            return False

        return (
            self.metrics.queue_depth > self.config.scale_up_threshold or
            self.metrics.avg_processing_time > self.config.processing_time_threshold or
            self.metrics.error_rate > self.config.error_rate_threshold
        )

    def should_scale_down(self) -> bool:
        """Determine if we should remove agents."""
        if len(self._agent_pool) <= self.config.min_agents:
            return False

        if not self._cooldown_elapsed():
            return False

        return (
            self.metrics.queue_depth < self.config.scale_down_threshold and
            self.metrics.avg_processing_time < self.config.processing_time_threshold / 2 and
            self.metrics.error_rate < self.config.error_rate_threshold / 2
        )

    async def scale_up(self) -> Optional[str]:
        """Add a new agent to the pool.

        Returns:
            New agent ID or None if scaling failed
        """
        if not self._agent_factory:
            logger.warning("No agent factory configured")
            return None

        if len(self._agent_pool) >= self.config.max_agents:
            return None

        agent_id = f"agent_{len(self._agent_pool)}_{datetime.now().strftime('%H%M%S')}"

        try:
            if asyncio.iscoroutinefunction(self._agent_factory):
                await self._agent_factory(agent_id)
            else:
                self._agent_factory(agent_id)

            self._agent_pool.append(agent_id)
            self.metrics.last_scale_time = datetime.now()
            self._record_scaling("scale_up", agent_id)
            logger.info(f"Scaled up: added agent {agent_id}")
            return agent_id
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
            return None

    async def scale_down(self) -> Optional[str]:
        """Remove an agent from the pool.

        Returns:
            Removed agent ID or None if scaling failed
        """
        if len(self._agent_pool) <= self.config.min_agents:
            return None

        agent_id = self._agent_pool.pop()

        try:
            if self._agent_destroyer:
                if asyncio.iscoroutinefunction(self._agent_destroyer):
                    await self._agent_destroyer(agent_id)
                else:
                    self._agent_destroyer(agent_id)

            self.metrics.last_scale_time = datetime.now()
            self._record_scaling("scale_down", agent_id)
            logger.info(f"Scaled down: removed agent {agent_id}")
            return agent_id
        except Exception as e:
            # Re-add agent if destruction failed
            self._agent_pool.append(agent_id)
            logger.error(f"Failed to scale down: {e}")
            return None

    async def auto_scale(self) -> Optional[str]:
        """Automatically scale based on metrics.

        Returns:
            Agent ID if scaling occurred, None otherwise
        """
        if self.should_scale_up():
            return await self.scale_up()
        elif self.should_scale_down():
            return await self.scale_down()
        return None

    def _cooldown_elapsed(self) -> bool:
        """Check if cooldown period has elapsed."""
        if not self.metrics.last_scale_time:
            return True
        return datetime.now() - self.metrics.last_scale_time > self.config.cooldown_period

    def _record_scaling(self, action: str, agent_id: str) -> None:
        """Record scaling action."""
        self._scaling_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "agent_id": agent_id,
            "pool_size": len(self._agent_pool),
            "metrics": {
                "queue_depth": self.metrics.queue_depth,
                "processing_time": self.metrics.avg_processing_time,
                "error_rate": self.metrics.error_rate
            }
        })

    def get_pool(self) -> List[str]:
        """Get current agent pool."""
        return list(self._agent_pool)

    def get_scaling_history(self, limit: int = 20) -> List[Dict]:
        """Get recent scaling history."""
        return self._scaling_history[-limit:]


# =============================================================================
# 3. DISTRIBUTED TRACING
# =============================================================================

@dataclass
class Span:
    """A single span in a distributed trace.

    Represents a unit of work with timing and metadata.
    Compatible with OpenTelemetry export format.
    """
    span_id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    status: str = "OK"

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_error(self, error: Exception) -> None:
        """Mark span as errored."""
        self.status = "ERROR"
        self.attributes["error.type"] = type(error).__name__
        self.attributes["error.message"] = str(error)

    def to_dict(self) -> Dict:
        """Convert span to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status
        }


class Tracer:
    """Distributed tracing for multi-agent workflows.

    Provides:
    - Trace context propagation
    - Span creation and management
    - OpenTelemetry-compatible export
    - Parent-child span relationships
    """

    def __init__(self, service_name: str = "langgang"):
        self.service_name = service_name
        self._traces: Dict[str, List[Span]] = {}
        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []
        self._active_trace_id: Optional[str] = None

    def start_trace(self, name: str, attributes: Optional[Dict] = None) -> str:
        """Start a new trace.

        Args:
            name: Name for the root span
            attributes: Optional attributes for the root span

        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_id=None,
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )
        self._traces[trace_id] = [span]
        self._current_span = span
        self._span_stack = [span]
        self._active_trace_id = trace_id
        return trace_id

    def end_trace(self) -> Optional[str]:
        """End the current trace.

        Returns:
            Trace ID that was ended
        """
        if self._current_span:
            self._current_span.end_time = time.time()

        trace_id = self._active_trace_id
        self._current_span = None
        self._span_stack = []
        self._active_trace_id = None
        return trace_id

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict] = None) -> Generator[Span, None, None]:
        """Create a child span context manager.

        Args:
            name: Span name
            attributes: Optional span attributes

        Yields:
            The created span
        """
        if not self._current_span:
            raise RuntimeError("No active trace. Call start_trace() first.")

        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=self._current_span.trace_id,
            parent_id=self._current_span.span_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )

        self._traces[span.trace_id].append(span)
        self._span_stack.append(span)
        previous = self._current_span
        self._current_span = span

        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.end_time = time.time()
            self._span_stack.pop()
            self._current_span = previous

    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return self._current_span

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self._traces.get(trace_id, [])

    def get_trace_tree(self, trace_id: str) -> Dict:
        """Get trace as a tree structure."""
        spans = self.get_trace(trace_id)
        if not spans:
            return {}

        # Build tree
        span_map = {s.span_id: {"span": s.to_dict(), "children": []} for s in spans}

        root = None
        for span in spans:
            if span.parent_id is None:
                root = span_map[span.span_id]
            elif span.parent_id in span_map:
                span_map[span.parent_id]["children"].append(span_map[span.span_id])

        return root or {}

    def export_trace(self, trace_id: str) -> Dict:
        """Export trace in OpenTelemetry-compatible format."""
        spans = self.get_trace(trace_id)
        return {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service_name}}
                    ]
                },
                "scopeSpans": [{
                    "scope": {"name": "langgang.tracer"},
                    "spans": [
                        {
                            "traceId": s.trace_id,
                            "spanId": s.span_id,
                            "parentSpanId": s.parent_id or "",
                            "name": s.name,
                            "startTimeUnixNano": int(s.start_time * 1e9),
                            "endTimeUnixNano": int((s.end_time or time.time()) * 1e9),
                            "attributes": [
                                {"key": k, "value": {"stringValue": str(v)}}
                                for k, v in s.attributes.items()
                            ],
                            "events": [
                                {
                                    "name": e["name"],
                                    "timeUnixNano": int(e["timestamp"] * 1e9),
                                    "attributes": [
                                        {"key": k, "value": {"stringValue": str(v)}}
                                        for k, v in e.get("attributes", {}).items()
                                    ]
                                }
                                for e in s.events
                            ],
                            "status": {"code": 1 if s.status == "OK" else 2}
                        }
                        for s in spans
                    ]
                }]
            }]
        }

    def clear_trace(self, trace_id: str) -> None:
        """Clear a trace from memory."""
        self._traces.pop(trace_id, None)


# =============================================================================
# 4. SEMANTIC CACHING FOR LLM CALLS
# =============================================================================

@dataclass
class CacheEntry:
    """Entry in the semantic cache."""
    key_hash: str
    embedding: np.ndarray
    prompt: str
    response: str
    created_at: datetime
    hit_count: int = 0
    ttl: timedelta = field(default_factory=lambda: timedelta(hours=24))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.now() > self.created_at + self.ttl


class SemanticCache:
    """Semantic caching for LLM responses using embeddings.

    Features:
    - Similarity-based cache lookup
    - Configurable similarity threshold
    - TTL-based expiration
    - Hit rate tracking
    - LRU eviction when over capacity
    """

    def __init__(
        self,
        embedding_func: Optional[Callable[[str], np.ndarray]] = None,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
        default_ttl: timedelta = timedelta(hours=24)
    ):
        """Initialize semantic cache.

        Args:
            embedding_func: Function to compute embeddings (async or sync)
            similarity_threshold: Minimum similarity for cache hit (0-1)
            max_entries: Maximum cache entries
            default_ttl: Default time-to-live for entries
        """
        self.embedding_func = embedding_func
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._embeddings: List[Tuple[str, np.ndarray]] = []

        # Stats
        self.hits = 0
        self.misses = 0

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.embedding_func:
            if asyncio.iscoroutinefunction(self.embedding_func):
                return await self.embedding_func(text)
            return self.embedding_func(text)

        # Fallback: simple hash-based "embedding" for testing
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return np.frombuffer(hash_bytes, dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def get(self, prompt: str) -> Optional[str]:
        """Get cached response for semantically similar prompt.

        Args:
            prompt: The prompt to look up

        Returns:
            Cached response or None if no match found
        """
        embedding = await self._get_embedding(prompt)

        best_match: Optional[CacheEntry] = None
        best_similarity = 0.0

        for key_hash, cached_embedding in self._embeddings:
            similarity = self._cosine_similarity(embedding, cached_embedding)

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                entry = self._cache.get(key_hash)
                if entry and not entry.is_expired:
                    best_match = entry
                    best_similarity = similarity

        if best_match:
            best_match.hit_count += 1
            self.hits += 1
            logger.debug(f"Cache hit with similarity {best_similarity:.3f}")
            return best_match.response

        self.misses += 1
        return None

    async def set(
        self,
        prompt: str,
        response: str,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Cache a response with its embedding.

        Args:
            prompt: The prompt
            response: The response to cache
            ttl: Optional custom TTL
            metadata: Optional metadata

        Returns:
            Cache key hash
        """
        embedding = await self._get_embedding(prompt)
        key_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        entry = CacheEntry(
            key_hash=key_hash,
            embedding=embedding,
            prompt=prompt,
            response=response,
            created_at=datetime.now(),
            ttl=ttl or self.default_ttl,
            metadata=metadata or {}
        )

        self._cache[key_hash] = entry
        self._embeddings.append((key_hash, embedding))

        # Cleanup if over limit
        self._cleanup()

        return key_hash

    def _cleanup(self) -> None:
        """Remove expired and LRU entries."""
        # Remove expired
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]

        # Remove LRU if over limit
        while len(self._cache) > self.max_entries:
            # Find entry with lowest hit count
            lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].hit_count)
            del self._cache[lru_key]

        # Rebuild embeddings list
        self._embeddings = [(k, e.embedding) for k, e in self._cache.items()]

    def invalidate(self, key_hash: str) -> bool:
        """Invalidate a specific cache entry.

        Returns:
            True if entry was found and removed
        """
        if key_hash in self._cache:
            del self._cache[key_hash]
            self._embeddings = [(k, e) for k, e in self._embeddings if k != key_hash]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._embeddings.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "max_entries": self.max_entries,
            "similarity_threshold": self.similarity_threshold
        }


# =============================================================================
# 5. PRIORITY-BASED TASK SCHEDULING
# =============================================================================

class Priority(IntEnum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class ScheduledTask:
    """A task in the priority queue."""
    sort_key: Tuple[int, float, float] = field(init=False, repr=False)
    priority: Priority = field(compare=False)
    deadline: float = field(compare=False, default=float('inf'))
    created_at: float = field(compare=False, default_factory=time.time)
    task_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    operation: Callable = field(compare=False, default=None)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: Dict = field(compare=False, default_factory=dict)
    age: int = field(compare=False, default=0)

    def __post_init__(self):
        # Sort by: priority (lower=better), deadline, creation time
        effective_priority = max(0, self.priority - (self.age // 10))
        self.sort_key = (effective_priority, self.deadline, self.created_at)


class PriorityScheduler:
    """Priority-based task scheduler with preemption.

    Features:
    - Priority levels (CRITICAL, HIGH, NORMAL, LOW)
    - Deadline-aware scheduling
    - Priority inheritance (aging prevents starvation)
    - Preemption for critical tasks
    - Fair scheduling within priority levels
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        aging_interval: float = 1.0
    ):
        """Initialize scheduler.

        Args:
            max_concurrent: Maximum concurrent task executions
            aging_interval: Seconds between aging increments
        """
        self.max_concurrent = max_concurrent
        self.aging_interval = aging_interval
        self._queue: List[ScheduledTask] = []
        self._running: Dict[str, ScheduledTask] = {}
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}
        self._lock = asyncio.Lock()
        self._running_flag = False
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "preempted": 0
        }

    async def submit(
        self,
        operation: Callable,
        priority: Priority = Priority.NORMAL,
        deadline: Optional[float] = None,
        task_id: Optional[str] = None,
        *args,
        **kwargs
    ) -> str:
        """Submit a task to the scheduler.

        Args:
            operation: Async operation to execute
            priority: Task priority
            deadline: Optional deadline timestamp
            task_id: Optional task ID
            *args, **kwargs: Arguments for the operation

        Returns:
            Task ID
        """
        task = ScheduledTask(
            priority=priority,
            deadline=deadline or float('inf'),
            task_id=task_id or str(uuid.uuid4()),
            operation=operation,
            args=args,
            kwargs=kwargs
        )

        async with self._lock:
            heapq.heappush(self._queue, task)
            self._stats["submitted"] += 1

            # Check for preemption if critical task
            if priority == Priority.CRITICAL and self._running:
                await self._preempt_lowest()

        return task.task_id

    async def _preempt_lowest(self) -> None:
        """Preempt lowest priority running task."""
        if not self._running:
            return

        lowest_task = max(
            self._running.values(),
            key=lambda t: t.priority
        )

        if lowest_task.priority > Priority.CRITICAL:
            # Re-queue the preempted task with boosted priority
            lowest_task.age += 20  # Boost to prevent repeated preemption
            heapq.heappush(self._queue, lowest_task)
            del self._running[lowest_task.task_id]
            self._stats["preempted"] += 1
            logger.info(f"Preempted task {lowest_task.task_id}")

    async def _execute(self, task: ScheduledTask) -> None:
        """Execute a scheduled task."""
        try:
            if asyncio.iscoroutinefunction(task.operation):
                result = await task.operation(*task.args, **task.kwargs)
            else:
                result = task.operation(*task.args, **task.kwargs)
            self._results[task.task_id] = result
            self._stats["completed"] += 1
        except Exception as e:
            self._errors[task.task_id] = e
            self._stats["failed"] += 1
            logger.error(f"Task {task.task_id} failed: {e}")
        finally:
            async with self._lock:
                self._running.pop(task.task_id, None)

    async def _age_tasks(self) -> None:
        """Periodically age tasks to prevent starvation."""
        while self._running_flag:
            await asyncio.sleep(self.aging_interval)
            async with self._lock:
                for task in self._queue:
                    task.age += 1
                # Re-heapify after aging
                heapq.heapify(self._queue)

    async def run(self, timeout: Optional[float] = None) -> None:
        """Run the scheduler.

        Args:
            timeout: Optional timeout in seconds
        """
        self._running_flag = True
        aging_task = asyncio.create_task(self._age_tasks())
        start_time = time.time()

        try:
            while self._running_flag:
                if timeout and (time.time() - start_time) > timeout:
                    break

                async with self._lock:
                    # Fill slots with highest priority tasks
                    while len(self._running) < self.max_concurrent and self._queue:
                        task = heapq.heappop(self._queue)
                        self._running[task.task_id] = task
                        asyncio.create_task(self._execute(task))

                # Check if done
                if not self._queue and not self._running:
                    break

                await asyncio.sleep(0.01)
        finally:
            self._running_flag = False
            aging_task.cancel()
            try:
                await aging_task
            except asyncio.CancelledError:
                pass

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running_flag = False

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Wait for and return task result.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Task result

        Raises:
            TimeoutError: If task doesn't complete in time
            Exception: If task failed
        """
        start = time.time()
        while task_id not in self._results and task_id not in self._errors:
            if time.time() - start > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            await asyncio.sleep(0.1)

        if task_id in self._errors:
            raise self._errors.pop(task_id)

        return self._results.pop(task_id)

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return len(self._queue)

    def get_running_count(self) -> int:
        """Get number of running tasks."""
        return len(self._running)

    def get_stats(self) -> Dict[str, int]:
        """Get scheduler statistics."""
        return {
            **self._stats,
            "queued": len(self._queue),
            "running": len(self._running)
        }


# =============================================================================
# 6. EVENT SOURCING FOR STATE CHANGES
# =============================================================================

@dataclass
class Event:
    """Base class for all events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    aggregate_id: str = ""

    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return {
            "event_type": self.__class__.__name__,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "aggregate_id": self.aggregate_id,
            "data": {
                k: v for k, v in self.__dict__.items()
                if k not in ("event_id", "timestamp", "version", "aggregate_id")
            }
        }


# Specific event types
@dataclass
class WorkflowStartedEvent(Event):
    """Event when a workflow starts."""
    workflow_id: str = ""
    description: str = ""
    initial_state: Dict = field(default_factory=dict)


@dataclass
class PhaseStartedEvent(Event):
    """Event when a phase starts."""
    phase: str = ""
    agents: List[str] = field(default_factory=list)


@dataclass
class PhaseCompletedEvent(Event):
    """Event when a phase completes."""
    phase: str = ""
    duration_ms: float = 0
    success: bool = True
    outputs: Dict = field(default_factory=dict)


@dataclass
class AgentSpawnedEvent(Event):
    """Event when an agent is spawned."""
    agent_id: str = ""
    agent_type: str = ""
    phase: str = ""


@dataclass
class AgentCompletedEvent(Event):
    """Event when an agent completes."""
    agent_id: str = ""
    success: bool = True
    result: Dict = field(default_factory=dict)


@dataclass
class DecisionMadeEvent(Event):
    """Event when a decision is made."""
    decision: str = ""
    rationale: str = ""
    agent_id: str = ""
    alternatives: List[str] = field(default_factory=list)


@dataclass
class FileGeneratedEvent(Event):
    """Event when a file is generated."""
    path: str = ""
    size_bytes: int = 0
    content_hash: str = ""


@dataclass
class ValidationResultEvent(Event):
    """Event for validation results."""
    check_name: str = ""
    passed: bool = False
    message: str = ""
    details: Dict = field(default_factory=dict)


@dataclass
class ErrorOccurredEvent(Event):
    """Event when an error occurs."""
    error_type: str = ""
    error_message: str = ""
    agent_id: str = ""
    recoverable: bool = True


@dataclass
class WorkflowCompletedEvent(Event):
    """Event when workflow completes."""
    workflow_id: str = ""
    success: bool = True
    duration_ms: float = 0
    outputs: Dict = field(default_factory=dict)


class EventStore:
    """Append-only store for events.

    Features:
    - Immutable event log
    - Event subscription
    - Filtering and querying
    - Optional persistence
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize event store.

        Args:
            storage_path: Optional path for persistent storage
        """
        self._events: List[Event] = []
        self._storage_path = storage_path
        self._subscribers: List[Callable[[Event], None]] = []
        self._event_index: Dict[str, List[int]] = {}  # event_type -> indices

    def append(self, event: Event) -> str:
        """Append event to store.

        Args:
            event: Event to append

        Returns:
            Event ID
        """
        index = len(self._events)
        self._events.append(event)

        # Update index
        event_type = event.__class__.__name__
        if event_type not in self._event_index:
            self._event_index[event_type] = []
        self._event_index[event_type].append(index)

        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")

        # Persist if storage configured
        if self._storage_path:
            self._persist(event)

        return event.event_id

    def get_events(
        self,
        event_type: Optional[str] = None,
        aggregate_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Event]:
        """Query events with filters.

        Args:
            event_type: Filter by event type name
            aggregate_id: Filter by aggregate ID
            since: Filter events after this time
            until: Filter events before this time
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        if event_type and event_type in self._event_index:
            indices = self._event_index[event_type]
            result = [self._events[i] for i in indices]
        else:
            result = list(self._events)

        if aggregate_id:
            result = [e for e in result if e.aggregate_id == aggregate_id]

        if since:
            result = [e for e in result if e.timestamp >= since]

        if until:
            result = [e for e in result if e.timestamp <= until]

        return result[-limit:]

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to new events.

        Args:
            callback: Function to call for each new event
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _persist(self, event: Event) -> None:
        """Persist event to storage."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def load_from_storage(self) -> int:
        """Load events from storage file.

        Returns:
            Number of events loaded
        """
        if not self._storage_path or not self._storage_path.exists():
            return 0

        count = 0
        with open(self._storage_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Reconstruct event (simplified - real impl would use registry)
                    event = Event(
                        event_id=data["event_id"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        version=data["version"],
                        aggregate_id=data.get("aggregate_id", "")
                    )
                    self._events.append(event)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load event: {e}")

        return count

    def get_event_count(self) -> int:
        """Get total event count."""
        return len(self._events)

    def clear(self) -> None:
        """Clear all events (for testing)."""
        self._events.clear()
        self._event_index.clear()


T = TypeVar('T')


class StateProjector(Generic[T], ABC):
    """Projects events into current state.

    Subclass this to create projections for different views of the event stream.
    """

    def __init__(self):
        self.state: T = self._initial_state()

    @abstractmethod
    def _initial_state(self) -> T:
        """Return initial state."""
        pass

    @abstractmethod
    def apply(self, event: Event) -> None:
        """Apply event to state."""
        pass

    def replay(self, events: List[Event]) -> T:
        """Replay events to rebuild state.

        Args:
            events: Events to replay

        Returns:
            Final state
        """
        self.state = self._initial_state()
        for event in events:
            self.apply(event)
        return self.state


class WorkflowProjector(StateProjector[Dict]):
    """Projects workflow events into workflow state."""

    def _initial_state(self) -> Dict:
        return {
            "workflow_id": None,
            "status": "pending",
            "current_phase": None,
            "phases_completed": [],
            "active_agents": [],
            "decisions": [],
            "files_generated": [],
            "validation_results": [],
            "errors": [],
            "started_at": None,
            "completed_at": None
        }

    def apply(self, event: Event) -> None:
        if isinstance(event, WorkflowStartedEvent):
            self.state["workflow_id"] = event.workflow_id
            self.state["status"] = "running"
            self.state["started_at"] = event.timestamp.isoformat()

        elif isinstance(event, PhaseStartedEvent):
            self.state["current_phase"] = event.phase
            self.state["active_agents"] = event.agents

        elif isinstance(event, PhaseCompletedEvent):
            self.state["phases_completed"].append({
                "phase": event.phase,
                "duration_ms": event.duration_ms,
                "success": event.success
            })
            self.state["current_phase"] = None
            self.state["active_agents"] = []

        elif isinstance(event, DecisionMadeEvent):
            self.state["decisions"].append({
                "decision": event.decision,
                "rationale": event.rationale,
                "agent": event.agent_id,
                "timestamp": event.timestamp.isoformat()
            })

        elif isinstance(event, FileGeneratedEvent):
            self.state["files_generated"].append({
                "path": event.path,
                "size": event.size_bytes,
                "hash": event.content_hash
            })

        elif isinstance(event, ValidationResultEvent):
            self.state["validation_results"].append({
                "check": event.check_name,
                "passed": event.passed,
                "message": event.message
            })

        elif isinstance(event, ErrorOccurredEvent):
            self.state["errors"].append({
                "type": event.error_type,
                "message": event.error_message,
                "agent": event.agent_id,
                "recoverable": event.recoverable
            })

        elif isinstance(event, WorkflowCompletedEvent):
            self.state["status"] = "completed" if event.success else "failed"
            self.state["completed_at"] = event.timestamp.isoformat()


# =============================================================================
# 7. WORKFLOW VERSIONING AND MIGRATION
# =============================================================================

@dataclass
class Version:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string."""
        parts = version_str.split(".")
        return cls(
            int(parts[0]),
            int(parts[1]) if len(parts) > 1 else 0,
            int(parts[2]) if len(parts) > 2 else 0
        )


class Migration(ABC):
    """Base class for state migrations."""

    @property
    @abstractmethod
    def from_version(self) -> Version:
        """Source version."""
        pass

    @property
    @abstractmethod
    def to_version(self) -> Version:
        """Target version."""
        pass

    @abstractmethod
    def migrate(self, state: Dict) -> Dict:
        """Migrate state from one version to another.

        Args:
            state: State in source version format

        Returns:
            State in target version format
        """
        pass

    @abstractmethod
    def validate(self, state: Dict) -> bool:
        """Validate state after migration.

        Args:
            state: Migrated state

        Returns:
            True if valid
        """
        pass


class MigrationV1_0ToV1_1(Migration):
    """Migrate from v1.0 to v1.1: Normalize template field."""

    @property
    def from_version(self) -> Version:
        return Version(1, 0, 0)

    @property
    def to_version(self) -> Version:
        return Version(1, 1, 0)

    def migrate(self, state: Dict) -> Dict:
        state = state.copy()
        template = state.get("template", "")
        if isinstance(template, str):
            state["template"] = {"type": template, "version": "latest"}
        state["_version"] = str(self.to_version)
        return state

    def validate(self, state: Dict) -> bool:
        return isinstance(state.get("template"), dict)


class MigrationV1_1ToV2_0(Migration):
    """Migrate from v1.1 to v2.0: Add context preservation fields."""

    @property
    def from_version(self) -> Version:
        return Version(1, 1, 0)

    @property
    def to_version(self) -> Version:
        return Version(2, 0, 0)

    def migrate(self, state: Dict) -> Dict:
        state = state.copy()
        state.setdefault("key_decisions", [])
        state.setdefault("phase_history", [])
        state.setdefault("completed_work", [])
        state.setdefault("pending_items", [])
        state["_version"] = str(self.to_version)
        return state

    def validate(self, state: Dict) -> bool:
        required = ["key_decisions", "phase_history", "completed_work", "pending_items"]
        return all(k in state for k in required)


class MigrationV2_0ToV2_1(Migration):
    """Migrate from v2.0 to v2.1: Add agent tracking."""

    @property
    def from_version(self) -> Version:
        return Version(2, 0, 0)

    @property
    def to_version(self) -> Version:
        return Version(2, 1, 0)

    def migrate(self, state: Dict) -> Dict:
        state = state.copy()
        state.setdefault("agent_states", {})
        state.setdefault("agent_outputs", {})
        state.setdefault("active_agents", [])
        state["_version"] = str(self.to_version)
        return state

    def validate(self, state: Dict) -> bool:
        return all(k in state for k in ["agent_states", "agent_outputs", "active_agents"])


class StateMigrator:
    """Manages state version migrations.

    Features:
    - Semantic versioning
    - Forward-only migrations
    - Automatic migration path finding
    - Validation after each step
    """

    def __init__(self, current_version: Optional[Version] = None):
        """Initialize migrator.

        Args:
            current_version: Current schema version
        """
        self._migrations: Dict[Tuple[Version, Version], Migration] = {}
        self._current_version = current_version or Version(2, 1, 0)

    def register(self, migration: Migration) -> None:
        """Register a migration.

        Args:
            migration: Migration to register
        """
        key = (migration.from_version, migration.to_version)
        self._migrations[key] = migration

    def get_version(self, state: Dict) -> Version:
        """Get version from state.

        Args:
            state: State dictionary

        Returns:
            Version of the state
        """
        version_str = state.get("_version", "1.0.0")
        return Version.parse(version_str)

    def needs_migration(self, state: Dict) -> bool:
        """Check if state needs migration.

        Args:
            state: State to check

        Returns:
            True if migration needed
        """
        return self.get_version(state) < self._current_version

    def get_migration_path(self, from_version: Version) -> List[Migration]:
        """Find migration path from version to current.

        Args:
            from_version: Starting version

        Returns:
            List of migrations to apply in order
        """
        path = []
        current = from_version

        while current < self._current_version:
            migration = self._find_migration(current)
            if not migration:
                break
            path.append(migration)
            current = migration.to_version

        return path

    def migrate(self, state: Dict) -> Dict:
        """Migrate state to current version.

        Args:
            state: State to migrate

        Returns:
            Migrated state

        Raises:
            RuntimeError: If migration fails
        """
        current = self.get_version(state)

        while current < self._current_version:
            migration = self._find_migration(current)
            if not migration:
                raise RuntimeError(f"No migration path from {current}")

            # Apply migration
            state = migration.migrate(state)

            # Validate
            if not migration.validate(state):
                raise RuntimeError(
                    f"Migration validation failed: {migration.from_version} -> {migration.to_version}"
                )

            logger.info(f"Migrated state: {migration.from_version} -> {migration.to_version}")
            current = migration.to_version

        return state

    def _find_migration(self, from_version: Version) -> Optional[Migration]:
        """Find migration from given version."""
        for (fv, tv), migration in self._migrations.items():
            if fv == from_version:
                return migration
        return None

    @property
    def current_version(self) -> Version:
        """Get current schema version."""
        return self._current_version


# Create default migrator with all migrations
def create_default_migrator() -> StateMigrator:
    """Create migrator with all built-in migrations."""
    migrator = StateMigrator()
    migrator.register(MigrationV1_0ToV1_1())
    migrator.register(MigrationV1_1ToV2_0())
    migrator.register(MigrationV2_0ToV2_1())
    return migrator


# =============================================================================
# 8. RETRY POLICIES WITH EXPONENTIAL BACKOFF
# =============================================================================

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.3
    retryable_exceptions: Tuple[type, ...] = (Exception,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0.1, delay)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Check if should retry given attempt and exception.

        Args:
            attempt: Current attempt number
            exception: The exception that occurred

        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False
        return isinstance(exception, self.retryable_exceptions)


class RetryError(Exception):
    """Raised when all retries are exhausted."""

    def __init__(self, message: str, attempts: List[Dict]):
        super().__init__(message)
        self.attempts = attempts


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""
    attempt: int
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    delay: float = 0.0


def with_retry(policy: Optional[RetryPolicy] = None):
    """Decorator for adding retry logic to async functions.

    Args:
        policy: Retry policy configuration

    Returns:
        Decorated function with retry logic
    """
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = []

            for attempt in range(policy.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    attempts.append({
                        "attempt": attempt + 1,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    })
                    return result

                except policy.retryable_exceptions as e:
                    attempts.append({
                        "attempt": attempt + 1,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": datetime.now().isoformat()
                    })

                    if attempt < policy.max_retries:
                        delay = policy.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{policy.max_retries} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise RetryError(
                            f"All {policy.max_retries + 1} attempts failed for {func.__name__}",
                            attempts
                        )

        wrapper.retry_policy = policy
        return wrapper
    return decorator


class RetryableOperation:
    """Context manager for retryable operations with logging.

    Provides more control than the decorator, including:
    - Access to attempt history
    - Custom retry callbacks
    - Manual retry control
    """

    def __init__(
        self,
        name: str,
        policy: Optional[RetryPolicy] = None,
        on_retry: Optional[Callable[[str, int, Exception], None]] = None,
        on_success: Optional[Callable[[str, int], None]] = None
    ):
        """Initialize retryable operation.

        Args:
            name: Operation name for logging
            policy: Retry policy
            on_retry: Callback on retry (name, attempt, error)
            on_success: Callback on success (name, attempt)
        """
        self.name = name
        self.policy = policy or RetryPolicy()
        self.on_retry = on_retry
        self.on_success = on_success
        self.attempts: List[RetryAttempt] = []

    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retries.

        Args:
            operation: Async operation to execute
            *args, **kwargs: Operation arguments

        Returns:
            Operation result

        Raises:
            RetryError: If all retries exhausted
        """
        for attempt in range(self.policy.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)

                self.attempts.append(RetryAttempt(
                    attempt=attempt + 1,
                    timestamp=datetime.now(),
                    success=True
                ))

                if self.on_success:
                    self.on_success(self.name, attempt + 1)

                return result

            except self.policy.retryable_exceptions as e:
                delay = self.policy.get_delay(attempt)
                self.attempts.append(RetryAttempt(
                    attempt=attempt + 1,
                    timestamp=datetime.now(),
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    delay=delay
                ))

                if self.on_retry:
                    self.on_retry(self.name, attempt + 1, e)

                if attempt < self.policy.max_retries:
                    await asyncio.sleep(delay)
                else:
                    raise RetryError(
                        f"Operation '{self.name}' failed after {self.policy.max_retries + 1} attempts",
                        [{"attempt": a.attempt, "error": a.error} for a in self.attempts]
                    )

    def get_attempts(self) -> List[Dict]:
        """Get attempt history as dicts."""
        return [
            {
                "attempt": a.attempt,
                "timestamp": a.timestamp.isoformat(),
                "success": a.success,
                "error": a.error,
                "delay": a.delay
            }
            for a in self.attempts
        ]


# =============================================================================
# 9. WORKFLOW COMPOSITION AND TEMPLATES
# =============================================================================

class WorkflowStep:
    """A single step in a composed workflow."""

    def __init__(self, name: str, operation: Callable):
        """Initialize workflow step.

        Args:
            name: Step name
            operation: Async operation to execute
        """
        self.name = name
        self.operation = operation

    async def execute(self, state: Dict) -> Dict:
        """Execute the step.

        Args:
            state: Input state

        Returns:
            Updated state
        """
        if asyncio.iscoroutinefunction(self.operation):
            return await self.operation(state)
        return self.operation(state)


class ParallelSteps:
    """Execute multiple steps in parallel."""

    def __init__(self, *steps: Optional[WorkflowStep]):
        """Initialize with steps to run in parallel.

        Args:
            *steps: Steps to execute (None values are filtered out)
        """
        self.steps = [s for s in steps if s is not None]

    async def execute(self, state: Dict) -> Dict:
        """Execute all steps in parallel.

        Args:
            state: Input state

        Returns:
            Merged state from all steps
        """
        if not self.steps:
            return state

        tasks = [step.execute(state.copy()) for step in self.steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        merged = state.copy()
        for result in results:
            if isinstance(result, Exception):
                raise result
            for key, value in result.items():
                if key in merged and isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = merged[key] + value
                elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value

        return merged


class SequentialSteps:
    """Execute steps sequentially."""

    def __init__(self, *steps: Optional[WorkflowStep]):
        """Initialize with steps to run sequentially.

        Args:
            *steps: Steps to execute in order
        """
        self.steps = [s for s in steps if s is not None]

    async def execute(self, state: Dict) -> Dict:
        """Execute steps in sequence.

        Args:
            state: Input state

        Returns:
            Final state after all steps
        """
        for step in self.steps:
            state = await step.execute(state)
        return state


class ConditionalStep:
    """Execute step based on condition."""

    def __init__(
        self,
        condition: Callable[[Dict], bool],
        if_true: WorkflowStep,
        if_false: Optional[WorkflowStep] = None
    ):
        """Initialize conditional step.

        Args:
            condition: Function to evaluate condition
            if_true: Step to execute if condition is true
            if_false: Optional step if condition is false
        """
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    async def execute(self, state: Dict) -> Dict:
        """Execute based on condition.

        Args:
            state: Input state

        Returns:
            Updated state
        """
        if self.condition(state):
            return await self.if_true.execute(state)
        elif self.if_false:
            return await self.if_false.execute(state)
        return state


class LoopStep:
    """Execute step in a loop until condition is met."""

    def __init__(
        self,
        step: WorkflowStep,
        continue_condition: Callable[[Dict], bool],
        max_iterations: int = 10
    ):
        """Initialize loop step.

        Args:
            step: Step to execute repeatedly
            continue_condition: Continue while this returns True
            max_iterations: Maximum iterations to prevent infinite loops
        """
        self.step = step
        self.continue_condition = continue_condition
        self.max_iterations = max_iterations

    async def execute(self, state: Dict) -> Dict:
        """Execute step in loop.

        Args:
            state: Input state

        Returns:
            Final state after loop
        """
        iteration = 0
        while self.continue_condition(state) and iteration < self.max_iterations:
            state = await self.step.execute(state)
            iteration += 1
        return state


# Helper functions for creating steps
def step(name: str):
    """Decorator to create a workflow step.

    Args:
        name: Step name

    Returns:
        WorkflowStep wrapping the function
    """
    def decorator(func: Callable) -> WorkflowStep:
        return WorkflowStep(name, func)
    return decorator


def parallel(*steps) -> ParallelSteps:
    """Create parallel execution group."""
    return ParallelSteps(*steps)


def sequence(*steps) -> SequentialSteps:
    """Create sequential execution group."""
    return SequentialSteps(*steps)


def when(
    condition: Callable[[Dict], bool],
    then: WorkflowStep,
    otherwise: Optional[WorkflowStep] = None
) -> ConditionalStep:
    """Create conditional step."""
    return ConditionalStep(condition, then, otherwise)


def loop(
    step: WorkflowStep,
    while_condition: Callable[[Dict], bool],
    max_iterations: int = 10
) -> LoopStep:
    """Create loop step."""
    return LoopStep(step, while_condition, max_iterations)


# Noop step for conditional composition
noop = WorkflowStep("noop", lambda state: state)


class WorkflowTemplate:
    """A parameterized workflow template.

    Allows defining reusable workflow patterns that can be
    instantiated with different configurations.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[..., SequentialSteps],
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Initialize workflow template.

        Args:
            name: Template name
            factory: Function that creates workflow steps
            description: Human-readable description
            parameters: Default parameter values
        """
        self.name = name
        self.factory = factory
        self.description = description
        self.default_parameters = parameters or {}

    def instantiate(self, **kwargs) -> "ComposedWorkflow":
        """Create workflow instance with given parameters.

        Args:
            **kwargs: Parameter overrides

        Returns:
            Instantiated workflow
        """
        params = {**self.default_parameters, **kwargs}
        steps = self.factory(**params)
        return ComposedWorkflow(self.name, steps, params)


class ComposedWorkflow:
    """An instantiated workflow ready for execution."""

    def __init__(
        self,
        name: str,
        root: Union[SequentialSteps, ParallelSteps, WorkflowStep],
        parameters: Optional[Dict] = None
    ):
        """Initialize composed workflow.

        Args:
            name: Workflow name
            root: Root step(s) to execute
            parameters: Parameters used to create this workflow
        """
        self.name = name
        self.root = root
        self.parameters = parameters or {}
        self._execution_log: List[Dict] = []

    async def run(self, initial_state: Dict) -> Dict:
        """Execute the workflow.

        Args:
            initial_state: Starting state

        Returns:
            Final state
        """
        start_time = time.time()
        self._execution_log.append({
            "event": "workflow_start",
            "name": self.name,
            "timestamp": datetime.now().isoformat()
        })

        try:
            result = await self.root.execute(initial_state)
            self._execution_log.append({
                "event": "workflow_complete",
                "success": True,
                "duration_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat()
            })
            return result
        except Exception as e:
            self._execution_log.append({
                "event": "workflow_error",
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat()
            })
            raise

    def get_execution_log(self) -> List[Dict]:
        """Get execution log."""
        return self._execution_log


class WorkflowRegistry:
    """Registry for workflow templates.

    Provides centralized management of workflow templates
    with discovery and instantiation support.
    """

    _templates: Dict[str, WorkflowTemplate] = {}

    @classmethod
    def register(cls, name: str, description: str = "", parameters: Optional[Dict] = None):
        """Decorator to register a workflow template.

        Args:
            name: Template name
            description: Template description
            parameters: Default parameters
        """
        def decorator(factory: Callable) -> WorkflowTemplate:
            template = WorkflowTemplate(name, factory, description, parameters)
            cls._templates[name] = template
            return template
        return decorator

    @classmethod
    def get(cls, name: str) -> WorkflowTemplate:
        """Get a registered template.

        Args:
            name: Template name

        Returns:
            WorkflowTemplate

        Raises:
            KeyError: If template not found
        """
        if name not in cls._templates:
            raise KeyError(f"Template not found: {name}")
        return cls._templates[name]

    @classmethod
    def list_templates(cls) -> List[Dict[str, str]]:
        """List all registered templates.

        Returns:
            List of template info dicts
        """
        return [
            {"name": t.name, "description": t.description}
            for t in cls._templates.values()
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered templates."""
        cls._templates.clear()


# =============================================================================
# 10. MULTI-TENANCY AND ISOLATION
# =============================================================================

# Context variable for current tenant
current_tenant: ContextVar[str] = ContextVar('current_tenant', default='default')


@dataclass
class TenantConfig:
    """Configuration for a tenant."""
    tenant_id: str
    max_concurrent_workflows: int = 10
    max_agents_per_workflow: int = 10
    checkpoint_retention_days: int = 30
    rate_limit_per_minute: int = 100
    max_storage_mb: int = 1000
    allowed_templates: Optional[List[str]] = None  # None = all allowed
    metadata: Dict[str, Any] = field(default_factory=dict)


class TenantContext:
    """Context manager for tenant scope.

    Sets the current tenant for the duration of the context,
    ensuring all operations are scoped to that tenant.
    """

    def __init__(self, tenant_id: str):
        """Initialize tenant context.

        Args:
            tenant_id: Tenant to scope to
        """
        self.tenant_id = tenant_id
        self._token = None

    def __enter__(self):
        self._token = current_tenant.set(self.tenant_id)
        return self

    def __exit__(self, *args):
        current_tenant.reset(self._token)

    async def __aenter__(self):
        self._token = current_tenant.set(self.tenant_id)
        return self

    async def __aexit__(self, *args):
        current_tenant.reset(self._token)


class TenantAwareResource:
    """Base class for tenant-aware resources.

    Provides common functionality for resources that need
    to be isolated per tenant.
    """

    def _get_tenant_key(self, key: str) -> str:
        """Prefix key with tenant ID."""
        tenant = current_tenant.get()
        return f"{tenant}:{key}"

    def _get_current_tenant(self) -> str:
        """Get current tenant ID."""
        return current_tenant.get()

    def _validate_tenant_access(self, resource_tenant: str) -> None:
        """Validate current tenant can access resource.

        Args:
            resource_tenant: Tenant that owns the resource

        Raises:
            PermissionError: If access denied
        """
        if resource_tenant != current_tenant.get():
            raise PermissionError(
                f"Tenant {current_tenant.get()} cannot access resource owned by {resource_tenant}"
            )


@dataclass
class TenantUsage:
    """Usage tracking for a tenant."""
    active_workflows: int = 0
    requests_this_minute: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    storage_used_mb: float = 0.0
    total_requests: int = 0


class TenantManager:
    """Manages tenant registration and configuration.

    Provides:
    - Tenant registration and configuration
    - Rate limiting
    - Usage tracking
    - Quota enforcement
    """

    def __init__(self):
        self._tenants: Dict[str, TenantConfig] = {}
        self._usage: Dict[str, TenantUsage] = {}
        self._lock = asyncio.Lock()

    def register_tenant(self, config: TenantConfig) -> None:
        """Register a new tenant.

        Args:
            config: Tenant configuration
        """
        self._tenants[config.tenant_id] = config
        self._usage[config.tenant_id] = TenantUsage()
        logger.info(f"Registered tenant: {config.tenant_id}")

    def unregister_tenant(self, tenant_id: str) -> None:
        """Unregister a tenant."""
        self._tenants.pop(tenant_id, None)
        self._usage.pop(tenant_id, None)

    def get_config(self, tenant_id: str) -> TenantConfig:
        """Get tenant configuration.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tenant configuration (default if not registered)
        """
        return self._tenants.get(tenant_id, TenantConfig(tenant_id=tenant_id))

    def get_usage(self, tenant_id: str) -> TenantUsage:
        """Get tenant usage stats."""
        return self._usage.get(tenant_id, TenantUsage())

    async def check_rate_limit(self, tenant_id: str) -> bool:
        """Check if tenant is within rate limits.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if within limits
        """
        async with self._lock:
            config = self.get_config(tenant_id)
            usage = self._usage.get(tenant_id)

            if not usage:
                usage = TenantUsage()
                self._usage[tenant_id] = usage

            # Reset counter if minute has passed
            now = datetime.now()
            if (now - usage.last_reset).total_seconds() >= 60:
                usage.requests_this_minute = 0
                usage.last_reset = now

            if usage.requests_this_minute >= config.rate_limit_per_minute:
                return False

            usage.requests_this_minute += 1
            usage.total_requests += 1
            return True

    async def check_workflow_limit(self, tenant_id: str) -> bool:
        """Check if tenant can start new workflow.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if within limits
        """
        async with self._lock:
            config = self.get_config(tenant_id)
            usage = self._usage.get(tenant_id)

            if not usage:
                return True

            return usage.active_workflows < config.max_concurrent_workflows

    async def start_workflow(self, tenant_id: str) -> bool:
        """Record workflow start.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if started successfully
        """
        if not await self.check_workflow_limit(tenant_id):
            return False

        async with self._lock:
            usage = self._usage.get(tenant_id)
            if usage:
                usage.active_workflows += 1

        return True

    async def end_workflow(self, tenant_id: str) -> None:
        """Record workflow end."""
        async with self._lock:
            usage = self._usage.get(tenant_id)
            if usage and usage.active_workflows > 0:
                usage.active_workflows -= 1

    def check_storage_limit(self, tenant_id: str, additional_mb: float = 0) -> bool:
        """Check if tenant is within storage limits.

        Args:
            tenant_id: Tenant ID
            additional_mb: Additional storage being requested

        Returns:
            True if within limits
        """
        config = self.get_config(tenant_id)
        usage = self._usage.get(tenant_id)

        if not usage:
            return True

        return (usage.storage_used_mb + additional_mb) <= config.max_storage_mb

    def update_storage(self, tenant_id: str, delta_mb: float) -> None:
        """Update storage usage.

        Args:
            tenant_id: Tenant ID
            delta_mb: Change in storage (positive or negative)
        """
        usage = self._usage.get(tenant_id)
        if usage:
            usage.storage_used_mb = max(0, usage.storage_used_mb + delta_mb)

    def is_template_allowed(self, tenant_id: str, template_name: str) -> bool:
        """Check if tenant can use a template.

        Args:
            tenant_id: Tenant ID
            template_name: Template name

        Returns:
            True if allowed
        """
        config = self.get_config(tenant_id)

        if config.allowed_templates is None:
            return True  # All allowed

        return template_name in config.allowed_templates

    def get_all_tenants(self) -> List[str]:
        """Get list of all tenant IDs."""
        return list(self._tenants.keys())

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive stats for a tenant."""
        config = self.get_config(tenant_id)
        usage = self.get_usage(tenant_id)

        return {
            "tenant_id": tenant_id,
            "config": {
                "max_concurrent_workflows": config.max_concurrent_workflows,
                "rate_limit_per_minute": config.rate_limit_per_minute,
                "max_storage_mb": config.max_storage_mb,
            },
            "usage": {
                "active_workflows": usage.active_workflows,
                "requests_this_minute": usage.requests_this_minute,
                "storage_used_mb": usage.storage_used_mb,
                "total_requests": usage.total_requests,
            }
        }


class IsolatedResource(Generic[T]):
    """A resource that is isolated per tenant.

    Wraps any resource type to provide automatic tenant isolation.
    """

    def __init__(self, factory: Callable[[], T]):
        """Initialize isolated resource.

        Args:
            factory: Function to create new resource instances
        """
        self._factory = factory
        self._resources: Dict[str, T] = {}

    def get(self) -> T:
        """Get resource for current tenant.

        Creates new resource if none exists for tenant.

        Returns:
            Tenant-specific resource instance
        """
        tenant = current_tenant.get()
        if tenant not in self._resources:
            self._resources[tenant] = self._factory()
        return self._resources[tenant]

    def get_for_tenant(self, tenant_id: str) -> Optional[T]:
        """Get resource for specific tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Resource or None if not exists
        """
        return self._resources.get(tenant_id)

    def clear_tenant(self, tenant_id: str) -> None:
        """Clear resource for a tenant."""
        self._resources.pop(tenant_id, None)

    def clear_all(self) -> None:
        """Clear all tenant resources."""
        self._resources.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Circuit Breaker
    "CircuitState",
    "CircuitBreaker",
    "AgentHealthMonitor",

    # Dynamic Scaling
    "ScalingConfig",
    "AgentMetrics",
    "DynamicScaler",

    # Distributed Tracing
    "Span",
    "Tracer",

    # Semantic Cache
    "CacheEntry",
    "SemanticCache",

    # Priority Scheduling
    "Priority",
    "ScheduledTask",
    "PriorityScheduler",

    # Event Sourcing
    "Event",
    "WorkflowStartedEvent",
    "PhaseStartedEvent",
    "PhaseCompletedEvent",
    "AgentSpawnedEvent",
    "AgentCompletedEvent",
    "DecisionMadeEvent",
    "FileGeneratedEvent",
    "ValidationResultEvent",
    "ErrorOccurredEvent",
    "WorkflowCompletedEvent",
    "EventStore",
    "StateProjector",
    "WorkflowProjector",

    # Version Migration
    "Version",
    "Migration",
    "MigrationV1_0ToV1_1",
    "MigrationV1_1ToV2_0",
    "MigrationV2_0ToV2_1",
    "StateMigrator",
    "create_default_migrator",

    # Retry Policies
    "RetryPolicy",
    "RetryError",
    "RetryAttempt",
    "with_retry",
    "RetryableOperation",

    # Workflow Composition
    "WorkflowStep",
    "ParallelSteps",
    "SequentialSteps",
    "ConditionalStep",
    "LoopStep",
    "step",
    "parallel",
    "sequence",
    "when",
    "loop",
    "noop",
    "WorkflowTemplate",
    "ComposedWorkflow",
    "WorkflowRegistry",

    # Multi-Tenancy
    "current_tenant",
    "TenantConfig",
    "TenantContext",
    "TenantAwareResource",
    "TenantUsage",
    "TenantManager",
    "IsolatedResource",
]
