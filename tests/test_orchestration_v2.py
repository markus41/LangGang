"""
Comprehensive tests for LangGraph orchestration v2 module.

Tests cover all 10 advanced orchestration patterns:
1. Circuit Breaker & Health Monitoring
2. Dynamic Agent Scaling
3. Distributed Tracing
4. Semantic Caching
5. Priority-Based Task Scheduling
6. Event Sourcing
7. Workflow Versioning and Migration
8. Retry Policies
9. Workflow Composition and Templates
10. Multi-Tenancy and Isolation
"""

import asyncio
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from langgang.orchestration_v2 import (
    # Circuit Breaker
    CircuitState,
    CircuitBreaker,
    AgentHealthMonitor,
    # Dynamic Scaling
    ScalingConfig,
    AgentMetrics,
    DynamicScaler,
    # Distributed Tracing
    Span,
    Tracer,
    # Semantic Cache
    CacheEntry,
    SemanticCache,
    # Priority Scheduling
    Priority,
    ScheduledTask,
    PriorityScheduler,
    # Event Sourcing
    Event,
    WorkflowStartedEvent,
    PhaseStartedEvent,
    PhaseCompletedEvent,
    DecisionMadeEvent,
    FileGeneratedEvent,
    ErrorOccurredEvent,
    WorkflowCompletedEvent,
    EventStore,
    WorkflowProjector,
    # Version Migration
    Version,
    MigrationV1_0ToV1_1,
    MigrationV1_1ToV2_0,
    MigrationV2_0ToV2_1,
    StateMigrator,
    create_default_migrator,
    # Retry Policies
    RetryPolicy,
    RetryError,
    with_retry,
    RetryableOperation,
    # Workflow Composition
    WorkflowStep,
    ParallelSteps,
    SequentialSteps,
    ConditionalStep,
    LoopStep,
    step,
    parallel,
    sequence,
    when,
    loop,
    noop,
    WorkflowTemplate,
    ComposedWorkflow,
    WorkflowRegistry,
    # Multi-Tenancy
    current_tenant,
    TenantConfig,
    TenantContext,
    TenantUsage,
    TenantManager,
    IsolatedResource,
)


# =============================================================================
# 1. CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker:
    """Tests for Circuit Breaker pattern."""

    def test_initial_state_is_closed(self):
        """Circuit starts in closed state."""
        cb = CircuitBreaker(agent_id="test_agent")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(agent_id="test", failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_resets_failure_count(self):
        """Success in closed state resets failure count."""
        cb = CircuitBreaker(agent_id="test", failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self):
        """Circuit goes to half-open after recovery timeout."""
        cb = CircuitBreaker(
            agent_id="test",
            failure_threshold=1,
            recovery_timeout=timedelta(milliseconds=100)
        )

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        assert cb.can_execute()  # This triggers half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """Circuit closes after successes in half-open state."""
        cb = CircuitBreaker(
            agent_id="test",
            failure_threshold=1,
            recovery_timeout=timedelta(milliseconds=50),
            half_open_max_calls=2
        )

        cb.record_failure()
        time.sleep(0.1)
        cb.can_execute()  # Trigger half-open

        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_half_open_opens_on_failure(self):
        """Circuit opens again on failure in half-open state."""
        cb = CircuitBreaker(
            agent_id="test",
            failure_threshold=1,
            recovery_timeout=timedelta(milliseconds=50)
        )

        cb.record_failure()
        time.sleep(0.1)
        cb.can_execute()  # Trigger half-open

        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_get_status(self):
        """Test status reporting."""
        cb = CircuitBreaker(agent_id="test_agent")
        cb.record_failure()

        status = cb.get_status()

        assert status["agent_id"] == "test_agent"
        assert status["state"] == "closed"
        assert status["failure_count"] == 1


class TestAgentHealthMonitor:
    """Tests for Agent Health Monitor."""

    def test_register_agent(self):
        """Test agent registration."""
        monitor = AgentHealthMonitor()
        monitor.register_agent("agent_1", fallbacks=["agent_2"])

        circuit = monitor.get_circuit("agent_1")
        assert circuit is not None
        assert circuit.agent_id == "agent_1"

    def test_is_healthy(self):
        """Test health status check."""
        monitor = AgentHealthMonitor()
        monitor.register_agent("agent_1")

        assert monitor.is_healthy("agent_1")

        # Make unhealthy
        circuit = monitor.get_circuit("agent_1")
        for _ in range(5):
            circuit.record_failure()

        assert not monitor.is_healthy("agent_1")

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self):
        """Test execution with fallback."""
        monitor = AgentHealthMonitor()
        monitor.register_agent("primary", fallbacks=["backup"])
        monitor.register_agent("backup")

        call_count = {"primary": 0, "backup": 0}

        async def operation(agent_id):
            call_count[agent_id] += 1
            if agent_id == "primary":
                raise RuntimeError("Primary failed")
            return f"Result from {agent_id}"

        # Open primary circuit
        circuit = monitor.get_circuit("primary")
        for _ in range(5):
            circuit.record_failure()

        result = await monitor.execute_with_fallback("primary", operation)

        assert result == "Result from backup"
        assert call_count["backup"] == 1

    @pytest.mark.asyncio
    async def test_run_health_checks(self):
        """Test health check execution."""
        monitor = AgentHealthMonitor()

        async def healthy_check():
            return True

        async def unhealthy_check():
            raise RuntimeError("Unhealthy")

        monitor.register_agent("healthy", health_check=healthy_check)
        monitor.register_agent("unhealthy", health_check=unhealthy_check)

        results = await monitor.run_health_checks()

        assert results["healthy"] is True
        assert results["unhealthy"] is False


# =============================================================================
# 2. DYNAMIC SCALING TESTS
# =============================================================================

class TestDynamicScaler:
    """Tests for Dynamic Agent Scaling."""

    def test_initial_metrics(self):
        """Test initial metrics state."""
        scaler = DynamicScaler()
        assert scaler.metrics.queue_depth == 0
        assert scaler.metrics.active_agents == 0

    def test_should_scale_up_on_high_queue(self):
        """Test scale up decision on high queue depth."""
        config = ScalingConfig(scale_up_threshold=5)
        scaler = DynamicScaler(config)

        scaler.update_metrics(
            queue_depth=10,
            processing_time=1.0,
            error_count=0,
            total_count=100
        )

        assert scaler.should_scale_up()

    def test_should_scale_up_on_high_processing_time(self):
        """Test scale up decision on high processing time."""
        config = ScalingConfig(processing_time_threshold=2.0)
        scaler = DynamicScaler(config)

        scaler.update_metrics(
            queue_depth=1,
            processing_time=5.0,
            error_count=0,
            total_count=100
        )

        assert scaler.should_scale_up()

    def test_should_scale_down_on_low_load(self):
        """Test scale down decision on low load."""
        config = ScalingConfig(
            min_agents=1,
            scale_down_threshold=2,
            processing_time_threshold=5.0
        )
        scaler = DynamicScaler(config)
        scaler._agent_pool = ["agent_1", "agent_2"]

        scaler.update_metrics(
            queue_depth=1,
            processing_time=1.0,
            error_count=0,
            total_count=100
        )

        assert scaler.should_scale_down()

    def test_respects_min_agents(self):
        """Test minimum agent constraint."""
        config = ScalingConfig(min_agents=2)
        scaler = DynamicScaler(config)
        scaler._agent_pool = ["agent_1", "agent_2"]

        scaler.update_metrics(
            queue_depth=0,
            processing_time=0.1,
            error_count=0,
            total_count=100
        )

        assert not scaler.should_scale_down()

    def test_respects_max_agents(self):
        """Test maximum agent constraint."""
        config = ScalingConfig(max_agents=3)
        scaler = DynamicScaler(config)
        scaler._agent_pool = ["a1", "a2", "a3"]

        scaler.update_metrics(
            queue_depth=100,
            processing_time=10.0,
            error_count=50,
            total_count=100
        )

        assert not scaler.should_scale_up()

    @pytest.mark.asyncio
    async def test_scale_up(self):
        """Test agent scale up."""
        scaler = DynamicScaler()
        created_agents = []

        async def factory(agent_id):
            created_agents.append(agent_id)

        scaler.set_agent_factory(factory)

        agent_id = await scaler.scale_up()

        assert agent_id is not None
        assert len(created_agents) == 1
        assert agent_id in scaler.get_pool()

    @pytest.mark.asyncio
    async def test_scale_down(self):
        """Test agent scale down."""
        config = ScalingConfig(min_agents=0)
        scaler = DynamicScaler(config)
        scaler._agent_pool = ["agent_1", "agent_2"]
        destroyed_agents = []

        async def destroyer(agent_id):
            destroyed_agents.append(agent_id)

        scaler.set_agent_destroyer(destroyer)

        agent_id = await scaler.scale_down()

        assert agent_id is not None
        assert len(destroyed_agents) == 1
        assert len(scaler.get_pool()) == 1


# =============================================================================
# 3. DISTRIBUTED TRACING TESTS
# =============================================================================

class TestTracer:
    """Tests for Distributed Tracing."""

    def test_start_trace(self):
        """Test starting a new trace."""
        tracer = Tracer()
        trace_id = tracer.start_trace("test_operation")

        assert trace_id is not None
        assert tracer.get_current_span() is not None
        assert tracer.get_current_span().name == "test_operation"

    def test_span_context_manager(self):
        """Test span creation with context manager."""
        tracer = Tracer()
        tracer.start_trace("root")

        with tracer.span("child_span") as span:
            assert span.name == "child_span"
            assert span.parent_id is not None

        # Span should be ended
        spans = tracer.get_trace(tracer.get_current_span().trace_id)
        child = [s for s in spans if s.name == "child_span"][0]
        assert child.end_time is not None

    def test_nested_spans(self):
        """Test nested span hierarchy."""
        tracer = Tracer()
        trace_id = tracer.start_trace("root")

        with tracer.span("level1") as span1:
            with tracer.span("level2") as span2:
                assert span2.parent_id == span1.span_id

        spans = tracer.get_trace(trace_id)
        assert len(spans) == 3

    def test_span_attributes(self):
        """Test setting span attributes."""
        tracer = Tracer()
        tracer.start_trace("root")

        with tracer.span("test", {"initial": "value"}) as span:
            span.set_attribute("key", "value")

        assert span.attributes["key"] == "value"
        assert span.attributes["initial"] == "value"

    def test_span_events(self):
        """Test adding events to span."""
        tracer = Tracer()
        tracer.start_trace("root")

        with tracer.span("test") as span:
            span.add_event("event1", {"detail": "info"})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "event1"

    def test_span_error(self):
        """Test error recording in span."""
        tracer = Tracer()
        tracer.start_trace("root")

        try:
            with tracer.span("failing") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert span.status == "ERROR"
        assert "ValueError" in span.attributes["error.type"]

    def test_export_trace(self):
        """Test OpenTelemetry-compatible export."""
        tracer = Tracer("test_service")
        trace_id = tracer.start_trace("root")

        with tracer.span("child"):
            pass

        tracer.end_trace()
        export = tracer.export_trace(trace_id)

        assert "resourceSpans" in export
        assert len(export["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2


class TestSpan:
    """Tests for Span class."""

    def test_duration_ms(self):
        """Test duration calculation."""
        span = Span(
            span_id="1",
            trace_id="t1",
            parent_id=None,
            name="test",
            start_time=1000.0,
            end_time=1000.5
        )

        assert span.duration_ms == 500.0

    def test_to_dict(self):
        """Test serialization."""
        span = Span(
            span_id="1",
            trace_id="t1",
            parent_id="p1",
            name="test",
            start_time=1000.0
        )
        span.set_attribute("key", "value")

        data = span.to_dict()

        assert data["span_id"] == "1"
        assert data["name"] == "test"
        assert data["attributes"]["key"] == "value"


# =============================================================================
# 4. SEMANTIC CACHE TESTS
# =============================================================================

class TestSemanticCache:
    """Tests for Semantic Caching."""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache set and get."""
        cache = SemanticCache(similarity_threshold=0.99)

        await cache.set("What is Python?", "Python is a programming language")
        result = await cache.get("What is Python?")

        assert result == "Python is a programming language"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss for different prompt."""
        cache = SemanticCache(similarity_threshold=0.99)

        await cache.set("What is Python?", "Python is a programming language")
        result = await cache.get("What is Java?")

        assert result is None

    @pytest.mark.asyncio
    async def test_hit_rate_tracking(self):
        """Test hit rate statistics."""
        cache = SemanticCache(similarity_threshold=0.99)

        await cache.set("prompt1", "response1")
        await cache.get("prompt1")  # Hit
        await cache.get("prompt2")  # Miss
        await cache.get("prompt1")  # Hit

        assert cache.hits == 2
        assert cache.misses == 1
        assert cache.hit_rate == 2 / 3

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test TTL-based expiration."""
        cache = SemanticCache(
            similarity_threshold=0.99,
            default_ttl=timedelta(milliseconds=100)
        )

        await cache.set("prompt", "response")
        time.sleep(0.15)

        result = await cache.get("prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test manual cache invalidation."""
        cache = SemanticCache(similarity_threshold=0.99)

        key = await cache.set("prompt", "response")
        assert await cache.get("prompt") is not None

        cache.invalidate(key)
        assert await cache.get("prompt") is None

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing cache."""
        cache = SemanticCache()

        await cache.set("p1", "r1")
        await cache.set("p2", "r2")
        cache.clear()

        assert len(cache._cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_get_stats(self):
        """Test statistics reporting."""
        cache = SemanticCache(max_entries=100, similarity_threshold=0.9)
        stats = cache.get_stats()

        assert stats["max_entries"] == 100
        assert stats["similarity_threshold"] == 0.9
        assert "hit_rate" in stats


# =============================================================================
# 5. PRIORITY SCHEDULING TESTS
# =============================================================================

class TestPriorityScheduler:
    """Tests for Priority-Based Task Scheduling."""

    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test task submission."""
        scheduler = PriorityScheduler()

        async def operation():
            return "result"

        task_id = await scheduler.submit(operation, Priority.NORMAL)

        assert task_id is not None
        assert scheduler.get_queue_depth() == 1

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test tasks execute in priority order."""
        scheduler = PriorityScheduler(max_concurrent=1)
        execution_order = []

        async def task(name):
            execution_order.append(name)
            return name

        await scheduler.submit(lambda: task("low"), Priority.LOW, task_id="low")
        await scheduler.submit(lambda: task("high"), Priority.HIGH, task_id="high")
        await scheduler.submit(lambda: task("critical"), Priority.CRITICAL, task_id="critical")

        await scheduler.run(timeout=1.0)

        assert execution_order[0] == "critical"
        assert execution_order[1] == "high"
        assert execution_order[2] == "low"

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent task execution."""
        scheduler = PriorityScheduler(max_concurrent=3)
        running = []

        async def task(name):
            running.append(name)
            await asyncio.sleep(0.1)
            return name

        for i in range(3):
            await scheduler.submit(lambda i=i: task(f"task_{i}"))

        # Start scheduler in background
        run_task = asyncio.create_task(scheduler.run(timeout=1.0))
        await asyncio.sleep(0.05)

        # All should be running
        assert scheduler.get_running_count() == 3

        await run_task

    @pytest.mark.asyncio
    async def test_get_result(self):
        """Test getting task result."""
        scheduler = PriorityScheduler()

        async def operation():
            return "expected_result"

        task_id = await scheduler.submit(operation)

        # Run scheduler briefly
        run_task = asyncio.create_task(scheduler.run(timeout=1.0))

        result = await scheduler.get_result(task_id, timeout=1.0)

        scheduler.stop()
        _ = await run_task  # Ensure scheduler.run completes before finishing test

        assert result == "expected_result"

    @pytest.mark.asyncio
    async def test_task_failure(self):
        """Test handling of failed tasks."""
        scheduler = PriorityScheduler()

        async def failing_task():
            raise ValueError("Task failed")

        task_id = await scheduler.submit(failing_task)

        run_task = asyncio.create_task(scheduler.run(timeout=1.0))

        with pytest.raises(ValueError):
            await scheduler.get_result(task_id)

        scheduler.stop()
        await run_task

    def test_get_stats(self):
        """Test scheduler statistics."""
        scheduler = PriorityScheduler()
        stats = scheduler.get_stats()

        assert "submitted" in stats
        assert "completed" in stats
        assert "queued" in stats


# =============================================================================
# 6. EVENT SOURCING TESTS
# =============================================================================

class TestEventStore:
    """Tests for Event Store."""

    def test_append_event(self):
        """Test appending events."""
        store = EventStore()

        event = WorkflowStartedEvent(workflow_id="wf1", description="Test")
        event_id = store.append(event)

        assert event_id is not None
        assert store.get_event_count() == 1

    def test_get_events_by_type(self):
        """Test filtering events by type."""
        store = EventStore()

        store.append(WorkflowStartedEvent(workflow_id="wf1"))
        store.append(PhaseStartedEvent(phase="explore"))
        store.append(PhaseCompletedEvent(phase="explore"))

        phase_events = store.get_events(event_type="PhaseStartedEvent")

        assert len(phase_events) == 1
        assert phase_events[0].phase == "explore"

    def test_get_events_by_time_range(self):
        """Test filtering events by time range."""
        store = EventStore()

        old_event = WorkflowStartedEvent(workflow_id="old")
        old_event.timestamp = datetime.now() - timedelta(hours=1)
        store.append(old_event)

        new_event = WorkflowStartedEvent(workflow_id="new")
        store.append(new_event)

        since = datetime.now() - timedelta(minutes=30)
        recent = store.get_events(since=since)

        assert len(recent) == 1
        assert recent[0].workflow_id == "new"

    def test_subscribe_to_events(self):
        """Test event subscription."""
        store = EventStore()
        received = []

        def handler(event):
            received.append(event)

        store.subscribe(handler)
        store.append(WorkflowStartedEvent(workflow_id="wf1"))

        assert len(received) == 1

    def test_event_to_dict(self):
        """Test event serialization."""
        event = DecisionMadeEvent(
            decision="Use template X",
            rationale="Best fit",
            agent_id="researcher"
        )

        data = event.to_dict()

        assert data["event_type"] == "DecisionMadeEvent"
        assert data["data"]["decision"] == "Use template X"


class TestWorkflowProjector:
    """Tests for Workflow Projector."""

    def test_project_workflow_started(self):
        """Test projecting workflow start event."""
        projector = WorkflowProjector()

        projector.apply(WorkflowStartedEvent(workflow_id="wf1"))

        assert projector.state["workflow_id"] == "wf1"
        assert projector.state["status"] == "running"

    def test_project_phase_events(self):
        """Test projecting phase events."""
        projector = WorkflowProjector()

        projector.apply(PhaseStartedEvent(phase="explore", agents=["a1", "a2"]))
        assert projector.state["current_phase"] == "explore"

        projector.apply(PhaseCompletedEvent(phase="explore", duration_ms=100))
        assert projector.state["current_phase"] is None
        assert len(projector.state["phases_completed"]) == 1

    def test_project_decision(self):
        """Test projecting decision event."""
        projector = WorkflowProjector()

        projector.apply(DecisionMadeEvent(
            decision="Choose A",
            rationale="Better performance",
            agent_id="agent1"
        ))

        assert len(projector.state["decisions"]) == 1
        assert projector.state["decisions"][0]["decision"] == "Choose A"

    def test_replay_events(self):
        """Test replaying event stream."""
        events = [
            WorkflowStartedEvent(workflow_id="wf1"),
            PhaseStartedEvent(phase="explore"),
            PhaseCompletedEvent(phase="explore", success=True),
            WorkflowCompletedEvent(workflow_id="wf1", success=True)
        ]

        projector = WorkflowProjector()
        state = projector.replay(events)

        assert state["status"] == "completed"
        assert len(state["phases_completed"]) == 1


# =============================================================================
# 7. VERSION MIGRATION TESTS
# =============================================================================

class TestVersion:
    """Tests for Version class."""

    def test_version_parsing(self):
        """Test version string parsing."""
        v = Version.parse("2.1.3")
        assert v.major == 2
        assert v.minor == 1
        assert v.patch == 3

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = Version(1, 0, 0)
        v2 = Version(2, 0, 0)
        v1_1 = Version(1, 1, 0)

        assert v1 < v2
        assert v1 < v1_1
        assert v2 > v1_1

    def test_version_equality(self):
        """Test version equality."""
        v1 = Version(1, 2, 3)
        v2 = Version(1, 2, 3)

        assert v1 == v2

    def test_version_string(self):
        """Test version to string."""
        v = Version(1, 2, 3)
        assert str(v) == "1.2.3"


class TestStateMigrator:
    """Tests for State Migrator."""

    def test_needs_migration(self):
        """Test migration detection."""
        migrator = create_default_migrator()

        old_state = {"_version": "1.0.0"}
        new_state = {"_version": "2.1.0"}

        assert migrator.needs_migration(old_state)
        assert not migrator.needs_migration(new_state)

    def test_migration_v1_to_v1_1(self):
        """Test v1.0 to v1.1 migration."""
        migration = MigrationV1_0ToV1_1()

        old_state = {"template": "python", "_version": "1.0.0"}
        new_state = migration.migrate(old_state)

        assert isinstance(new_state["template"], dict)
        assert new_state["template"]["type"] == "python"
        assert migration.validate(new_state)

    def test_migration_v1_1_to_v2(self):
        """Test v1.1 to v2.0 migration."""
        migration = MigrationV1_1ToV2_0()

        old_state = {"_version": "1.1.0"}
        new_state = migration.migrate(old_state)

        assert "key_decisions" in new_state
        assert "phase_history" in new_state
        assert migration.validate(new_state)

    def test_full_migration_path(self):
        """Test migrating through multiple versions."""
        migrator = create_default_migrator()

        old_state = {
            "template": "python",
            "_version": "1.0.0"
        }

        new_state = migrator.migrate(old_state)

        assert new_state["_version"] == "2.1.0"
        assert "key_decisions" in new_state
        assert "agent_states" in new_state

    def test_get_migration_path(self):
        """Test migration path finding."""
        migrator = create_default_migrator()
        path = migrator.get_migration_path(Version(1, 0, 0))

        assert len(path) == 3


# =============================================================================
# 8. RETRY POLICY TESTS
# =============================================================================

class TestRetryPolicy:
    """Tests for Retry Policies."""

    def test_get_delay_exponential(self):
        """Test exponential backoff delay."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, jitter=0)

        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0

    def test_get_delay_max_cap(self):
        """Test delay is capped at max."""
        policy = RetryPolicy(base_delay=10.0, max_delay=30.0, jitter=0)

        assert policy.get_delay(5) == 30.0

    def test_get_delay_jitter(self):
        """Test jitter is applied."""
        policy = RetryPolicy(base_delay=10.0, jitter=0.5)

        delays = [policy.get_delay(0) for _ in range(10)]

        # Should have variation
        assert len(set(delays)) > 1
        # Should be within jitter range
        assert all(5.0 <= d <= 15.0 for d in delays)

    def test_should_retry(self):
        """Test retry decision."""
        policy = RetryPolicy(
            max_retries=3,
            retryable_exceptions=(ValueError,)
        )

        assert policy.should_retry(0, ValueError("test"))
        assert policy.should_retry(2, ValueError("test"))
        assert not policy.should_retry(3, ValueError("test"))
        assert not policy.should_retry(0, RuntimeError("test"))


class TestWithRetryDecorator:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Test success on first attempt."""
        call_count = 0

        @with_retry(RetryPolicy(max_retries=3))
        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await operation()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry after failures."""
        call_count = 0

        @with_retry(RetryPolicy(max_retries=3, base_delay=0.01))
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await operation()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries(self):
        """Test error when retries exhausted."""
        @with_retry(RetryPolicy(max_retries=2, base_delay=0.01))
        async def operation():
            raise ValueError("Permanent failure")

        with pytest.raises(RetryError) as exc_info:
            await operation()

        assert len(exc_info.value.attempts) == 3


class TestRetryableOperation:
    """Tests for RetryableOperation class."""

    @pytest.mark.asyncio
    async def test_execute_with_callbacks(self):
        """Test execution with callbacks."""
        retries = []
        successes = []

        op = RetryableOperation(
            name="test_op",
            policy=RetryPolicy(max_retries=2, base_delay=0.01),
            on_retry=lambda n, a, e: retries.append((n, a)),
            on_success=lambda n, a: successes.append((n, a))
        )

        attempt = 0

        async def operation():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise ValueError("Fail")
            return "ok"

        result = await op.execute(operation)

        assert result == "ok"
        assert len(retries) == 1
        assert len(successes) == 1

    @pytest.mark.asyncio
    async def test_get_attempts(self):
        """Test attempt history."""
        op = RetryableOperation(
            name="test",
            policy=RetryPolicy(max_retries=1, base_delay=0.01)
        )

        async def operation():
            return "ok"

        await op.execute(operation)
        attempts = op.get_attempts()

        assert len(attempts) == 1
        assert attempts[0]["success"] is True


# =============================================================================
# 9. WORKFLOW COMPOSITION TESTS
# =============================================================================

class TestWorkflowStep:
    """Tests for WorkflowStep."""

    @pytest.mark.asyncio
    async def test_execute_async(self):
        """Test executing async step."""
        async def operation(state):
            return {**state, "executed": True}

        step = WorkflowStep("test", operation)
        result = await step.execute({"initial": "value"})

        assert result["executed"] is True
        assert result["initial"] == "value"

    @pytest.mark.asyncio
    async def test_execute_sync(self):
        """Test executing sync step."""
        def operation(state):
            return {**state, "executed": True}

        step = WorkflowStep("test", operation)
        result = await step.execute({})

        assert result["executed"] is True


class TestSequentialSteps:
    """Tests for SequentialSteps."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test steps execute in order."""
        order = []

        async def step1(state):
            order.append("step1")
            return {**state, "step1": True}

        async def step2(state):
            order.append("step2")
            return {**state, "step2": True}

        steps = SequentialSteps(
            WorkflowStep("s1", step1),
            WorkflowStep("s2", step2)
        )

        result = await steps.execute({})

        assert order == ["step1", "step2"]
        assert result["step1"] is True
        assert result["step2"] is True


class TestParallelSteps:
    """Tests for ParallelSteps."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test steps execute in parallel."""
        start_times = {}

        async def step(name, state):
            start_times[name] = time.time()
            await asyncio.sleep(0.1)
            return {**state, name: True}

        steps = ParallelSteps(
            WorkflowStep("s1", lambda s: step("s1", s)),
            WorkflowStep("s2", lambda s: step("s2", s))
        )

        result = await steps.execute({})

        # Both should have started nearly simultaneously
        assert abs(start_times["s1"] - start_times["s2"]) < 0.05
        assert result["s1"] is True
        assert result["s2"] is True

    @pytest.mark.asyncio
    async def test_merge_list_results(self):
        """Test merging list results."""
        async def step1(state):
            return {**state, "items": ["a"]}

        async def step2(state):
            return {**state, "items": ["b"]}

        steps = ParallelSteps(
            WorkflowStep("s1", step1),
            WorkflowStep("s2", step2)
        )

        result = await steps.execute({"items": []})

        assert set(result["items"]) == {"a", "b"}


class TestConditionalStep:
    """Tests for ConditionalStep."""

    @pytest.mark.asyncio
    async def test_condition_true(self):
        """Test executing when condition is true."""
        async def if_true(state):
            return {**state, "path": "true"}

        async def if_false(state):
            return {**state, "path": "false"}

        step = ConditionalStep(
            condition=lambda s: s.get("flag", False),
            if_true=WorkflowStep("true", if_true),
            if_false=WorkflowStep("false", if_false)
        )

        result = await step.execute({"flag": True})

        assert result["path"] == "true"

    @pytest.mark.asyncio
    async def test_condition_false(self):
        """Test executing when condition is false."""
        async def if_true(state):
            return {**state, "path": "true"}

        async def if_false(state):
            return {**state, "path": "false"}

        step = ConditionalStep(
            condition=lambda s: s.get("flag", False),
            if_true=WorkflowStep("true", if_true),
            if_false=WorkflowStep("false", if_false)
        )

        result = await step.execute({"flag": False})

        assert result["path"] == "false"


class TestLoopStep:
    """Tests for LoopStep."""

    @pytest.mark.asyncio
    async def test_loop_execution(self):
        """Test loop executes until condition is false."""
        async def increment(state):
            return {**state, "count": state.get("count", 0) + 1}

        step = LoopStep(
            step=WorkflowStep("inc", increment),
            continue_condition=lambda s: s.get("count", 0) < 3
        )

        result = await step.execute({"count": 0})

        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        """Test loop respects max iterations."""
        async def increment(state):
            return {**state, "count": state.get("count", 0) + 1}

        step = LoopStep(
            step=WorkflowStep("inc", increment),
            continue_condition=lambda s: True,  # Would run forever
            max_iterations=5
        )

        result = await step.execute({"count": 0})

        assert result["count"] == 5


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        WorkflowRegistry.clear()

    def test_register_template(self):
        """Test template registration."""
        @WorkflowRegistry.register("test_workflow", "A test workflow")
        def create_workflow():
            return SequentialSteps(noop)

        assert "test_workflow" in [t["name"] for t in WorkflowRegistry.list_templates()]

    def test_get_template(self):
        """Test getting template."""
        @WorkflowRegistry.register("my_workflow")
        def create_workflow():
            return SequentialSteps(noop)

        template = WorkflowRegistry.get("my_workflow")
        assert template.name == "my_workflow"

    def test_get_nonexistent_template(self):
        """Test error for nonexistent template."""
        with pytest.raises(KeyError):
            WorkflowRegistry.get("nonexistent")


class TestComposedWorkflow:
    """Tests for ComposedWorkflow."""

    @pytest.mark.asyncio
    async def test_run_workflow(self):
        """Test running composed workflow."""
        @step("add_one")
        async def add_one(state):
            return {**state, "value": state.get("value", 0) + 1}

        workflow = ComposedWorkflow(
            "test",
            SequentialSteps(add_one, add_one)
        )

        result = await workflow.run({"value": 0})

        assert result["value"] == 2

    @pytest.mark.asyncio
    async def test_execution_log(self):
        """Test execution logging."""
        workflow = ComposedWorkflow("test", SequentialSteps(noop))

        await workflow.run({})
        log = workflow.get_execution_log()

        assert len(log) == 2
        assert log[0]["event"] == "workflow_start"
        assert log[1]["event"] == "workflow_complete"


# =============================================================================
# 10. MULTI-TENANCY TESTS
# =============================================================================

class TestTenantContext:
    """Tests for TenantContext."""

    def test_context_sets_tenant(self):
        """Test context sets current tenant."""
        with TenantContext("tenant_a"):
            assert current_tenant.get() == "tenant_a"

        assert current_tenant.get() == "default"

    def test_nested_contexts(self):
        """Test nested tenant contexts."""
        with TenantContext("outer"):
            assert current_tenant.get() == "outer"

            with TenantContext("inner"):
                assert current_tenant.get() == "inner"

            assert current_tenant.get() == "outer"

    @pytest.mark.asyncio
    async def test_async_context(self):
        """Test async context manager."""
        async with TenantContext("async_tenant"):
            assert current_tenant.get() == "async_tenant"


class TestTenantManager:
    """Tests for TenantManager."""

    def test_register_tenant(self):
        """Test tenant registration."""
        manager = TenantManager()
        config = TenantConfig(
            tenant_id="test_tenant",
            max_concurrent_workflows=5
        )

        manager.register_tenant(config)

        assert "test_tenant" in manager.get_all_tenants()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limit enforcement."""
        manager = TenantManager()
        manager.register_tenant(TenantConfig(
            tenant_id="limited",
            rate_limit_per_minute=3
        ))

        # Should allow first 3 requests
        for _ in range(3):
            assert await manager.check_rate_limit("limited")

        # Should deny 4th request
        assert not await manager.check_rate_limit("limited")

    @pytest.mark.asyncio
    async def test_workflow_limit(self):
        """Test workflow limit enforcement."""
        manager = TenantManager()
        manager.register_tenant(TenantConfig(
            tenant_id="test",
            max_concurrent_workflows=2
        ))

        assert await manager.start_workflow("test")
        assert await manager.start_workflow("test")
        assert not await manager.check_workflow_limit("test")

        await manager.end_workflow("test")
        assert await manager.check_workflow_limit("test")

    def test_storage_limit(self):
        """Test storage limit check."""
        manager = TenantManager()
        manager.register_tenant(TenantConfig(
            tenant_id="test",
            max_storage_mb=100
        ))

        manager.update_storage("test", 50)
        assert manager.check_storage_limit("test", 40)
        assert not manager.check_storage_limit("test", 60)

    def test_template_access(self):
        """Test template access control."""
        manager = TenantManager()
        manager.register_tenant(TenantConfig(
            tenant_id="restricted",
            allowed_templates=["template_a", "template_b"]
        ))

        assert manager.is_template_allowed("restricted", "template_a")
        assert not manager.is_template_allowed("restricted", "template_c")

    def test_tenant_stats(self):
        """Test tenant statistics."""
        manager = TenantManager()
        manager.register_tenant(TenantConfig(tenant_id="stats_test"))

        stats = manager.get_tenant_stats("stats_test")

        assert "config" in stats
        assert "usage" in stats
        assert stats["tenant_id"] == "stats_test"


class TestIsolatedResource:
    """Tests for IsolatedResource."""

    def test_per_tenant_isolation(self):
        """Test resources are isolated per tenant."""
        isolated = IsolatedResource(lambda: {"data": []})

        with TenantContext("tenant_a"):
            res_a = isolated.get()
            res_a["data"].append("a")

        with TenantContext("tenant_b"):
            res_b = isolated.get()
            res_b["data"].append("b")

        with TenantContext("tenant_a"):
            assert isolated.get()["data"] == ["a"]

        with TenantContext("tenant_b"):
            assert isolated.get()["data"] == ["b"]

    def test_get_for_tenant(self):
        """Test getting resource for specific tenant."""
        isolated = IsolatedResource(lambda: {"value": 0})

        with TenantContext("test"):
            isolated.get()["value"] = 42

        assert isolated.get_for_tenant("test")["value"] == 42
        assert isolated.get_for_tenant("other") is None

    def test_clear_tenant(self):
        """Test clearing tenant resource."""
        isolated = IsolatedResource(lambda: {})

        with TenantContext("test"):
            isolated.get()

        isolated.clear_tenant("test")

        assert isolated.get_for_tenant("test") is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry policy with circuit breaker."""
        monitor = AgentHealthMonitor()
        monitor.register_agent("agent1", failure_threshold=2)

        call_count = 0

        @with_retry(RetryPolicy(max_retries=5, base_delay=0.01))
        async def operation():
            nonlocal call_count
            call_count += 1

            circuit = monitor.get_circuit("agent1")
            if not circuit.can_execute():
                raise RuntimeError("Circuit open")

            if call_count < 3:
                circuit.record_failure()
                raise ValueError("Temporary error")

            circuit.record_success()
            return "success"

        result = await operation()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_workflow_with_tracing(self):
        """Test workflow execution with tracing."""
        tracer = Tracer()

        @step("traced_step")
        async def traced_step(state):
            with tracer.span("processing"):
                await asyncio.sleep(0.01)
            return {**state, "processed": True}

        trace_id = tracer.start_trace("workflow")

        workflow = ComposedWorkflow(
            "traced_workflow",
            SequentialSteps(traced_step)
        )

        with tracer.span("workflow_execution"):
            result = await workflow.run({})

        tracer.end_trace()

        spans = tracer.get_trace(trace_id)
        assert len(spans) >= 2
        assert result["processed"] is True

    @pytest.mark.asyncio
    async def test_tenant_isolated_workflow(self):
        """Test workflow isolation per tenant."""
        manager = TenantManager()
        manager.register_tenant(TenantConfig(tenant_id="tenant_a"))
        manager.register_tenant(TenantConfig(tenant_id="tenant_b"))

        results = IsolatedResource(lambda: [])

        @step("record")
        async def record_step(state):
            results.get().append(state["value"])
            return state

        workflow = ComposedWorkflow("test", SequentialSteps(record_step))

        async with TenantContext("tenant_a"):
            await manager.start_workflow("tenant_a")
            await workflow.run({"value": "a"})

        async with TenantContext("tenant_b"):
            await manager.start_workflow("tenant_b")
            await workflow.run({"value": "b"})

        assert results.get_for_tenant("tenant_a") == ["a"]
        assert results.get_for_tenant("tenant_b") == ["b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
