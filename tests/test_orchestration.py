"""
Comprehensive tests for LangGraph orchestration module.

Tests cover all 5 orchestration patterns:
1. Context Preservation Protocol
2. Advanced Checkpointing with Recovery
3. Distributed Lock Manager
4. Inter-Agent Communication Channels
5. Six-Phase Multi-Agent Architecture
"""

import asyncio
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langgang.orchestration import (
    # Context Preservation
    KeyDecision,
    PhaseHandoffContext,
    ContextPreservingState,
    validate_handoff_context,
    create_handoff_context,
    record_decision,
    # Enhanced Checkpointing
    CheckpointTrigger,
    CheckpointMetadata,
    EnhancedCheckpoint,
    EnhancedCheckpointer,
    checkpoint_before_risky,
    # Lock Manager
    LockType,
    Lock,
    LockManager,
    # Message Bus
    MessageType,
    ChannelType,
    AgentMessage,
    MessageBus,
    # Multi-Agent Architecture
    PhaseType,
    AgentConfig,
    PhaseState,
    MultiAgentState,
    DEFAULT_AGENTS,
    create_phase_subgraph,
    create_orchestrated_workflow,
    run_orchestrated_generation,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_state() -> ContextPreservingState:
    """Create a sample context-preserving state."""
    return {
        "messages": [],
        "current_phase": "explore",
        "phase_history": [],
        "active_agents": ["researcher", "analyzer"],
        "completed_work": ["dependency analysis"],
        "pending_items": ["security review"],
        "key_decisions": [],
        "file_locations": {"config": "src/config.py"},
        "dependencies": ["langchain", "pydantic"],
        "issues": [],
        "next_steps": ["create plan", "design API"],
        "project_description": "Test project",
        "template_type": "cookiecutter",
        "template_name": "python-langchain-project",
        "output_dir": "/tmp/test_output",
        "context_variables": {"author": "Test User"},
    }


@pytest.fixture
def lock_manager():
    """Create a lock manager instance."""
    return LockManager()


@pytest.fixture
def message_bus():
    """Create a message bus instance."""
    return MessageBus()


@pytest.fixture
def checkpointer(temp_dir):
    """Create an enhanced checkpointer instance."""
    return EnhancedCheckpointer(storage_dir=temp_dir / "checkpoints")


# =============================================================================
# Test Context Preservation Protocol
# =============================================================================

class TestContextPreservation:
    """Tests for Context Preservation Protocol."""

    def test_record_decision_creates_valid_decision(self):
        """Test that record_decision creates a properly formatted KeyDecision."""
        decision = record_decision(
            decision="Use FastAPI",
            rationale="Good async support",
            alternatives=["Flask", "Django"],
            agent_id="researcher",
            phase="explore"
        )

        assert decision["decision"] == "Use FastAPI"
        assert decision["rationale"] == "Good async support"
        assert decision["alternatives_considered"] == ["Flask", "Django"]
        assert decision["made_by"] == "researcher"
        assert decision["phase"] == "explore"
        assert "timestamp" in decision

    def test_validate_handoff_context_valid(self):
        """Test validation passes for complete context."""
        context: PhaseHandoffContext = {
            "from_phase": "explore",
            "to_phase": "plan",
            "active_agents": ["researcher"],
            "completed_work": ["analysis complete"],
            "pending_items": [],
            "key_decisions": [],
            "file_locations": {},
            "dependencies_discovered": [],
            "issues_requiring_attention": [],
            "explicit_next_steps": ["create plan"],
            "code_snippets": {},
            "external_references": [],
            "timestamp": datetime.now().isoformat(),
            "handoff_id": "test_handoff"
        }

        errors = validate_handoff_context(context)
        assert len(errors) == 0

    def test_validate_handoff_context_missing_required(self):
        """Test validation fails for incomplete context."""
        context: PhaseHandoffContext = {
            "from_phase": "explore",
            "to_phase": "plan",
            "active_agents": [],
            "completed_work": [],  # Empty - should fail
            "pending_items": [],
            "key_decisions": [],
            "file_locations": {},
            "dependencies_discovered": [],
            "issues_requiring_attention": [],
            "explicit_next_steps": [],  # Empty - should fail
            "code_snippets": {},
            "external_references": [],
            "timestamp": datetime.now().isoformat(),
            "handoff_id": "test_handoff"
        }

        errors = validate_handoff_context(context)
        assert len(errors) == 2
        assert any("completed_work" in e for e in errors)
        assert any("explicit_next_steps" in e for e in errors)

    def test_create_handoff_context_from_state(self, sample_state):
        """Test creating handoff context from state."""
        handoff = create_handoff_context(sample_state, "plan")

        assert handoff["from_phase"] == "explore"
        assert handoff["to_phase"] == "plan"
        assert handoff["active_agents"] == ["researcher", "analyzer"]
        assert "dependency analysis" in handoff["completed_work"]
        assert "create plan" in handoff["explicit_next_steps"]
        assert handoff["file_locations"] == {"config": "src/config.py"}
        assert "langchain" in handoff["dependencies_discovered"]


# =============================================================================
# Test Enhanced Checkpointing
# =============================================================================

class TestEnhancedCheckpointer:
    """Tests for Enhanced Checkpointing with Recovery."""

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, checkpointer, sample_state):
        """Test creating a basic checkpoint."""
        checkpoint_id = await checkpointer.checkpoint(
            sample_state,
            CheckpointTrigger.TASK_START,
            "Test checkpoint"
        )

        assert checkpoint_id is not None
        assert "task_start" in checkpoint_id

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, checkpointer, sample_state):
        """Test restoring state from checkpoint."""
        checkpoint_id = await checkpointer.checkpoint(
            sample_state,
            CheckpointTrigger.TASK_START,
            "Test checkpoint"
        )

        restored = await checkpointer.restore(checkpoint_id)

        assert restored["current_phase"] == sample_state["current_phase"]
        assert restored["template_type"] == sample_state["template_type"]
        assert restored["_restored_from"] == checkpoint_id

    @pytest.mark.asyncio
    async def test_checkpoint_with_file_snapshot(self, checkpointer, temp_dir, sample_state):
        """Test checkpoint captures file snapshots."""
        # Create test files
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        (output_dir / "test.py").write_text("print('hello')")
        (output_dir / "config.json").write_text("{}")

        sample_state["output_dir"] = str(output_dir)

        checkpoint_id = await checkpointer.checkpoint(
            sample_state,
            CheckpointTrigger.PRE_RISKY,
            "Before risky operation",
            output_dir
        )

        # Verify snapshot was created
        checkpoint = checkpointer._checkpoints[checkpoint_id]
        assert checkpoint.metadata.snapshot_path is not None
        assert len(checkpoint.metadata.file_hashes) == 2

    @pytest.mark.asyncio
    async def test_rollback_restores_files(self, checkpointer, temp_dir, sample_state):
        """Test rollback restores file state."""
        # Create initial files
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        (output_dir / "test.py").write_text("original content")

        sample_state["output_dir"] = str(output_dir)

        # Create checkpoint
        checkpoint_id = await checkpointer.checkpoint(
            sample_state,
            CheckpointTrigger.PRE_RISKY,
            "Before change",
            output_dir
        )

        # Modify files
        (output_dir / "test.py").write_text("modified content")
        (output_dir / "new_file.py").write_text("new file")

        # Rollback
        await checkpointer.rollback(checkpoint_id, output_dir)

        # Verify restoration
        assert (output_dir / "test.py").read_text() == "original content"

    @pytest.mark.asyncio
    async def test_periodic_checkpoint(self, checkpointer, sample_state):
        """Test periodic checkpoint respects interval."""
        # First periodic checkpoint should be created
        checkpoint_id = await checkpointer.maybe_periodic_checkpoint(sample_state)
        assert checkpoint_id is not None

        # Immediate second call should not create checkpoint
        checkpoint_id_2 = await checkpointer.maybe_periodic_checkpoint(sample_state)
        assert checkpoint_id_2 is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer, sample_state):
        """Test listing checkpoints with filters."""
        await checkpointer.checkpoint(sample_state, CheckpointTrigger.TASK_START)
        await checkpointer.checkpoint(sample_state, CheckpointTrigger.PHASE_TRANSITION)
        await checkpointer.checkpoint(sample_state, CheckpointTrigger.ERROR)

        all_checkpoints = checkpointer.list_checkpoints()
        assert len(all_checkpoints) == 3

        error_checkpoints = checkpointer.list_checkpoints(trigger=CheckpointTrigger.ERROR)
        assert len(error_checkpoints) == 1

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, temp_dir, sample_state):
        """Test old checkpoints are cleaned up."""
        checkpointer = EnhancedCheckpointer(
            storage_dir=temp_dir / "checkpoints",
            max_checkpoints=3
        )

        # Create more checkpoints than max
        for i in range(5):
            await checkpointer.checkpoint(
                sample_state,
                CheckpointTrigger.PERIODIC,
                f"Checkpoint {i}"
            )
            await asyncio.sleep(0.01)  # Small delay for unique IDs

        assert len(checkpointer._checkpoints) <= 3


class TestCheckpointDecorator:
    """Tests for checkpoint_before_risky decorator."""

    @pytest.mark.asyncio
    async def test_decorator_creates_checkpoint(self, checkpointer, sample_state):
        """Test decorator creates checkpoint before operation."""
        @checkpoint_before_risky(checkpointer)
        async def risky_operation(state):
            return {"success": True}

        result = await risky_operation(sample_state)

        assert result == {"success": True}
        assert len(checkpointer._checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_decorator_checkpoints_on_error(self, checkpointer, sample_state):
        """Test decorator creates error checkpoint on failure."""
        @checkpoint_before_risky(checkpointer)
        async def failing_operation(state):
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_operation(sample_state)

        # Should have both pre-risky and error checkpoints
        error_checkpoints = checkpointer.list_checkpoints(trigger=CheckpointTrigger.ERROR)
        assert len(error_checkpoints) >= 1


# =============================================================================
# Test Distributed Lock Manager
# =============================================================================

class TestLockManager:
    """Tests for Distributed Lock Manager."""

    @pytest.mark.asyncio
    async def test_acquire_and_release_lock(self, lock_manager):
        """Test basic lock acquisition and release."""
        async with lock_manager.acquire(
            LockType.FILE,
            "test.py",
            "agent_1"
        ):
            # Lock should be held
            holder = lock_manager.who_holds(LockType.FILE, "test.py")
            assert holder == "agent_1"

        # Lock should be released
        holder = lock_manager.who_holds(LockType.FILE, "test.py")
        assert holder is None

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_access(self, lock_manager):
        """Test that lock prevents concurrent access."""
        acquired_order = []

        async def acquire_lock(agent_id: str):
            async with lock_manager.acquire(
                LockType.FILE,
                "shared.py",
                agent_id,
                wait_timeout=5.0
            ):
                acquired_order.append(agent_id)
                await asyncio.sleep(0.1)

        # Start both acquisitions
        await asyncio.gather(
            acquire_lock("agent_1"),
            acquire_lock("agent_2")
        )

        # Both should have acquired, one after the other
        assert len(acquired_order) == 2
        assert set(acquired_order) == {"agent_1", "agent_2"}

    @pytest.mark.asyncio
    async def test_lock_timeout(self, lock_manager):
        """Test lock acquisition timeout."""
        # First agent holds lock
        async with lock_manager.acquire(
            LockType.FILE,
            "test.py",
            "agent_1",
            timeout=timedelta(seconds=10)
        ):
            # Second agent should timeout
            with pytest.raises(TimeoutError):
                async with lock_manager.acquire(
                    LockType.FILE,
                    "test.py",
                    "agent_2",
                    wait_timeout=0.5
                ):
                    pass

    @pytest.mark.asyncio
    async def test_lock_expiration(self, lock_manager):
        """Test that expired locks are released."""
        async with lock_manager.acquire(
            LockType.FILE,
            "test.py",
            "agent_1",
            timeout=timedelta(milliseconds=100)
        ):
            await asyncio.sleep(0.2)  # Wait for expiration

        # Lock should be expired
        holder = lock_manager.who_holds(LockType.FILE, "test.py")
        assert holder is None

    def test_who_waits(self, lock_manager):
        """Test querying waiting agents."""
        # Initially no one waiting
        waiters = lock_manager.who_waits(LockType.FILE, "test.py")
        assert len(waiters) == 0

    @pytest.mark.asyncio
    async def test_get_locks_by_agent(self, lock_manager):
        """Test getting all locks held by an agent."""
        async with lock_manager.acquire(LockType.FILE, "file1.py", "agent_1"):
            async with lock_manager.acquire(LockType.DIRECTORY, "/src", "agent_1"):
                locks = lock_manager.get_locks_by_agent("agent_1")
                assert len(locks) == 2

    def test_detect_deadlock_no_deadlock(self, lock_manager):
        """Test deadlock detection with no deadlock."""
        cycle = lock_manager.detect_deadlock()
        assert cycle is None

    @pytest.mark.asyncio
    async def test_different_lock_types(self, lock_manager):
        """Test different lock types work independently."""
        async with lock_manager.acquire(LockType.FILE, "resource", "agent_1"):
            # Same resource ID but different type should work
            async with lock_manager.acquire(LockType.TASK, "resource", "agent_2"):
                file_holder = lock_manager.who_holds(LockType.FILE, "resource")
                task_holder = lock_manager.who_holds(LockType.TASK, "resource")

                assert file_holder == "agent_1"
                assert task_holder == "agent_2"


# =============================================================================
# Test Inter-Agent Communication
# =============================================================================

class TestMessageBus:
    """Tests for Inter-Agent Communication Channels."""

    @pytest.mark.asyncio
    async def test_register_and_unregister_agent(self, message_bus):
        """Test agent registration and unregistration."""
        await message_bus.register_agent("agent_1")
        assert "agent_1" in message_bus._queues

        await message_bus.unregister_agent("agent_1")
        assert "agent_1" not in message_bus._queues

    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_bus):
        """Test broadcasting message to all agents."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")
        await message_bus.register_agent("agent_3")

        # Broadcast from agent_1
        msg_id = await message_bus.broadcast(
            "agent_1",
            "Hello everyone",
            MessageType.NOTIFICATION
        )

        assert msg_id is not None

        # Other agents should receive
        msg_2 = await message_bus.receive("agent_2")
        msg_3 = await message_bus.receive("agent_3")

        assert msg_2 is not None
        assert msg_2.content == "Hello everyone"
        assert msg_3 is not None
        assert msg_3.content == "Hello everyone"

        # Sender should not receive own broadcast
        msg_1 = await message_bus.receive("agent_1", timeout=0.1)
        assert msg_1 is None

    @pytest.mark.asyncio
    async def test_direct_message(self, message_bus):
        """Test direct point-to-point messaging."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")
        await message_bus.register_agent("agent_3")

        # Send direct message
        await message_bus.send_direct(
            "agent_1",
            "agent_2",
            "Private message",
            MessageType.REQUEST
        )

        # Only agent_2 should receive
        msg_2 = await message_bus.receive("agent_2")
        msg_3 = await message_bus.receive("agent_3", timeout=0.1)

        assert msg_2 is not None
        assert msg_2.content == "Private message"
        assert msg_2.type == MessageType.REQUEST
        assert msg_3 is None

    @pytest.mark.asyncio
    async def test_task_scoped_channel(self, message_bus):
        """Test task-scoped channel messaging."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")
        await message_bus.register_agent("agent_3")

        # Subscribe some agents to task
        await message_bus.subscribe_to_task("agent_1", "task_123")
        await message_bus.subscribe_to_task("agent_2", "task_123")

        # Send to task channel
        await message_bus.send_to_task(
            "agent_1",
            "task_123",
            "Task update"
        )

        # Only subscribed agent_2 should receive (not sender)
        msg_2 = await message_bus.receive("agent_2")
        msg_3 = await message_bus.receive("agent_3", timeout=0.1)

        assert msg_2 is not None
        assert msg_2.content == "Task update"
        assert msg_2.task_id == "task_123"
        assert msg_3 is None

    @pytest.mark.asyncio
    async def test_receive_all_messages(self, message_bus):
        """Test receiving all pending messages."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")

        # Send multiple messages
        await message_bus.send_direct("agent_1", "agent_2", "Message 1")
        await message_bus.send_direct("agent_1", "agent_2", "Message 2")
        await message_bus.send_direct("agent_1", "agent_2", "Message 3")

        messages = await message_bus.receive_all("agent_2")

        assert len(messages) == 3
        assert messages[0].content == "Message 1"
        assert messages[2].content == "Message 3"

    @pytest.mark.asyncio
    async def test_message_history(self, message_bus):
        """Test querying message history."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")

        await message_bus.broadcast("agent_1", "Broadcast 1")
        await message_bus.send_direct("agent_1", "agent_2", "Direct 1", MessageType.REQUEST)
        await message_bus.send_direct("agent_2", "agent_1", "Direct 2", MessageType.RESPONSE)

        # Query all history
        all_history = message_bus.get_history()
        assert len(all_history) == 3

        # Query by sender
        agent1_history = message_bus.get_history(sender="agent_1")
        assert len(agent1_history) == 2

        # Query by type
        request_history = message_bus.get_history(msg_type=MessageType.REQUEST)
        assert len(request_history) == 1

    @pytest.mark.asyncio
    async def test_receive_with_filter(self, message_bus):
        """Test receiving messages with type filter."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")

        await message_bus.send_direct("agent_1", "agent_2", "Request", MessageType.REQUEST)

        # Try to receive only RESPONSE type
        msg = await message_bus.receive("agent_2", filter_type=MessageType.RESPONSE)
        assert msg is None  # Should not match

    @pytest.mark.asyncio
    async def test_unsubscribe_from_task(self, message_bus):
        """Test unsubscribing from task channel."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")

        await message_bus.subscribe_to_task("agent_2", "task_123")
        await message_bus.unsubscribe_from_task("agent_2", "task_123")

        await message_bus.send_to_task("agent_1", "task_123", "Message")

        msg = await message_bus.receive("agent_2", timeout=0.1)
        assert msg is None


# =============================================================================
# Test Multi-Agent Architecture
# =============================================================================

class TestMultiAgentArchitecture:
    """Tests for Six-Phase Multi-Agent Architecture."""

    def test_phase_type_enum(self):
        """Test PhaseType enum values."""
        assert PhaseType.EXPLORE.value == "explore"
        assert PhaseType.PLAN.value == "plan"
        assert PhaseType.CODE.value == "code"
        assert PhaseType.TEST.value == "test"
        assert PhaseType.FIX.value == "fix"
        assert PhaseType.DOCUMENT.value == "document"

    def test_default_agents_configuration(self):
        """Test default agent configurations exist for all phases."""
        for phase in PhaseType:
            assert phase in DEFAULT_AGENTS
            agents = DEFAULT_AGENTS[phase]
            assert len(agents) >= 1

            for agent in agents:
                assert isinstance(agent, AgentConfig)
                assert agent.phase == phase
                assert agent.name
                assert agent.role

    def test_agent_config_creation(self):
        """Test creating custom AgentConfig."""
        config = AgentConfig(
            name="custom_agent",
            role="Custom Role",
            description="A custom agent for testing",
            phase=PhaseType.CODE,
            tools=["tool1", "tool2"],
            system_prompt="You are a custom agent."
        )

        assert config.name == "custom_agent"
        assert config.phase == PhaseType.CODE
        assert len(config.tools) == 2

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["name"] == "custom_agent"
        assert config_dict["phase"] == "code"

    def test_phase_state_structure(self):
        """Test PhaseState structure."""
        state: PhaseState = {
            "phase": "explore",
            "agents": ["researcher", "analyzer"],
            "inputs": {"description": "Test project"},
            "outputs": {},
            "completed_agents": [],
            "pending_agents": ["researcher", "analyzer"],
            "errors": []
        }

        assert state["phase"] == "explore"
        assert len(state["agents"]) == 2

    @pytest.mark.asyncio
    async def test_create_phase_subgraph(self):
        """Test creating a phase subgraph."""
        subgraph = create_phase_subgraph(PhaseType.EXPLORE)

        assert subgraph is not None
        # The subgraph should be compiled

    @pytest.mark.asyncio
    async def test_create_orchestrated_workflow(self):
        """Test creating the main orchestrated workflow."""
        workflow = create_orchestrated_workflow()

        assert workflow is not None
        # The workflow should be compiled

    @pytest.mark.asyncio
    async def test_create_orchestrated_workflow_custom_phases(self):
        """Test creating workflow with custom phase order."""
        phases = [PhaseType.EXPLORE, PhaseType.PLAN, PhaseType.CODE]
        workflow = create_orchestrated_workflow(phases=phases)

        assert workflow is not None


class TestMultiAgentState:
    """Tests for MultiAgentState structure."""

    def test_multi_agent_state_structure(self, sample_state):
        """Test MultiAgentState has all required fields."""
        state: MultiAgentState = {
            **sample_state,
            "agent_states": {},
            "agent_outputs": {},
            "phase_results": {},
            "current_agents": [],
            "lock_manager_state": {},
            "message_bus_state": {},
        }

        assert "agent_states" in state
        assert "phase_results" in state
        assert "current_phase" in state


# =============================================================================
# Integration Tests
# =============================================================================

class TestOrchestrationIntegration:
    """Integration tests for the complete orchestration system."""

    @pytest.mark.asyncio
    async def test_lock_manager_with_message_bus(self, lock_manager, message_bus):
        """Test lock manager and message bus work together."""
        await message_bus.register_agent("agent_1")
        await message_bus.register_agent("agent_2")

        async def agent_with_locking(agent_id: str, resource: str):
            async with lock_manager.acquire(LockType.FILE, resource, agent_id):
                await message_bus.broadcast(
                    agent_id,
                    f"{agent_id} acquired lock on {resource}",
                    MessageType.STATUS
                )
                await asyncio.sleep(0.1)

        await asyncio.gather(
            agent_with_locking("agent_1", "shared_resource"),
            agent_with_locking("agent_2", "other_resource")
        )

        # Check message history
        history = message_bus.get_history(msg_type=MessageType.STATUS)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_checkpointer_with_context_preservation(self, checkpointer, sample_state):
        """Test checkpointer preserves context correctly."""
        # Add key decisions to state
        decision = record_decision(
            decision="Use template X",
            rationale="Best fit for requirements",
            alternatives=["Y", "Z"],
            agent_id="analyzer",
            phase="explore"
        )
        sample_state["key_decisions"] = [decision]

        # Create checkpoint
        checkpoint_id = await checkpointer.checkpoint(
            sample_state,
            CheckpointTrigger.PHASE_TRANSITION,
            "Transition to plan phase"
        )

        # Restore and verify
        restored = await checkpointer.restore(checkpoint_id)
        assert len(restored["key_decisions"]) == 1
        assert restored["key_decisions"][0]["decision"] == "Use template X"

    @pytest.mark.asyncio
    async def test_phase_handoff_with_checkpoint(self, checkpointer, sample_state):
        """Test phase handoff creates proper checkpoint."""
        # Create handoff context
        handoff = create_handoff_context(sample_state, "plan")

        # Checkpoint the handoff
        sample_state["phase_history"] = [handoff]
        checkpoint_id = await checkpointer.checkpoint(
            sample_state,
            CheckpointTrigger.PHASE_TRANSITION,
            f"Handoff: {handoff['from_phase']} -> {handoff['to_phase']}"
        )

        # Verify checkpoint contains handoff
        restored = await checkpointer.restore(checkpoint_id)
        assert len(restored["phase_history"]) == 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_restore_nonexistent_checkpoint(self, checkpointer):
        """Test restoring from non-existent checkpoint raises error."""
        with pytest.raises(KeyError):
            await checkpointer.restore("nonexistent_id")

    @pytest.mark.asyncio
    async def test_receive_from_unregistered_agent(self, message_bus):
        """Test receiving from unregistered agent returns None."""
        msg = await message_bus.receive("nonexistent_agent")
        assert msg is None

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_recipient(self, message_bus):
        """Test sending to non-existent recipient doesn't crash."""
        await message_bus.register_agent("sender")
        # Should not raise
        await message_bus.send_direct("sender", "nonexistent", "message")

    @pytest.mark.asyncio
    async def test_empty_phase_agents(self):
        """Test handling phases with no configured agents."""
        with pytest.raises(ValueError):
            create_phase_subgraph(PhaseType.EXPLORE, agent_configs=[])

    def test_handoff_without_phases(self):
        """Test handoff context validation without phase info."""
        context: PhaseHandoffContext = {
            "from_phase": "",  # Empty
            "to_phase": "",    # Empty
            "active_agents": [],
            "completed_work": ["work"],
            "pending_items": [],
            "key_decisions": [],
            "file_locations": {},
            "dependencies_discovered": [],
            "issues_requiring_attention": [],
            "explicit_next_steps": ["next"],
            "code_snippets": {},
            "external_references": [],
            "timestamp": datetime.now().isoformat(),
            "handoff_id": "test"
        }

        errors = validate_handoff_context(context)
        assert any("phase" in e for e in errors)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_message_bus_high_volume(self, message_bus):
        """Test message bus handles high message volume."""
        await message_bus.register_agent("sender")
        await message_bus.register_agent("receiver")

        # Send many messages
        for i in range(100):
            await message_bus.send_direct("sender", "receiver", f"Message {i}")

        # Receive all
        messages = await message_bus.receive_all("receiver")
        assert len(messages) == 100

    @pytest.mark.asyncio
    async def test_checkpoint_performance(self, checkpointer, sample_state):
        """Test checkpoint creation performance."""
        import time

        start = time.time()
        for _ in range(10):
            await checkpointer.checkpoint(
                sample_state,
                CheckpointTrigger.PERIODIC
            )
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0  # 10 checkpoints in under 5 seconds

    @pytest.mark.asyncio
    async def test_concurrent_lock_acquisition(self, lock_manager):
        """Test concurrent lock acquisition performance."""
        async def acquire_and_release(agent_id: str, resource_id: str):
            for _ in range(5):
                async with lock_manager.acquire(
                    LockType.FILE,
                    resource_id,
                    agent_id,
                    wait_timeout=10.0
                ):
                    await asyncio.sleep(0.01)

        # Multiple agents competing for same resource
        await asyncio.gather(
            acquire_and_release("agent_1", "shared"),
            acquire_and_release("agent_2", "shared"),
            acquire_and_release("agent_3", "shared"),
        )

        # All should complete without deadlock
        assert lock_manager.who_holds(LockType.FILE, "shared") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
